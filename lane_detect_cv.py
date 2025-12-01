import cv2
import numpy as np

# ============ GLOBALS FOR TEMPORAL SMOOTHING ============
SMOOTHING_ALPHA = 0.7  # 0–1, higher = more weight on current frame

prev_left_coeffs = None
prev_right_coeffs = None

def smooth_coeffs(new, prev, alpha=SMOOTHING_ALPHA):
    """
    Exponential moving average on polynomial coefficients.
    new, prev: np.array of shape (3,) or None
    """
    if new is None:
        return prev
    if prev is None:
        return new
    return prev * (1 - alpha) + new * alpha

# ============ ROI ============
def region_of_interest(img):
    """
    Mask everything except a polygon covering the lower part of the frame
    (road area). Tuned for 1280x720-ish driving videos.
    """
    height, width = img.shape[:2]

    polygon = np.array([[
        (int(0.10 * width), height),
        (int(0.45 * width), int(0.55 * height)),
        (int(0.60 * width), int(0.55 * height)),
        (int(0.90 * width), height)
    ]], dtype=np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# ============ POLY FIT + TEMPORAL SMOOTHING ============

def average_slope_intercept(image, lines):
    """
    Fit a 2nd order polynomial separately for left and right lanes,
    and smooth coefficients across frames to reduce jitter.
    Includes:
      - Fix E: stricter slope gate for right lane
      - Fix F: geometric constraint on right lane position
    """
    global prev_left_coeffs, prev_right_coeffs

    height, width = image.shape[:2]

    left_points = []
    right_points = []

    # ---- If no lines this frame, reuse previous polynomials ----
    if lines is None:
        lane_lines = []

        def build_from_coeffs(coeffs):
            if coeffs is None:
                return
            a, b, c = coeffs
            y_min = int(height * 0.6)
            y_max = height
            ys = np.linspace(y_max, y_min, num=30)
            xs = a * ys**2 + b * ys + c
            curve = []
            for x, y in zip(xs, ys):
                xi, yi = int(x), int(y)
                if 0 <= xi < width:
                    curve.append([xi, yi])
            if len(curve) >= 2:
                for i in range(len(curve) - 1):
                    x1, y1 = curve[i]
                    x2, y2 = curve[i + 1]
                    lane_lines.append([x1, y1, x2, y2])

        build_from_coeffs(prev_left_coeffs)
        build_from_coeffs(prev_right_coeffs)

        return np.array(lane_lines) if lane_lines else None

    # ---- Collect all left/right points from Hough segments ----
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue  # skip vertical lines

        slope = (y2 - y1) / (x2 - x1)

        # Fix E: use different gates for left/right instead of abs(slope)<0.3
        # left = clearly negative, right = clearly positive
        if slope < -0.3:               # left lane candidates
            left_points.append((x1, y1))
            left_points.append((x2, y2))
        elif slope > 0.3:              # right lane candidates (ignore shallow noisy edges)
            right_points.append((x1, y1))
            right_points.append((x2, y2))
        else:
            # slopes between -0.3 and +0.3 are treated as noise
            continue

    def fit_poly(points):
        if len(points) < 6:  # need enough points for a stable fit
            return None
        pts = np.array(points)
        X = pts[:, 0]
        Y = pts[:, 1]
        # Fit x as function of y: x = a*y**2 + b*y + c
        coeffs = np.polyfit(Y, X, 2)
        return coeffs

    # Raw polynomial fits this frame
    left_coeffs_new = fit_poly(left_points)
    right_coeffs_new = fit_poly(right_points)

    # -------- Fix F: geometric constraint on RIGHT lane position --------
    if right_coeffs_new is not None:
        y_test = height
        x_new = (
            right_coeffs_new[0] * y_test**2
            + right_coeffs_new[1] * y_test
            + right_coeffs_new[2]
        )
        # right lane bottom x must stay in right half, not too close to border
        if x_new > width * 0.95 or x_new < width * 0.50:
            # if we have a previous stable right lane, fall back to it
            right_coeffs_new = prev_right_coeffs if prev_right_coeffs is not None else None

    # ---- Optional: reject crazy jumps (safety net) ----
    def reject_if_jump(new, prev, max_delta=80):
        if new is None or prev is None:
            return new
        y_test = height
        x_new = new[0] * y_test**2 + new[1] * y_test + new[2]
        x_prev = prev[0] * y_test**2 + prev[1] * y_test + prev[2]
        if abs(x_new - x_prev) > max_delta:
            return prev
        return new

    left_coeffs_new = reject_if_jump(left_coeffs_new, prev_left_coeffs)
    right_coeffs_new = reject_if_jump(right_coeffs_new, prev_right_coeffs)

    # ---- Temporal smoothing (EMA) ----
    left_coeffs_smooth = smooth_coeffs(left_coeffs_new, prev_left_coeffs)
    right_coeffs_smooth = smooth_coeffs(right_coeffs_new, prev_right_coeffs)

    # Update global state
    prev_left_coeffs = left_coeffs_smooth
    prev_right_coeffs = right_coeffs_smooth

    # ---- Build lane line segments from smoothed polynomials ----
    lane_lines = []
    y_min = int(height * 0.6)
    y_max = height

    def add_curve_segments(coeffs):
        nonlocal lane_lines
        if coeffs is None:
            return
        a, b, c = coeffs
        ys = np.linspace(y_max, y_min, num=30)
        xs = a * ys**2 + b * ys + c
        curve = []
        for x, y in zip(xs, ys):
            xi, yi = int(x), int(y)
            if 0 <= xi < width:
                curve.append([xi, yi])
        if len(curve) >= 2:
            for i in range(len(curve) - 1):
                x1, y1 = curve[i]
                x2, y2 = curve[i + 1]
                lane_lines.append([x1, y1, x2, y2])

    add_curve_segments(left_coeffs_smooth)
    add_curve_segments(right_coeffs_smooth)

    if not lane_lines:
        return None

    return np.array(lane_lines)

# ============ DRAWING ============

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 8)
    return line_image

# ============ PER-FRAME PIPELINE ============

def process_frame(frame):
    """
    Full pipeline for one frame: grayscale -> blur -> edges -> ROI ->
    Hough -> poly fit + smoothing -> draw over original frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # mask for green (grass, trees)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # invert mask -> keep everything NOT green
    non_green = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(green_mask))
    gray = cv2.cvtColor(non_green, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    cropped_edges = region_of_interest(edges)

    # Hough transform tuned for curved lanes
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=25,
        maxLineGap=60
    )

    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combo_image

# ============ VIDEO LOOP ============

def infer_video(input_path="input.mp4", output_path="output_lanes.mp4"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # fallback

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {input_path} -> {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")

    warmup_frames = 10
    frame_count = 0

    print(f"Warm-up: processing first {warmup_frames} frames without output...")

    while frame_count < warmup_frames:
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)   # update smoothing history, but don't draw/write
        frame_count += 1

    # Reset video to start after warm-up
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    print("Warm-up complete. Starting real processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(frame)
        out.write(processed)
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Done. Total frames: {frame_count}")
    print(f"Saved video: {output_path}")

if __name__ == "__main__":
    infer_video("input.mp4", "output_lanes.mp4")
