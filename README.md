# Classical Lane Detection on Curved Forest Roads

This repository contains a **real-time lane detection pipeline** built entirely with **classical computer vision** (no deep learning).  
The system takes a road video as input (e.g. 30-second dashcam clip) and outputs a new video with **smooth green lane overlays** on both sides of the road.

The pipeline is designed and tuned for **narrow, curved forest roads** with strong green surroundings (trees/grass) and clear road edges.

---

## Project Overview

**Goal:**  
Detect left and right lane boundaries from a front-facing camera video and draw stable, curved lane lines frame-by-frame.

**Key properties:**

- No training / no neural networks  
- Runs in real time on CPU  
- Handles curved lanes with polynomial fitting  
- Uses temporal smoothing to reduce jitter  
- ⚠️ Optimized for a specific camera view on a small road

---

## Core Pipeline (What “model” is used)

This project uses a **deterministic CV pipeline**, not a learned model.

For each frame:

1. **Color Filtering (optional but used in final version)**  
   - Convert frame to HSV.  
   - Suppress **green regions** (trees, grass) so that edges on vegetation are ignored.

2. **Grayscale + Blur**  
   - Convert to grayscale.  
   - Apply Gaussian blur to reduce noise.

3. **Canny Edge Detection**  
   - Detect high-contrast edges that correspond to lane markings / road boundaries.

4. **Region of Interest (ROI) Mask**  
   - Apply a trapezoid mask focusing on the **lower center** of the frame (where the road lies).  
   - This removes sky, trees, and most off-road clutter.

5. **Probabilistic Hough Transform (`cv2.HoughLinesP`)**  
   - Detect multiple short **line segments** from the edge image.  
   - Filter segments by slope:
     - Negative slopes → left-lane candidates  
     - Positive slopes → right-lane candidates  
     - Very shallow slopes are discarded as noise.

6. **Quadratic Polynomial Fitting (Per Lane)**  
   - Collect all left-lane points and fit a **2nd-order polynomial**:  
     `x_left(y) = a*y^2 + b*y + c`  
   - Do the same for the right lane.  
   - Sample these polynomials to create smooth curved lane lines.

7. **Temporal Smoothing (Exponential Moving Average)**  
   - Smooth polynomial coefficients over time:  
     `coeff_smooth = α * coeff_new + (1 - α) * coeff_prev`  
   - This stabilizes the lanes and removes frame-to-frame jitter.

8. **Geometric Constraints**  
   - Reject right-lane fits that jump too far horizontally or move outside a valid x-range.  
   - Fallback to previous frame’s lane if a new estimate is inconsistent.

9. **Rendering**  
   - Draw the smoothed lane curves as green polylines on top of the original frame.  
   - Write the processed frames into an output video.

---

## 📂 Repository Structure

```text
.
├── lane_detect_cv.py      # Main inference script (classical CV pipeline)
├── README.md              # Project documentation
└── (optional) notebooks/  # Colab / Jupyter notebooks (if you add them)
