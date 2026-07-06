# 2D Baseline Correction Methods & Potential 2D FLATT Improvements

This document summarizes the current 2D baseline correction methods implemented in deep-picker and details five potential improvements for extending the first step of the FLATT algorithm (peak and water region classification) from 1D to true 2D.

---

## 1. Current Implementation in fid_2d.cpp

Baseline correction in 2D spectra is handled by the function fid_2d::polynorminal_baseline (in fid_2d.cpp), which operates in two distinct modes depending on the b_trace_by_trace parameter:

### A. Trace-by-Trace Mode (b_trace_by_trace == true)
* **Default FLATT Method (order == -1)**:
  * Loops over each 1D trace i along the outer dimension n_outer_dim.
  * Invokes fid_2d_math::flatt_baseline_1d on each trace independently.
  * Parameters: Uses window size 17, noise level noise_level, scaling factor -1.0, and user-definable parameters flatt_a and flatt_b.
* **Trace-by-Trace Polynomial (order >= 0)**:
  * Invokes fid_2d_math::polynomial_baseline_1d along each row/trace independently with the specified order (up to 4th order).

### B. 2D Polynomial Surface Fitting Mode (b_trace_by_trace == false)
* **Grid Downsampling**:
  * Downsamples the 2D spectrum matrix into a coarse grid of size n_dim1 x n_dim2 (default starting dimensions are 4 x 64).
  * Grid points are sampled at regular step intervals (step1 and step2).
* **Surface Fitting**:
  * Calls fid_2d_math::polynomial_baseline on the downsampled grid to estimate baseline parameters and identify baseline-selected points (excluding peak regions).
  * Fits a 2D polynomial surface of order O (0 <= O <= 4) without cross terms (e.g., no i * j term, which is standard in NMR processing).
  * Subtracts the calculated baseline surface from the entire spectrum.

---

## 2. Five Potential Improvements for 2D FLATT (Step 1 Region Detection)

Step 1 of the FLATT algorithm classifies the spectrum into positive peaks, negative peaks, and solvent (water) regions before solving the baseline optimization problem. When moving from 1D (trace-by-trace) to true 2D, five potential methods have been proposed:

### Method 1: Cross-Directional 1D FLATT Consensus (Hybrid 1D/2D)
* **Algorithm**: 
  * Run the existing 1D FLATT along the rows (F2 dimension) to get row-wise masks: M_row_pos, M_row_neg, and M_row_water.
  * Run 1D FLATT along the columns (F1 dimension) to get column-wise masks: M_col_pos, M_col_neg, and M_col_water.
  * Merge them using consensus logic:
    * 2D Positive Peak Mask: M_row_pos AND M_col_pos (Intersection avoids false-positive noise lines).
    * 2D Negative Peak Mask: M_row_neg AND M_col_neg.
    * 2D Water Mask: M_row_water OR M_col_water (Union ensures broad solvent ridges are fully covered).
* **Pros**: Simple to implement, computationally very fast, and directly reuses the existing, optimized 1D C++ solver.
* **Cons**: Approximates 2D topology using 1D slices; might miss diagonally oriented peaks.

### Method 2: Local 2D Surface Regression (Bilinear/Planar Fitting)
* **Algorithm**: 
  * Replace the 1D sliding window of length W with a 2D sliding window of size W1 x W2 (e.g., 17 x 17).
  * Fit a local plane z(x, y) = c0 + cx*(x - xc) + cy*(y - yc) centered at (xc, yc) using Ordinary Least Squares.
  * Define Residual Variance: R = sqrt(1/(N-1) * sum((z_data - z_fit)^2)). If R >= 4.0 * noise, the region contains active signal.
  * Identify peak regions based on gradient vector g = (cx, cy) direction sign transitions across neighboring 2D windows.
* **Pros**: Models local 2D gradients physically and filters isotropic high-frequency noise.
* **Cons**: Local plane fitting for every pixel in a large 2D matrix can be computationally expensive.

### Method 3: 2D Connected Component Sign-Topology Analysis
* **Algorithm**: 
  * Threshold the spectrum using local variance or gradient magnitude to find "active" pixels.
  * Group active pixels into blobs using 2D Connected Component Analysis (CCA).
  * Analyze the spatial distribution of the partial derivatives dz/dx and dz/dy across each blob:
    * Positive Peak: Gradient field points inward toward the center (horizontal sign goes + to - left-to-right, vertical sign goes + to - top-to-bottom).
    * Negative Peak: Gradient field points outward (signs are reversed).
    * Water Region: Multi-oscillatory signs (multiple zero-crossings) or very large footprint.
* **Pros**: Elegant extension of 1D sign-transition logic to 2D topological features.
* **Cons**: Needs robust handling of overlapping peaks (multiplets) that merge into single large blobs.

### Method 4: Curvature Analysis via Hessian Eigenvalues
* **Algorithm**: 
  * Fit a local quadratic surface z(x, y) = c0 + cx*x + cy*y + cxx*x^2 + cyy*y^2 + 2*cxy*x*y.
  * Construct the local Hessian matrix at each pixel:
    H = [ 2*cxx, 2*cxy; 2*cxy, 2*cyy ]
  * Calculate the eigenvalues L1, L2 (representing principal curvatures):
    * Positive Peak (Dome): L1 << 0 and L2 << 0.
    * Negative Peak (Bowl): L1 >> 0 and L2 >> 0.
    * Solvent Ridge: L1 approx 0 and L2 << 0 (curved in one direction, flat/elongated in the other).
    * Empty Region: L1 approx 0 and L2 approx 0.
* **Pros**: Coordinate-invariant, highly robust, mathematically rigorous, and handles elongated diagonal ridges naturally.
* **Cons**: Fitting quadratic surfaces is computationally demanding.

### Method 5: 2D Morphological Post-Processing
* **Algorithm**: 
  * Run the fast 1D FLATT classification (trace-by-trace) to get initial row-wise peak and solvent masks.
  * Replace the 1D expansion/cleaning steps with 2D mathematical morphology:
    * Dilation: Dilate negative peak masks using a 2D disk-shaped structuring element of radius R to expand them isotropically in all directions.
    * Opening & Closing: Use 2D opening to remove isolated single-pixel noise, and 2D closing to bridge fractured water regions.
* **Pros**: Standard, highly efficient (custom sweeps or standard morphology kernels), and preserves 2D geometric shapes.
* **Cons**: Requires tuning the disk radius R based on the digital resolution (points per Hz).

---

## 3. Comparison of 2D Expansion Methods

| Method | Complexity | Computational Cost | Robustness | Reusability of 1D Code |
| :--- | :--- | :--- | :--- | :--- |
| 1. Cross-Directional Consensus | Low | Low | Medium | High |
| 2. Local 2D Surface Regression | Medium | Medium-High | High | Low |
| 3. Connected Component Topology | High | Medium | High | Low |
| 4. Hessian Curvature Analysis | High | High | Very High | Low |
| 5. 2D Morphological Post-Processing | Low | Low | High | Medium |

---

## 4. Current Region Classification Cutoffs (flatt_identify_regions_1d)

The following parameter thresholds and logic rules are currently used to classify regions (positive peaks, negative peaks, water, and empty regions):

1. **Empty / No Signal Region (Value 0)**:
   * A data point is classified as empty if both the local variance and slope fall below their respective thresholds.
2. **Residual Variance Threshold**:
   * Calculated over a sliding window of size $W = 17$.
   * A region is considered to contain active signal if the local standard deviation is $\ge 5.5 \times \text{noise\_level}$ (increased from the original `4.0`).
3. **Slope Threshold**:
   * Set to $4.0 \times \text{noise\_level}$, which effectively requires the fitted difference between neighboring points on the trace to exceed this value to be considered active.
4. **Water Region Definition**:
   * Relaxed: Any active block containing **3 or more segments** (`num_segments >= 3`) is classified as a water/solvent region, dropping the original matching boundary sign constraint.
5. **Peak Slope Segment Length Limit**:
   * Reduced: Simple peaks (which have exactly 2 segments) must have at least **2 points** in each slope segment (both positive and negative slopes must span at least 2 points to be classified as a peak).
