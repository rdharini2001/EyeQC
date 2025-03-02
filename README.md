# EyeQC
A Quality Assessment Tool in Oculomics

Available at - https://umnqws6lm48n7m3lgadvyz.streamlit.app/

![alt text](https://github.com/rdharini2001/EyeQC/blob/main/streamlit_UI.png)

Sample dataset - https://www.kaggle.com/datasets/linchundan/fundusimage1000

# Overview

This Image Quality Dashboard is a Streamlit-based application designed for quality assessment, preprocessing, and batch effect correction of retinal Fundus and OCTA images. It provides interactive visualizations and deep clinical insights by computing various image quality metrics, detecting segmentation failures, and analyzing batch effects.

# Features

1. Quality Metric Computation for OCTA and Fundus: Evaluates multiple image quality metrics like contrast, PSNR, entropy, blur, coefficient of variation etc.

2. Batch Effect Detection & Correction: Visualizes and corrects batch effects using PCA in 2D & 3D.

3. Interactive 3D Visualizations: Uses PCA, t-SNE, and UMAP for in-depth metric analysis.

4. Custom 3D Scatter & Correlation Analysis: Provides interactive 3D plots for metric comparison and correlation.

5. Static Visualizations: Includes scatter plots, violin plots, radar charts, and correlation heatmaps.

6. Preprocessing for Fundus & OCTA: Segment and preprocess images for further quality analysis.

# Installation

1. Clone the Repository

```
git clone https://github.com/rdharini2001/EyeQC.git
cd image-quality-dashboard
```

2. Install Dependencies
```
pip install -r requirements.txt
```

3. Run the Dashboard

```
streamlit run streamlit_app.py
```

# Usage

Upload Fundus or OCTA images via the sidebar.

# Choose Processing Mode

1. Adjustable Quality Metrics: Computes metrics and identifies pass/fail images.

2. Fundus and OCTA Preprocessing: Segments FAZ regions in OCTA and the central fundus alongside applying preprocessing algorithms.

3. Analyze Batch Effects: Check PCA clsuters before/after batch correction.

# Metrics 

1. Mean (Average intensity of fundus region)
   
- Definition: The mean pixel intensity within the detected fundus region.
- Significance: Provides an overall measure of the image’s brightness.

Clinical Interpretation:

- A low mean may indicate underexposed images (dark images).
- A high mean may suggest overexposure, washing out important details.
- Optimal fundus images should have a balanced mean intensity to retain both bright and dark regions for proper diagnosis.

2. Variance (Spread of intensity values)
   
- Definition: Measures how much the pixel intensities deviate from the mean.
- Significance: Indicates contrast and image detail distribution.
  
Clinical Interpretation:

- Low variance suggests a flat or low-contrast image, which may lack important details.
- High variance suggests strong contrast, which may improve visibility of vessels and lesions but could also indicate overexposure in some cases.
  
3. Range (Intensity range)
   
- Definition: The difference between the maximum and minimum pixel intensity in the fundus region.
- Significance: Indicates the dynamic range of brightness levels in the image.
  
Clinical Interpretation:

- A high range suggests strong variations in brightness, which is good for differentiation of structures.
- A low range may indicate a washed-out image (either too bright or too dark), making it difficult to distinguish fine details.
  
4. Coefficient of Variation (CV) (%)
   
- Definition: The ratio of the standard deviation to the mean intensity, expressed as a percentage.
- Significance: Measures the relative variation of intensities.
  
Clinical Interpretation:

- A high CV suggests that the image has regions of high contrast.
- A low CV suggests a more uniform image, which could be good or bad depending on whether the uniformity hides important details.
  
5. Shannon Entropy (Detail richness)
   
- Definition: Measures the amount of information content in the image.
- Significance: Higher entropy suggests higher texture richness and better image details.
  
Clinical Interpretation:

- High entropy is desirable as it indicates a well-detailed image with clear textures.
- Low entropy suggests that the image is blurry or low contrast, making it difficult to analyze fine retinal structures.
  
6-9. Signal-to-Noise Ratio (SNR) Variants

These four SNR variants measure different aspects of signal quality vs. background noise.

6. SNR1: Signal-to-Noise Ratio (Foreground vs. Background)
   
- Definition: Ratio of the foreground region’s standard deviation to the background’s standard deviation.
- Significance: Higher values indicate better contrast between the fundus and background.
  
Clinical Interpretation:

- High SNR1 → Clear separation between fundus and background, better diagnostic quality.
- Low SNR1 → The fundus is difficult to distinguish from the background.

7. SNR2: Mean Intensity of Fundus vs. Background Noise
   
- Definition: Ratio of mean intensity of the fundus region to the background standard deviation.
- Significance: Indicates how well the fundus stands out compared to the surrounding noise.
  
Clinical Interpretation:

- High SNR2 → Good contrast and visibility.
- Low SNR2 → Background noise may be affecting visibility.
  
8. SNR3: Mean Intensity of Fundus vs. Standard Deviation of Fundus
   
- Definition: Ratio of mean fundus intensity to the fundus’ own standard deviation.
- Significance: Measures the contrast within the fundus region.
  
Clinical Interpretation:

- High SNR3 → The fundus has consistent brightness, which may be good for some applications but may also indicate over-smoothing.
- Low SNR3 → Variability within the fundus is high, which could indicate poor quality.
  
9. SNR4: Standard Deviation of Fundus vs. Mean Background Intensity

- Definition: Ratio of fundus standard deviation to mean background intensity.
- Significance: A high value suggests strong contrast between the fundus and background.
  
Clinical Interpretation:

- High SNR4 → Good differentiation of fundus from background.
- Low SNR4 → The background might be interfering with the fundus details.
  
10. Contrast-to-Noise Ratio (CNR)
    
- Definition: Measures the difference in intensity between the fundus region and background noise.
- Significance: A high CNR suggests a clear distinction between the fundus and background.
  
Clinical Interpretation:

- High CNR → Good-quality images with clear contrast.
- Low CNR → The fundus is hard to distinguish, which could indicate a poor-quality image.

11. Peak Signal-to-Noise Ratio (PSNR)
    
- Definition: Measures the ratio between the maximum possible intensity and the noise level.
- Significance: Higher PSNR means less noise and better image quality.
  
Clinical Interpretation:

- High PSNR → The image has low noise and good sharpness.
- Low PSNR → The image is noisy or compressed, leading to loss of fine details.
  
12. Foreground Contrast
    
- Definition: Measures the contrast within the fundus region.
- Significance: Contrast is essential for detecting retinal abnormalities.
  
Clinical Interpretation:

- High contrast → Clear details of the retinal structures.
- Low contrast → Retinal blood vessels or lesions may not be visible.
  
13. Blur (Laplacian Variance)
    
- Definition: Measures how much high-frequency detail (sharp edges) is present in the image using the Laplacian operator.
- Significance: Lower values indicate more blurring.
  
Clinical Interpretation:

- High blur value → The image is sharp and likely of good quality.
- Low blur value → The image is blurry, making clinical assessment difficult.
  
14. Foreground Area (Fraction of image as fundus)
    
- Definition: Measures how much of the image contains fundus pixels.
- Significance: Ensures that enough of the fundus is captured.
  
Clinical Interpretation:

- High foreground area → Good coverage of the retina.
- Low foreground area → The image might be cropped or poorly framed.
  
15. Edge Density (Edge density in fundus region)
    
- Definition: Measures the density of edges detected in the fundus region.
- Significance: Higher values indicate more fine details.
  
Clinical Interpretation:

- High edge density → Sharp retinal vessels and structures.
- Low edge density → The image may be blurry or low-contrast.

# Contributing

Contributions are welcome! Please follow these steps:

Fork the repository.

Create a new branch: git checkout -b feature-branch.

Commit changes: git commit -m 'Added a new feature'.

Push to GitHub: git push origin feature-branch.

Submit a Pull Request.

# License

This project is licensed under the MIT License. See LICENSE for details.
