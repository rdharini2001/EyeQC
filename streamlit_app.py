#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from skimage.measure import shannon_entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image, ImageFile
import plotly.express as px
import plotly.graph_objects as go

# Optional: UMAP is used if available
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Optional: neuroCombat (not used in this version)
try:
    import neuroCombat
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------------
# CONFIGURATION
# ------------------------------
st.set_page_config(page_title="Image Quality Dashboard", layout="wide")

# Output directories
OUTPUT_DIR = "results"
PASSED_DIR = os.path.join(OUTPUT_DIR, "passed_images")
FAILED_DIR = os.path.join(OUTPUT_DIR, "failed_images")
PREPROCESS_DIR = os.path.join(OUTPUT_DIR, "preprocessed_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PASSED_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)
os.makedirs(PREPROCESS_DIR, exist_ok=True)

# ------------------------------
# QUALITY METRIC DEFINITIONS
# ------------------------------
ALL_METRICS = {
    "mean": "Average intensity of region",
    "var": "Variance (spread of intensity values)",
    "range": "Intensity range",
    "cv": "Coefficient of variation (%)",
    "entropy": "Shannon entropy (detail richness)",
    "snr1": "SNR variant #1",
    "snr2": "SNR variant #2",
    "snr3": "SNR variant #3",
    "snr4": "SNR variant #4",
    "cnr": "Contrast-to-noise ratio",
    "psnr": "Peak Signal-to-Noise Ratio",
    "contrast": "Foreground contrast",
    "blur": "Blur (Laplacian variance)",
    "foreground_area": "Fraction of image as region",
    "edge_density": "Edge density in region",
}

ALL_METRIC_THRESHOLDS = {
    "mean": (30, 240),
    "var": (50, float("inf")),
    "range": (30, float("inf")),
    "cv": (0, 100),
    "entropy": (3, float("inf")),
    "snr1": (1, float("inf")),
    "snr2": (5, float("inf")),
    "snr3": (3, float("inf")),
    "snr4": (1, float("inf")),
    "cnr": (0.5, float("inf")),
    "psnr": (18, float("inf")),
    "contrast": (0.15, float("inf")),
    "blur": (20, float("inf")),
    "foreground_area": (0.5, float("inf")),
    "edge_density": (0.05, 0.5)
}

DEFAULT_CRITERIA = ["contrast", "psnr", "entropy", "blur"]

# Color mapping for quality groups
COLOR_MAP = {"Low": "red", "Medium": "yellow", "High": "green", "Failed": "gray"}

# ------------------------------
# LOCAL METRIC FUNCTIONS FOR HEATMAPS
# ------------------------------
LOCAL_METRIC_FUNCS = {
    "mean": lambda patch: np.mean(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)),
    "var": lambda patch: np.var(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)),
    "range": lambda patch: np.ptp(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)),
    "cv": lambda patch: (np.std(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)) / np.mean(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)) * 100 
                         if np.mean(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)) != 0 else 0),
    "entropy": lambda patch: shannon_entropy(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)),
    "contrast": lambda patch: (float(np.max(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY))) - float(np.min(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY))))
                              / max((float(np.max(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY))) + float(np.min(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)))), 1e-6),
    "blur": lambda patch: np.var(cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY), cv2.CV_64F)),
    "edge_density": lambda patch: np.count_nonzero(cv2.Canny(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY), 100, 200)) / (patch.shape[0]*patch.shape[1])
}

# ------------------------------
# FUNDUS & OCTA PREPROCESSING FUNCTIONS
# ------------------------------
def preprocess_fundus_image(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb, np.zeros_like(mask)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)[y:y+h, x:x+w]
    return cropped_img, mask

# ------------------------------
# FUNDUS QUALITY METRIC FUNCTIONS
# ------------------------------
def detect_foreground_fundus(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def compute_edge_density(img_bgr, mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    fg_edges = edges[mask == 255]
    return np.count_nonzero(fg_edges) / fg_edges.size if fg_edges.size > 0 else -1

def compute_quality_metrics(img_bgr, mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fg_pixels = gray[mask == 255]
    bg_pixels = gray[mask == 0]
    if fg_pixels.size == 0 or bg_pixels.size == 0:
        return {m: -1 for m in ALL_METRICS.keys()}
    mean_val = np.mean(fg_pixels)
    var_val = np.var(fg_pixels)
    range_val = np.ptp(fg_pixels)
    std_val = np.std(fg_pixels)
    cv_val = (std_val / mean_val) * 100 if mean_val != 0 else -1
    entropy_val = shannon_entropy(fg_pixels)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_val = np.var(lap[mask == 255])
    # Convert min and max to float to avoid uint8 overflow
    min_val, max_val = float(np.min(fg_pixels)), float(np.max(fg_pixels))
    contrast_val = (max_val - min_val) / max((max_val + min_val), 1e-6)
    mse = np.mean((fg_pixels - mean_val) ** 2)
    psnr_val = 10 * np.log10(255**2 / max(mse, 1e-6))
    fg_std = np.std(fg_pixels)
    bg_std = np.std(bg_pixels)
    fg_mean = mean_val
    bg_mean = np.mean(bg_pixels)
    snr1_val = fg_std / max(bg_std, 1e-6)
    snr2_val = fg_mean / max(bg_std, 1e-6)
    snr3_val = fg_mean / max((fg_std - fg_mean), 1e-6)
    snr4_val = fg_std / max(bg_mean, 1e-6)
    cnr_val = (fg_mean - bg_mean) / max(bg_std, 1e-6)
    foreground_area_val = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
    edge_density_val = compute_edge_density(img_bgr, mask)
    return {
        "mean": mean_val,
        "var": var_val,
        "range": range_val,
        "cv": cv_val,
        "entropy": entropy_val,
        "snr1": snr1_val,
        "snr2": snr2_val,
        "snr3": snr3_val,
        "snr4": snr4_val,
        "cnr": cnr_val,
        "psnr": psnr_val,
        "contrast": contrast_val,
        "blur": blur_val,
        "foreground_area": foreground_area_val,
        "edge_density": edge_density_val
    }

def normalize_df(df):
    norm_df = df.copy()
    for col in ALL_METRICS.keys():
        if col in norm_df.columns:
            valid = norm_df[col] != -1
            if valid.sum() > 1:
                mn = norm_df.loc[valid, col].min()
                mx = norm_df.loc[valid, col].max()
                if mx > mn:
                    norm_df.loc[valid, col] = (norm_df.loc[valid, col] - mn) / (mx - mn)
                else:
                    norm_df.loc[valid, col] = 0
            else:
                norm_df[col] = 0
    return norm_df

# ------------------------------
# SIMPLE BATCH CORRECTION FUNCTION
# ------------------------------
def simple_batch_correct_metrics(df, batch_col="batch"):
    df_corrected = df.copy()
    numeric_cols = list(ALL_METRICS.keys())
    overall_means = df[numeric_cols].mean()
    overall_stds = df[numeric_cols].std()
    for b in df[batch_col].unique():
        idx = df[batch_col] == b
        batch_means = df.loc[idx, numeric_cols].mean()
        batch_stds = df.loc[idx, numeric_cols].std()
        for col in numeric_cols:
            if batch_stds[col] == 0:
                df_corrected.loc[idx, col] = overall_means[col]
            else:
                df_corrected.loc[idx, col] = ((df.loc[idx, col] - batch_means[col]) /
                                              batch_stds[col] * overall_stds[col] +
                                              overall_means[col])
    return df_corrected

# ------------------------------
# BATCH EFFECTS VISUALIZATION (2D)
# ------------------------------
def plot_batch_effects(df, corrected=False):
    plot_df = df.copy()
    title_suffix = " (Corrected)" if corrected else " (Original)"
    st.markdown(f"#### PCA {title_suffix}")
    pca_model = PCA(n_components=2)
    features = plot_df[list(ALL_METRICS.keys())].fillna(0).values
    pcs = pca_model.fit_transform(features)
    plot_df["PC1"] = pcs[:, 0]
    plot_df["PC2"] = pcs[:, 1]
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="batch",
                     title=f"Batch Effects{title_suffix}", template="plotly_white")
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(legend_font_size=14)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Clinical Insight:**  
    Batch effects indicate systematic differences between acquisition sessions.
    After correction, images from the same class should group together.
    """)

def plot_metric_distributions(df):
    st.markdown("### Quality Metric Distributions by Batch")
    for metric in ALL_METRICS.keys():
        if metric in df.columns:
            fig = px.box(df, x="batch", y=metric, title=f"{metric} distribution by Batch", template="plotly_white")
            fig.update_traces(marker=dict(size=12))
            fig.update_layout(legend_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# BATCH EFFECTS VISUALIZATION (3D)
# ------------------------------
def batch_effects_3d(df):
    st.markdown("## Batch Effects in 3D (PCA)")
    # 3D PCA before correction
    st.markdown("### Before Correction")
    pca_model = PCA(n_components=3)
    features = df[list(ALL_METRICS.keys())].fillna(0).values
    pcs = pca_model.fit_transform(features)
    df_3d = df.copy()
    df_3d["PC1"], df_3d["PC2"], df_3d["PC3"] = pcs[:, 0], pcs[:, 1], pcs[:, 2]
    fig_before = px.scatter_3d(df_3d, x="PC1", y="PC2", z="PC3", color="batch",
                               title="3D PCA Before Batch Correction", template="plotly_white")
    fig_before.update_traces(marker=dict(size=12))
    fig_before.update_layout(legend_font_size=14)
    st.plotly_chart(fig_before, use_container_width=True)
    
    # 3D PCA after correction
    df_corrected = simple_batch_correct_metrics(df, batch_col="batch")
    pca_model2 = PCA(n_components=3)
    features_corr = df_corrected[list(ALL_METRICS.keys())].fillna(0).values
    pcs_corr = pca_model2.fit_transform(features_corr)
    df_corr_3d = df_corrected.copy()
    df_corr_3d["PC1"], df_corr_3d["PC2"], df_corr_3d["PC3"] = pcs_corr[:, 0], pcs_corr[:, 1], pcs_corr[:, 2]
    st.markdown("### After Correction")
    fig_after = px.scatter_3d(df_corr_3d, x="PC1", y="PC2", z="PC3", color="batch",
                              title="3D PCA After Batch Correction", template="plotly_white")
    fig_after.update_traces(marker=dict(size=12))
    fig_after.update_layout(legend_font_size=14)
    st.plotly_chart(fig_after, use_container_width=True)

    st.markdown("""
    **Clinical Insight:**  
    The 3D PCA plots show the data distribution before and after batch correction.
    After correction, images from the same class (batch) should form tighter clusters.
    """)

# ------------------------------
# FAILURE EXPLANATION (ALL METRICS)
# ------------------------------
def explain_failure(metrics_row, threshold_dict):
    explanations = []
    for m, desc in ALL_METRICS.items():
        val = metrics_row.get(m, -1)
        if m in threshold_dict:
            low, high = threshold_dict[m]
        else:
            low, high = ALL_METRIC_THRESHOLDS.get(m, (None, None))
        if low is not None and high is not None:
            if val == -1:
                explanations.append(f"Metric '{m}' could not be computed (value=-1).")
            elif val < low:
                explanations.append(f"Metric '{m}' too low: {val:.2f} < {low}")
            elif val > high:
                explanations.append(f"Metric '{m}' too high: {val:.2f} > {high}")
    return "\n".join(explanations)

def local_contrast(patch_rgb):
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    return np.ptp(gray)

def compute_local_metric(img_rgb, patch_size=50, metric_func=None):
    h, w = img_rgb.shape[:2]
    heatmap = np.zeros((h, w))
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img_rgb[i:min(i+patch_size, h), j:min(j+patch_size, w)]
            value = metric_func(patch) if metric_func else np.ptp(patch)
            heatmap[i:min(i+patch_size, h), j:min(j+patch_size, w)] = value
    hm_min, hm_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - hm_min) / (hm_max - hm_min + 1e-6)
    return heatmap

# ------------------------------
# IMAGE PROCESSING FOR QUALITY METRICS
# ------------------------------
def process_image_for_metrics(uploaded_file, image_number, selected_metrics, threshold_dict, batch_id="default", modality="Fundus"):
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_image)
    if modality == "Fundus":
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        mask = detect_foreground_fundus(img_bgr)
        metrics = compute_quality_metrics(img_bgr, mask)
        segmented_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        mask_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif modality == "OCTA":
        import opsfaz  # Ensure opsfaz.py is available
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faz_mask, area, cnt = opsfaz.detectFAZ(gray, mm=3, prof=0, precision=0.7)
        mask = (faz_mask * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        metrics = compute_quality_metrics(img_bgr, mask)
        metrics["faz_area"] = area
        segmented_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        mask_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        mask = detect_foreground_fundus(img_bgr)
        metrics = compute_quality_metrics(img_bgr, mask)
        segmented_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        mask_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    filename = f"{image_number}_{uploaded_file.name}"
    metrics["filename"] = filename
    metrics["batch"] = batch_id
    pass_count = 0
    for m in selected_metrics:
        val = metrics.get(m, -1)
        if val == -1:
            continue
        low, high = threshold_dict[m]
        if low <= val <= high:
            pass_count += 1
    is_pass = (pass_count >= 0.75 * len(selected_metrics))
    metrics["is_pass"] = is_pass
    save_dir = PASSED_DIR if is_pass else FAILED_DIR
    cv2.imwrite(os.path.join(save_dir, filename), img_bgr)
    return {
        "original_bgr": img_bgr,
        "mask_bgr": mask_disp,
        "segmented_bgr": segmented_img,
        "metrics": metrics,
        "is_pass": is_pass
    }

# ------------------------------
# QUALITY GROUP ANALYSIS
# ------------------------------
def assign_quality_group(df):
    df = df.copy()
    df["Pass_Status"] = df["is_pass"].apply(lambda x: "Passed" if x else "Failed")
    passed = df[df["is_pass"]]
    if not passed.empty:
        passed = passed.copy()
        passed["Quality_Score"] = passed.get("psnr", 0) + passed.get("snr1", 0) + passed.get("contrast", 0) - passed.get("blur", 0)
        try:
            passed["Quality_Group"] = pd.qcut(passed["Quality_Score"], 3, labels=["Low", "Medium", "High"])
        except Exception:
            passed["Quality_Group"] = "Medium"
    failed = df[~df["is_pass"]].copy()
    failed["Quality_Group"] = "Failed"
    return pd.concat([passed, failed])

def display_quality_group_analysis(df, selected_metrics, threshold_dict):
    st.markdown("## Quality Group Analysis")
    df = assign_quality_group(df)
    avg_metrics = df.groupby("Quality_Group")[list(ALL_METRICS.keys())].mean().reset_index()
    fig_bar = px.bar(
        avg_metrics,
        x="Quality_Group",
        y=avg_metrics.columns[1:],
        title="Average Quality Metrics by Group",
        barmode="group",
        color="Quality_Group",
        color_discrete_map=COLOR_MAP,
        template="plotly_white"
    )
    fig_bar.update_layout(legend_font_size=14)
    st.plotly_chart(fig_bar, use_container_width=True)
    selected_group = st.selectbox("Select Quality Group for detailed analysis:", df["Quality_Group"].unique())
    group_df = df[df["Quality_Group"] == selected_group]
    st.markdown(f"### Images in {selected_group} Group")
    st.dataframe(group_df)
    st.markdown("### Explanations for Each Image in This Group")
    for _, row in group_df.iterrows():
        explanation = explain_failure(row, threshold_dict)
        with st.expander(f"Image: {row['filename']}"):
            st.write(explanation)

# ------------------------------
# CUSTOM 3D VISUALIZATIONS
# ------------------------------
def custom_3d_scatter(df):
    st.markdown("## Custom 3D Scatter Plot")
    norm_df = normalize_df(df)
    available_cols = list(ALL_METRICS.keys())
    x_axis = st.selectbox("X-axis metric:", available_cols, index=0)
    y_axis = st.selectbox("Y-axis metric:", available_cols, index=1)
    z_axis = st.selectbox("Z-axis metric:", available_cols, index=2)
    if "Quality_Group" not in norm_df.columns:
        norm_df["Quality_Group"] = assign_quality_group(df)["Quality_Group"]
    fig = px.scatter_3d(
        norm_df,
        x=x_axis,
        y=y_axis,
        z=z_axis,
        color="Quality_Group",
        color_discrete_map=COLOR_MAP,
        title="Custom 3D Scatter Plot (Quality Groups)",
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(legend_font_size=14)
    st.plotly_chart(fig, use_container_width=True)

def interactive_3d_analysis(df):
    st.markdown("## Interactive 3D Analysis of Quality Metrics")
    method_options = ["PCA", "t-SNE"]
    if UMAP_AVAILABLE:
        method_options.append("UMAP")
    method = st.selectbox("Select Dimensionality Reduction Method:", options=method_options)
    feats = normalize_df(df)[list(ALL_METRICS.keys())].fillna(0).values
    if method == "PCA":
        dr_model = PCA(n_components=3)
        X_dr = dr_model.fit_transform(feats)
    elif method == "t-SNE":
        # Adjust perplexity if n_samples is too low
        n_samples = feats.shape[0]
        perplexity = 5
        if n_samples <= perplexity:
            perplexity = max(1, n_samples - 1)
        dr_model = TSNE(n_components=3, perplexity=perplexity, random_state=42)
        X_dr = dr_model.fit_transform(feats)
    else:
        dr_model = UMAP(n_components=3)
        X_dr = dr_model.fit_transform(feats)
    display_df = normalize_df(df).copy()
    display_df["Dim1"], display_df["Dim2"], display_df["Dim3"] = X_dr[:, 0], X_dr[:, 1], X_dr[:, 2]
    if "Quality_Group" not in display_df.columns:
        display_df["Quality_Group"] = assign_quality_group(df)["Quality_Group"]
    fig_dr = px.scatter_3d(
        display_df,
        x="Dim1",
        y="Dim2",
        z="Dim3",
        color="Quality_Group",
        hover_data=display_df.columns,
        title=f"{method} 3D Interactive Plot (Colored by Quality Group)",
        template="plotly_white",
        color_discrete_map=COLOR_MAP
    )
    fig_dr.update_traces(marker=dict(size=12))
    fig_dr.update_layout(legend_font_size=14)
    st.plotly_chart(fig_dr, use_container_width=True)
    st.markdown("""
    **Clinical Insight:**  
    This 3D interactive plot reveals the clustering of images based on quality metrics.
    After batch correction, images from the same class should group together more tightly.
    """)

def surface_correlation_plot(df):
    st.markdown("## 3D Surface Plot of Metric Correlation")
    metrics_names = list(ALL_METRICS.keys())
    corr = df[metrics_names].corr().values
    x = np.arange(len(metrics_names))
    y = np.arange(len(metrics_names))
    fig = go.Figure(data=[go.Surface(z=corr, x=x, y=y)])
    fig.update_layout(
        title="Correlation Surface",
        scene=dict(
            xaxis=dict(title='Metric', tickmode='array', tickvals=x, ticktext=metrics_names),
            yaxis=dict(title='Metric', tickmode='array', tickvals=y, ticktext=metrics_names),
            zaxis=dict(title='Correlation')
        ),
        legend_font_size=14
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# STATIC VISUALIZATIONS
# ------------------------------
def static_visualizations(df):
    st.markdown("## Static Visualizations")
    display_metric = normalize_df(df)
    
    # Pairwise Scatter Plot
    st.markdown("### Pairwise Scatter Plot")
    if len(display_metric) > 1:
        g = sns.PairGrid(display_metric.dropna(), corner=True)
        g.map_lower(sns.scatterplot, s=100, alpha=0.7, color="purple")
        g.map_diag(sns.kdeplot, fill=True, color="purple", warn_singular=False)
        g.map_upper(sns.scatterplot, s=100, alpha=0.7, color="purple")
        st.pyplot(g.fig)
        plt.close('all')
    else:
        st.warning("Not enough data for pairwise scatter plot.")
    
    # Violin Plot
    st.markdown("### Violin Plot (Normalized)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=display_metric[ALL_METRICS.keys()], scale="width",
                   inner="quartile", palette="coolwarm", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    st.pyplot(fig)
    plt.close('all')
    
    # Correlation Heatmap
    st.markdown("### Correlation Heatmap")
    corr = display_metric[list(ALL_METRICS.keys())].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
    plt.close('all')
    
    # Radar Chart for Average Metrics by Quality Group
    st.markdown("### Radar Chart of Average Metrics by Quality Group")
    df_qg = assign_quality_group(df)
    avg_qg = df_qg.groupby("Quality_Group")[list(ALL_METRICS.keys())].mean().reset_index()
    categories = list(ALL_METRICS.keys())
    fig_radar = go.Figure()
    for group in avg_qg["Quality_Group"].unique():
        group_data = avg_qg[avg_qg["Quality_Group"] == group]
        values = group_data[categories].values.flatten().tolist()
        values += values[:1]  # Close the loop
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=str(group)
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Radar Chart of Average Metrics by Quality Group",
        legend_font_size=14
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("""
    **Clinical Insight:**  
    These static plots (pairwise scatter, violin plot, heatmap, and radar chart) provide comprehensive insights into the relationships among quality metrics.
    """)

# ------------------------------
# STREAMLIT APP MAIN
# ------------------------------
def main():
    st.title("Image Quality Dashboard")
    st.markdown("This app performs quality assessment, preprocessing, and batch effect correction for retinal Fundus and OCTA images. It offers both interactive and static visualizations for deep clinical insights.")
    
    # Sidebar
    modality = st.sidebar.selectbox("Select Image Modality", ["Fundus", "OCTA"])
    st.sidebar.title("Upload Images")
    uploaded_files = st.sidebar.file_uploader("Choose images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    batch_ids = {}
    if uploaded_files:
        st.sidebar.markdown("### Assign Batch IDs")
        for uf in uploaded_files:
            batch_ids[uf.name] = st.sidebar.text_input(f"Batch ID for {uf.name}", value="Batch_1", key=uf.name)
    
    process_mode = st.sidebar.radio("Select Processing Mode", ("Quality Metrics", "Preprocessing"))
    
    if process_mode == "Quality Metrics":
        st.sidebar.markdown("### Select Metrics for Pass/Fail")
        selected_metrics = st.sidebar.multiselect(
            "Choose at least one metric:",
            list(ALL_METRICS.keys()),
            default=DEFAULT_CRITERIA
        )
        if not selected_metrics:
            st.sidebar.warning("No metrics selected. Using default set.")
            selected_metrics = DEFAULT_CRITERIA
        st.sidebar.markdown("### Define Thresholds for Each Selected Metric")
        threshold_dict = {}
        for m in selected_metrics:
            default_low, default_high = ALL_METRIC_THRESHOLDS.get(m, (0,300))
            if default_low == float('-inf'):
                default_low = 0
            if default_high == float('inf'):
                default_high = 300
            min_v, max_v = st.sidebar.slider(
                f"{m} ({ALL_METRICS[m]})",
                0.0, 300.0,
                (float(default_low), float(default_high)),
                key=f"thresh_{m}"
            )
            threshold_dict[m] = (min_v, max_v)
    
    if uploaded_files:
        if process_mode == "Quality Metrics":
            results = []
            passed_images_data = []
            failed_images_data = []
            for idx, uf in enumerate(uploaded_files, start=1):
                bid = batch_ids.get(uf.name, "Batch_1")
                processed = process_image_for_metrics(uf, idx, selected_metrics, threshold_dict, batch_id=bid, modality=modality)
                metrics = processed["metrics"]
                results.append(metrics)
                orig_rgb = cv2.cvtColor(processed["original_bgr"], cv2.COLOR_BGR2RGB)
                mask_rgb = cv2.cvtColor(processed["mask_bgr"], cv2.COLOR_BGR2RGB)
                seg_rgb = cv2.cvtColor(processed["segmented_bgr"], cv2.COLOR_BGR2RGB)
                if processed["is_pass"]:
                    passed_images_data.append((orig_rgb, mask_rgb, seg_rgb, metrics["filename"]))
                else:
                    failed_images_data.append((orig_rgb, metrics["filename"], metrics))
            df = pd.DataFrame(results)
            df = assign_quality_group(df)
            
            tabs = st.tabs([
                "Metrics Table",
                "Passed Images",
                "Failed Images",
                "Interactive Plots",
                "Interactive 3D Analysis",
                "Custom 3D Scatter",
                "Batch Effects (3D)",
                "Static Visualizations",
                "Batch Effects (2D)",
                "Quality Group Analysis"
            ])
            
            with tabs[0]:
                st.subheader("Quality Metrics Table")
                st.dataframe(df)
            
            with tabs[1]:
                st.subheader("✅ Passed Images")
                if passed_images_data:
                    for orig_img, mask_img, seg_img, fname in passed_images_data:
                        with st.expander(f"Passed Image: {fname}"):
                            cols = st.columns(3)
                            cols[0].image(orig_img, caption="Original Image", use_container_width=True)
                            cols[1].image(mask_img, caption="Segmentation Mask", use_container_width=True)
                            cols[2].image(seg_img, caption="Segmented Region", use_container_width=True)
                else:
                    st.info("No passed images found.")
            
            with tabs[2]:
                st.subheader("❌ Failed Images")
                if failed_images_data:
                    for orig_img, fname, metrics_row in failed_images_data:
                        with st.expander(f"Failed Image: {fname}"):
                            st.image(orig_img, caption=f"Original Image: {fname}", use_container_width=True)
                            heatmap = compute_local_metric(orig_img, patch_size=50, metric_func=local_contrast)
                            plt.figure(figsize=(5, 5))
                            plt.imshow(orig_img)
                            plt.imshow(heatmap, cmap='jet', alpha=0.5)
                            plt.colorbar(label="Local Contrast")
                            plt.title("Heatmap: Local Contrast")
                            st.pyplot(plt)
                            plt.close('all')
                            st.text("Reasons for Failure:\n" + explain_failure(metrics_row, threshold_dict))
                            for m in selected_metrics:
                                val = metrics_row.get(m, -1)
                                low, high = threshold_dict[m]
                                if val != -1 and (val < low or val > high) and m in LOCAL_METRIC_FUNCS:
                                    func = LOCAL_METRIC_FUNCS[m]
                                    heatmap_metric = compute_local_metric(orig_img, patch_size=50, metric_func=func)
                                    plt.figure(figsize=(5, 5))
                                    plt.imshow(orig_img)
                                    plt.imshow(heatmap_metric, cmap='jet', alpha=0.5)
                                    plt.colorbar(label=f"{m} value")
                                    plt.title(f"Heatmap for {m}")
                                    st.pyplot(plt)
                                    plt.close('all')
                else:
                    st.info("No failed images found.")
            
            with tabs[3]:
                st.subheader("Interactive Plots (All Metrics)")
                norm_df = normalize_df(df)
                st.markdown("### Parallel Coordinates")
                color_choices = [m for m in ALL_METRICS.keys() if m in norm_df.columns]
                if not color_choices:
                    st.info("No metrics available for parallel coordinates.")
                else:
                    color_metric = st.selectbox("Select metric for coloring:", color_choices, index=0)
                    dims = [m for m in ALL_METRICS.keys() if m in norm_df.columns]
                    fig_px = px.parallel_coordinates(
                        norm_df,
                        dimensions=dims,
                        color=color_metric,
                        color_continuous_scale=px.colors.sequential.Plasma,
                        color_continuous_midpoint=norm_df[color_metric].mean(),
                        template="plotly_white"
                    )
                    fig_px.update_layout(legend_font_size=14)
                    st.plotly_chart(fig_px, use_container_width=True)
                st.markdown("""
                **Clinical Insight:**  
                Parallel coordinates visualize all quality metrics simultaneously.
                """)
            
            with tabs[4]:
                interactive_3d_analysis(df)
            
            with tabs[5]:
                custom_3d_scatter(df)
            
            with tabs[6]:
                st.subheader("Batch Effects (3D)")
                batch_effects_3d(df)
                st.markdown("### 3D Surface of Metric Correlation")
                surface_correlation_plot(df)
            
            with tabs[7]:
                static_visualizations(df)
            
            with tabs[8]:
                st.subheader("Batch Effects (2D)")
                st.markdown("#### Before Correction")
                plot_batch_effects(df, corrected=False)
                df_corrected = simple_batch_correct_metrics(df, batch_col="batch")
                st.markdown("#### After Correction")
                plot_batch_effects(df_corrected, corrected=True)
                plot_metric_distributions(df)
            
            with tabs[9]:
                display_quality_group_analysis(df, selected_metrics, threshold_dict)
            
            csv_path = os.path.join(OUTPUT_DIR, "quality_metrics.csv")
            df.to_csv(csv_path, index=False)
            with open(csv_path, "rb") as f:
                st.sidebar.download_button(
                    "Download Quality Metrics CSV",
                    f,
                    file_name="quality_metrics.csv",
                    mime="text/csv"
                )
            
            st.markdown("""
            ---
            **Clinical Insights:**  
            - *Batch Effects:* The 2D and 3D PCA views show data distribution before and after correction.
            - *Interactive 3D Analysis:* Clustering methods reveal groupings based on quality metrics.
            - *Static Visualizations:* Pairwise scatter, violin plots, heatmap, and radar charts offer comprehensive insights.
            ---
            """)
        
        elif process_mode == "Preprocessing":
            st.subheader(f"{modality} Preprocessing")
            preprocessed_data = []
            for idx, uf in enumerate(uploaded_files, start=1):
                try:
                    pil_image = Image.open(uf).convert("RGB")
                except Exception as e:
                    st.error(f"Error reading image {uf.name}: {e}")
                    continue
                img_array = np.array(pil_image)
                if modality == "Fundus":
                    proc_img, mask_img = preprocess_fundus_image(img_array)
                elif modality == "OCTA":
                    import opsfaz
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    faz_mask, area, cnt = opsfaz.detectFAZ(gray, mm=3, prof=0, precision=0.7)
                    proc_img = cv2.bitwise_and(img_array, img_array, mask=(faz_mask*255).astype(np.uint8))
                    mask_img = (faz_mask*255).astype(np.uint8)
                else:
                    proc_img, mask_img = preprocess_fundus_image(img_array)
                filename = f"preprocessed_{idx}_{uf.name}"
                save_path = os.path.join(PREPROCESS_DIR, filename)
                proc_bgr = cv2.cvtColor(proc_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, proc_bgr)
                preprocessed_data.append((img_array, proc_img, mask_img, filename))
            for orig_img, proc_img, mask_img, fname in preprocessed_data:
                with st.expander(f"Preprocessed Image: {fname}"):
                    st.image(orig_img, caption="Original Image", use_container_width=True)
                    cols = st.columns(2)
                    cols[0].image(proc_img, caption="Preprocessed (Cropped/Segmented) Image", use_container_width=True)
                    mask_3ch = np.stack([mask_img]*3, axis=-1) if mask_img.ndim == 2 else mask_img
                    cols[1].image(mask_3ch, caption="Mask", use_container_width=True)
    else:
        st.info("📂 Upload at least one image to analyze.")

if __name__ == "__main__":
    main()
