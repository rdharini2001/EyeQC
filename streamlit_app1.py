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
from PIL import Image
from pandas.plotting import parallel_coordinates
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# -----------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------
st.set_page_config(page_title="Fundus Image Quality Dashboard", layout="wide")

# -----------------------------------------------
# Create Output Directories
# -----------------------------------------------
OUTPUT_DIR = "results"
PASSED_DIR = os.path.join(OUTPUT_DIR, "passed_images")
FAILED_DIR = os.path.join(OUTPUT_DIR, "failed_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PASSED_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)

# -----------------------------------------------
# Comprehensive List of All Metrics
# -----------------------------------------------
ALL_METRICS = {
    "mean": "Average intensity of fundus region",
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
    "foreground_area": "Fraction of image as fundus",
    "edge_density": "Edge density (fundus region)",
}

# Default thresholds
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

# -----------------------------------------------
# Foreground Mask & Additional Metric
# -----------------------------------------------
def detect_foreground_fundus(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def compute_edge_density(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    fg_edges = edges[mask == 255]
    if fg_edges.size == 0:
        return -1
    return np.count_nonzero(fg_edges) / fg_edges.size

# -----------------------------------------------
# Quality Metrics
# -----------------------------------------------
def compute_quality_metrics(img, mask):
    """Compute all metrics in ALL_METRICS."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fg_pixels = gray[mask == 255]
    bg_pixels = gray[mask == 0]

    if fg_pixels.size == 0 or bg_pixels.size == 0:
        return {m: -1 for m in ALL_METRICS.keys()}

    mean_val = np.mean(fg_pixels)
    var_val = np.var(fg_pixels)
    range_val = np.ptp(fg_pixels)
    cv_val = (np.std(fg_pixels) / mean_val) * 100 if mean_val != 0 else -1
    entropy_val = shannon_entropy(fg_pixels)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_val = np.var(lap[mask == 255])

    min_val, max_val = np.min(fg_pixels), np.max(fg_pixels)
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
    edge_density_val = compute_edge_density(img, mask)

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

# -----------------------------------------------
# Normalize All Metrics
# -----------------------------------------------
def normalize_df(df):
    norm_df = df.copy()
    for col in ALL_METRICS.keys():
        if col in norm_df.columns:
            mn = norm_df[col].min()
            mx = norm_df[col].max()
            if mx > mn:
                norm_df[col] = (norm_df[col] - mn) / (mx - mn)
            else:
                norm_df[col] = 0
    return norm_df

# -----------------------------------------------
# Process Image with Pass/Fail
# -----------------------------------------------
def process_image(uploaded_file, image_number, selected_metrics, threshold_dict):
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    mask = detect_foreground_fundus(img_cv)
    metrics = compute_quality_metrics(img_cv, mask)

    # For display if passed
    segmented_img = cv2.bitwise_and(img_cv, img_cv, mask=mask)
    mask_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    filename = f"{image_number}_{uploaded_file.name}"
    metrics["filename"] = filename

    # Count passes
    pass_count = 0
    for m in selected_metrics:
        val = metrics.get(m, -1)
        if val == -1:
            continue
        low, high = threshold_dict[m]
        if low <= val <= high:
            pass_count += 1

    # At least 75% must pass
    is_pass = (pass_count >= 0.75 * len(selected_metrics))

    save_dir = PASSED_DIR if is_pass else FAILED_DIR
    cv2.imwrite(os.path.join(save_dir, filename), img_cv)

    return {
        "original_bgr": img_cv,
        "mask_bgr": mask_disp,
        "segmented_bgr": segmented_img,
        "metrics": metrics,
        "is_pass": is_pass
    }

# -----------------------------------------------
# Interactive Plot (All metrics)
# -----------------------------------------------
def display_interactive_plots(df):
    norm_df = normalize_df(df)

    st.markdown("### Parallel Coordinates of All Metrics")
    color_choices = [m for m in ALL_METRICS.keys() if m in norm_df.columns]
    if not color_choices:
        st.info("No metrics available for interactive plot.")
        return

    color_metric = st.selectbox("Select a metric to color the parallel coordinates:", color_choices)

    dims = [m for m in ALL_METRICS.keys() if m in norm_df.columns]
    if len(dims) < 1:
        st.info("No metrics to show in parallel coordinates.")
        return

    fig_px = px.parallel_coordinates(
        norm_df,
        dimensions=dims,
        color=color_metric,
        color_continuous_scale=px.colors.sequential.Plasma,
        color_continuous_midpoint=norm_df[color_metric].mean() if color_metric in norm_df.columns else 0.5,
        labels={d: d for d in dims}
    )
    fig_px.update_traces(dimensions=[dict(label=d, values=norm_df[d]) for d in dims])
    st.plotly_chart(fig_px, use_container_width=True)

    st.markdown("> **Interpretation**: Parallel coordinates let you compare multiple metrics at once. "
                "Each line is an image, crossing each vertical axis at its normalized metric value. "
                "Coloring by a chosen metric highlights differences among images. "
                "Look for lines that stay high (potential high-quality) vs. those dipping low across many axes.")

    st.markdown("### Image + Features")
    st.markdown("Below, expand each row to see the original (passed/failed) image alongside all metrics.")
    for i, row in df.iterrows():
        fname = row["filename"]
        is_pass_str = "Passed" if row.get("is_pass") else "Failed"
        with st.expander(f"[{is_pass_str}] Image: {fname}"):
            c1, c2 = st.columns([1, 2])
            passed_path = os.path.join(PASSED_DIR, fname)
            failed_path = os.path.join(FAILED_DIR, fname)
            disp = None
            if os.path.exists(passed_path):
                disp = cv2.imread(passed_path)
            elif os.path.exists(failed_path):
                disp = cv2.imread(failed_path)

            if disp is not None:
                c1.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), caption=f"{fname}", width=170)
            else:
                c1.write("Not found on disk")

            feat_dict = {}
            for m in ALL_METRICS.keys():
                val = row.get(m, None)
                feat_dict[m] = round(val, 3) if val is not None else None
            c2.write(pd.DataFrame([feat_dict]))

# -----------------------------------------------
# Static Visualizations (All metrics)
# -----------------------------------------------
def display_static_visualizations(df):
    st.markdown("## Advanced Static Visualizations (All Metrics)")
    norm_df = normalize_df(df)
    df_plot = norm_df[list(ALL_METRICS.keys())].replace(-1, np.nan).copy()

    # Pairwise Scatter
    st.markdown("### Pairwise Scatter (All Feature Combinations)")
    if len(df_plot) > 1:
        g = sns.PairGrid(df_plot.dropna(), corner=True)
        g.map_lower(sns.scatterplot, s=40, alpha=0.7, color="purple")
        g.map_diag(sns.kdeplot, fill=True, color="purple")
        g.map_upper(sns.scatterplot, s=40, alpha=0.7, color="purple")
        st.pyplot(g.fig)
        st.markdown("> **Interpretation**: Correlations between pairs of normalized metrics. Diagonal plots show distributions.")
    else:
        st.warning("Not enough data for pairwise scatter among all features.")

    # Violin Plot
    st.markdown("### Violin Plot of All Metrics (Normalized)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=df_plot, scale="width", inner="quartile", palette="coolwarm", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

    # Radar Chart
    st.markdown("### Radar (Spider) Chart for Up to 5 Images")
    if len(df_plot) > 0:
        categories = df_plot.columns.tolist()
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        max_images = min(len(df_plot), 5)
        for i in range(max_images):
            row_vals = df_plot.iloc[i].values.flatten().tolist()
            row_vals += row_vals[:1]
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            label_name = f"Img {i+1}"
            ax.plot(angles, row_vals, linewidth=1, linestyle='solid', label=label_name)
            ax.fill(angles, row_vals, alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)
    else:
        st.info("No data for radar chart.")

    # Quality Score & Grouping
    st.markdown("### Quality Score & Grouping")
    grouping_df = df_plot.copy()
    grouping_df["Quality_Score"] = (
        grouping_df.get("psnr", 0)
        + grouping_df.get("snr1", 0)
        + grouping_df.get("contrast", 0)
        - grouping_df.get("blur", 0)
    )
    grouping_df["Quality_Group"] = pd.qcut(grouping_df["Quality_Score"], 3, labels=["Low", "Medium", "High"])

    # Boxen Plot
    st.markdown("### Boxen Plot (Choose Metric)")
    all_cols = list(df_plot.columns)
    boxen_metric = st.selectbox("Select metric for Boxen Plot:", all_cols, key="boxen_metric")
    st.markdown("#### Quality Group Dropdown (beside the boxen plot)")
    chosen_group_boxen = st.selectbox("Select L/M/H group:", ["Low", "Medium", "High"], key="boxen_group")
    matched_idx = grouping_df.index[grouping_df["Quality_Group"] == chosen_group_boxen].tolist()
    matched_fnames = df.loc[matched_idx, "filename"].tolist()
    st.write(f"**Images in {chosen_group_boxen} group** for the Boxen Plot context:", matched_fnames)

    fig, ax = plt.subplots(figsize=(8, 4))
    boxen_data = pd.DataFrame({
        "Quality_Group": grouping_df["Quality_Group"],
        boxen_metric: df_plot[boxen_metric]
    }).dropna()
    sns.boxenplot(x="Quality_Group", y=boxen_metric, data=boxen_data, palette="coolwarm", ax=ax)
    ax.set_title(f"Boxen Plot of {boxen_metric} by Quality Group")
    st.pyplot(fig)

    # Dual Axis
    st.markdown("### Dual Axis Plot: Choose Two Metrics")
    axis_cols = [c for c in df_plot.columns if c in ALL_METRICS]
    metric_left = st.selectbox("Select Left Axis Metric:", axis_cols, key="dual_left")
    metric_right = st.selectbox("Select Right Axis Metric:", axis_cols, key="dual_right")
    st.markdown("#### Quality Group Dropdown (beside the dual axis plot)")
    chosen_group_dual = st.selectbox("Select L/M/H group for the dual axis context:", ["Low", "Medium", "High"], key="dual_group")
    matched_idx_dual = grouping_df.index[grouping_df["Quality_Group"] == chosen_group_dual].tolist()
    matched_fnames_dual = df.loc[matched_idx_dual, "filename"].tolist()
    st.write(f"**Images in {chosen_group_dual} group** for the Dual Axis context:", matched_fnames_dual)

    if metric_left != metric_right and metric_left in df_plot.columns and metric_right in df_plot.columns:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        c1 = "tab:blue"
        c2 = "tab:red"
        ax1.set_xlabel("Image Index (Row Order)")
        ax1.set_ylabel(metric_left, color=c1)
        ax1.plot(df_plot.index, df_plot[metric_left], marker="o", color=c1, label=metric_left)
        ax1.tick_params(axis="y", labelcolor=c1)
        ax2 = ax1.twinx()
        ax2.set_ylabel(metric_right, color=c2)
        ax2.plot(df_plot.index, df_plot[metric_right], marker="s", linestyle="dashed", color=c2, label=metric_right)
        ax2.tick_params(axis="y", labelcolor=c2)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Please select two different metrics for a meaningful Dual Axis plot.")

    # PCA & t-SNE
    st.markdown("### PCA & t-SNE (All Metrics)")
    plot_df = df_plot.dropna().copy()
    if len(plot_df) < 2:
        st.warning("Not enough data for PCA/t-SNE.")
    else:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(plot_df.values)
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        X_tsne = tsne.fit_transform(plot_df.values)

        plot_df["PCA1"], plot_df["PCA2"] = X_pca[:, 0], X_pca[:, 1]
        plot_df["TSNE1"], plot_df["TSNE2"] = X_tsne[:, 0], X_tsne[:, 1]

        st.markdown("> **PCA** reveals the primary axes of variation among the metrics, while **t-SNE** can uncover subtle local clusters.")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x="PCA1", y="PCA2", hue=grouping_df["Quality_Group"].reindex(plot_df.index), data=plot_df, palette="coolwarm", ax=ax)
        ax.set_title("PCA (All Metrics)")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x="TSNE1", y="TSNE2", hue=grouping_df["Quality_Group"].reindex(plot_df.index), data=plot_df, palette="coolwarm", ax=ax)
        ax.set_title("t-SNE (All Metrics)")
        st.pyplot(fig)

    # Parallel Coordinates
    st.markdown("### Parallel Coordinates Plot (All Metrics)")
    parallel_df = df_plot.dropna().copy()
    if len(parallel_df) < 2:
        st.warning("Not enough data for parallel coordinates.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        parallel_df["ID"] = parallel_df.index.astype(str)
        col_order = list(df_plot.columns)
        pc_data = parallel_df[col_order].copy()
        pc_data["Quality_Group"] = grouping_df["Quality_Group"].reindex(pc_data.index)
        parallel_coordinates(pc_data, "Quality_Group", color=sns.color_palette("Set2"))
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    # Ward Linkage Dendrogram
    st.markdown("### Hierarchical Clustering (Ward)")
    dendro_df = df_plot.dropna().copy()
    if len(dendro_df) > 1:
        link_mat = linkage(dendro_df.values, method="ward")
        fig, ax = plt.subplots(figsize=(8, 4))
        dendrogram(link_mat, ax=ax)
        ax.set_title("Ward Linkage Dendrogram")
        st.pyplot(fig)
    else:
        st.info("Not enough data for hierarchical clustering.")

    # -------------- More Flexible Plotly 3D --------------
    st.markdown("### Plotly 3D Scatter (User-Defined Axes and Color)")
    st.markdown("You can select which metrics go on the X, Y, Z axes, as well as which metric is used for color and size (optional).")

    # Make a dropdown for x, y, z, color, size
    available_cols = df_plot.columns.tolist()
    x_3d = st.selectbox("X-axis metric:", available_cols, key="3d_x")
    y_3d = st.selectbox("Y-axis metric:", available_cols, key="3d_y")
    z_3d = st.selectbox("Z-axis metric:", available_cols, key="3d_z")
    color_3d = st.selectbox("Color metric:", ["None"] + available_cols, key="3d_color")
    size_3d = st.selectbox("Size metric:", ["None"] + available_cols, key="3d_size")

    if all(ax in df_plot.columns for ax in [x_3d, y_3d, z_3d]):
        hover_data = [col for col in df_plot.columns if col != "filename"]
        color_arg = color_3d if color_3d != "None" else None
        size_arg = size_3d if size_3d != "None" else None

        fig_3d_custom = px.scatter_3d(
            df_plot,
            x=x_3d,
            y=y_3d,
            z=z_3d,
            color=color_arg,
            size=size_arg,
            hover_data=hover_data,
            title="Custom 3D Scatter"
        )
        st.plotly_chart(fig_3d_custom, use_container_width=True)
        st.markdown(f"> **Interpretation**: You are plotting X={x_3d}, Y={y_3d}, Z={z_3d}, color by {color_arg}, size by {size_arg}. Look for clusters or outliers in 3D space.")
    else:
        st.info("Please select valid columns for X, Y, and Z to visualize the 3D plot.")

# -----------------------------------------------
# Additional Clinical Visualizations
# -----------------------------------------------
def display_clinical_visualizations(df):
    """
    Show domain-specific plots, using normalized data.
    Removed the scatter from clinical tab, replaced with a more appealing 3D plot.
    """
    st.markdown("## Additional Clinical Visualizations (Normalized)")
    norm_df = normalize_df(df)
    numeric_cols = ["mean","blur","contrast","psnr","entropy","snr1","snr2","snr3","snr4","cnr","foreground_area"]
    numeric_cols = [c for c in numeric_cols if c in norm_df.columns]

    if not numeric_cols:
        st.warning("No valid numeric metrics for clinical visuals.")
        return

    sub_df = norm_df[numeric_cols].copy().dropna()

    # Boxplot
    st.markdown("### Boxplot of Key Clinical Metrics (Normalized)")
    fig, ax = plt.subplots(figsize=(8, 5))
    sub_df.boxplot(ax=ax)
    st.pyplot(fig)

    # Correlation heatmap
    st.markdown("### Correlation Heatmap (Normalized Clinical Metrics)")
    fig, ax = plt.subplots(figsize=(8, 5))
    corr_mat = sub_df.corr()
    sns.heatmap(corr_mat, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    '''
    # Another 3D Plotly
    st.markdown("### Plotly 3D Plot: Another Clinical Perspective")
    if all(x in sub_df.columns for x in ["mean","contrast","snr1","psnr","entropy"]):
        fig_3d_clinical = px.scatter_3d(
            sub_df,
            x="mean",
            y="contrast",
            z="snr1",
            color="psnr",
            size="entropy",
            hover_data=sub_df.columns.tolist(),
            title="Clinical 3D Plot: Mean vs. Contrast vs. SNR1"
        )
        st.plotly_chart(fig_3d_clinical, use_container_width=True)
    else:
        st.info("Not enough columns for the clinical 3D Plot. Need [mean, contrast, snr1, psnr, entropy].")'''

# -----------------------------------------------
# Streamlit Main
# -----------------------------------------------
st.sidebar.title("Upload Retinal Images")
uploaded_files = st.sidebar.file_uploader("Choose images...", type=["png","jpg","jpeg"], accept_multiple_files=True)

st.sidebar.markdown("### Select Metrics for Pass/Fail")
selected_metrics = st.sidebar.multiselect(
    "Choose at least one metric:", list(ALL_METRICS.keys()), default=DEFAULT_CRITERIA
)
if not selected_metrics:
    st.sidebar.warning("No metrics selected. Using default set.")
    selected_metrics = DEFAULT_CRITERIA

st.sidebar.markdown("### Define Thresholds for Each Selected Metric")
threshold_dict = {}
for m in selected_metrics:
    default_low, default_high = ALL_METRIC_THRESHOLDS.get(m,(0,300))
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
    results = []
    passed_images_data = []
    failed_images_data = []

    for idx, uf in enumerate(uploaded_files, start=1):
        processed = process_image(uf, idx, selected_metrics, threshold_dict)
        metrics = processed["metrics"]
        metrics["is_pass"] = processed["is_pass"]
        results.append(metrics)

        # For display
        orig_rgb = cv2.cvtColor(processed["original_bgr"], cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(processed["mask_bgr"], cv2.COLOR_BGR2RGB)
        seg_rgb = cv2.cvtColor(processed["segmented_bgr"], cv2.COLOR_BGR2RGB)

        if processed["is_pass"]:
            passed_images_data.append((orig_rgb, mask_rgb, seg_rgb, metrics["filename"]))
        else:
            failed_images_data.append((orig_rgb, metrics["filename"]))

    df = pd.DataFrame(results)

    tabs = st.tabs([
        "Metrics Table",
        "Passed Images",
        "Failed Images",
        "Interactive Plots",
        "Static Visualizations",
        "Clinical Visualizations"
    ])

    with tabs[0]:
        st.subheader("Quality Metrics Table")
        st.dataframe(df)

    with tabs[1]:
        st.subheader("✅ Passed Images")
        if passed_images_data:
            for orig_img, mask_img, seg_img, fname in passed_images_data:
                with st.expander(f"Passed Image: {fname}"):
                    c = st.columns(3)
                    c[0].image(orig_img, caption="Original Image", use_container_width=True)
                    c[1].image(mask_img, caption="Segmentation Mask", use_container_width=True)
                    c[2].image(seg_img, caption="Segmented Region", use_container_width=True)
        else:
            st.info("No passed images found.")

    with tabs[2]:
        st.subheader("❌ Failed Images")
        if failed_images_data:
            for orig_img, fname in failed_images_data:
                with st.expander(f"Failed Image: {fname}"):
                    st.image(orig_img, caption=fname, use_container_width=True)
        else:
            st.info("No failed images found.")

    with tabs[3]:
        st.subheader("Interactive Plots (All Metrics)")
        display_interactive_plots(df)

    with tabs[4]:
        st.subheader("Advanced Static Visualizations (All Metrics)")
        display_static_visualizations(df)

    with tabs[5]:
        st.subheader("Additional Clinical Visualizations")
        display_clinical_visualizations(df)

    # CSV Download
    csv_path = os.path.join(OUTPUT_DIR, "fundus_quality_metrics.csv")
    df.to_csv(csv_path, index=False)
    with open(csv_path, "rb") as f:
        st.sidebar.download_button("Download Quality Metrics CSV", f, file_name="fundus_quality_metrics.csv", mime="text/csv")

    st.markdown("""
    ---
    **User-Friendly Interpretation**:

    1. **Pass/Fail**  
       - Based on the selected metrics and thresholds in the sidebar.  
       - Image passes if it meets at least 75% of those thresholds.

    2. **Segmentation**  
       - For **passed images**, we display the **segmentation mask** and **segmented region** so you can see if the fundus was correctly identified.

    3. **Interactive Plots**  
       - Shows **all metrics** in parallel coordinates.
       - A dropdown lets you pick which metric colors the plot (not just PSNR).
       - Each image can be expanded to see its features.

    4. **Static Visualizations**  
       - Informative pairwise scatter, violin, radar, PCA/t-SNE, hierarchical clustering, etc.  
       - The 3D Plotly scatter is now fully **user-customizable**: pick which metrics go on x, y, z, plus color and size.  

    5. **Clinical Visualizations**  
       - Uses **normalized data** for box plots, correlation heatmap, and a new 3D Plotly view.  

    6. **CSV Download**  
       - All metrics in the CSV.  
       - Download from the sidebar.

    **Generating a Public Link**:
    - If you're running locally, you can upload your app to [Streamlit Cloud](https://streamlit.io/cloud) for free.
    - Once deployed, you'll get a **public URL** that anyone can open.
    - Alternatively, you can use [ngrok](https://ngrok.com/) or similar tunneling to share your local port.

    ---
    """)
else:
    st.info("📂 Upload at least one retinal image to analyze.")
