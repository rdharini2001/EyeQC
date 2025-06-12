#!/usr/bin/env python3
import warnings
# hide “invalid value encountered in subtract” runtime warnings everywhere
warnings.filterwarnings("ignore", category=RuntimeWarning)
import types, torch
# Streamlit's path-traversal chokes because torch.classes lacks __path__.
if not hasattr(torch.classes, "__path__"):
    torch.classes.__path__ = []  # pretend it is a namespace pkg
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
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from PIL import Image, ImageFile
import plotly.express as px
import plotly.graph_objects as go
import sklearn.preprocessing as _prep
from sklearn.preprocessing import OneHotEncoder as _OHE
import os
import cv2
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as compare_ssim
from skimage import img_as_float
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from PIL import Image, ImageFile
import plotly.express as px
import plotly.graph_objects as go
import sklearn.preprocessing as _prep
from sklearn.preprocessing import OneHotEncoder as _OHE
from weasyprint import HTML
import tensorflow as tf
# ─── RAG / LangChain Imports ───
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_ollama import ChatOllama
import shap
# ─── IQA Libraries ───
try:
    import niqe
    NIQE_AVAILABLE = True
except ImportError:
    NIQE_AVAILABLE = False

try:
    import brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    BRISQUE_AVAILABLE = False

# for web scraping inside build_local_rag
from bs4 import BeautifulSoup
import requests
from duckduckgo_search import DDGS
from langchain.docstore.document import Document

import numpy as np
from scipy.ndimage import distance_transform_edt, convolve
from skimage.morphology    import skeletonize
from skimage.measure       import label
from skimage.feature       import graycomatrix, graycoprops

import numpy as np, warnings


def _safe(arr, func, default=np.nan):
    """Return func(arr) unless it is all-NaN or empty."""
    arr = np.asarray(arr, float).ravel()
    if arr.size == 0 or np.isnan(arr).all():
        return default
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return func(arr)

_NEIGH_KERN = np.ones((3,3), dtype=int)

def load_mask_array(binary_mask: np.ndarray) -> np.ndarray:
    """Assumes mask is uint8 0/255; returns 0/1 array."""
    return (binary_mask>0).astype(np.uint8)

def area(mask):             return float(mask.sum())
def density(mask):          return float(mask.sum())/mask.size
def make_skel(mask):        return skeletonize(mask>0).astype(np.uint8)
def skeleton_length(skel):  return float(skel.sum())

def branch_end_counts(skel):
    neigh = convolve(skel, _NEIGH_KERN, mode="constant", cval=0) - skel
    return {"branch_pts": int(((skel==1)&(neigh>=3)).sum()),
            "end_pts":    int(((skel==1)&(neigh==1)).sum())}

def diameters(mask, skel):

    dist = distance_transform_edt(mask)
    d = (2*dist[skel>0]).ravel()
    mean_d = _safe(d, np.nanmean, default=0.0)
    std_d  = _safe(d, np.nanstd,  default=0.0)
    #return {"mean_diam": float(np.nanmean(mean_d)),
    #        "std_diam":  float(np.nanstd(std_d))}
    return {"mean_diam": mean_d,
            "std_diam":  std_d}


def fractal_dimension(skel):
    S = skel>0
    maxpow = int(np.log2(min(S.shape)))
    sizes  = 2**np.arange(maxpow,1,-1)
    counts = []
    for sz in sizes:
        h,w = (S.shape[0]//sz)*sz, (S.shape[1]//sz)*sz
        B   = S[:h,:w].reshape(h//sz,sz,w//sz,sz)
        counts.append(B.any((1,3)).sum())
    return float(np.polyfit(np.log(1/sizes), np.log(counts), 1)[0])

def lacunarity(mask, box_sizes=[2,4,8,16]):
    Ls=[]
    M = mask.astype(int)
    for bs in box_sizes:
        h,w=(M.shape[0]//bs)*bs, (M.shape[1]//bs)*bs
        S = M[:h,:w].reshape(h//bs,bs,w//bs,bs).sum((1,3)).ravel()
        μ,σ2=S.mean(),S.var()
        if μ>0: Ls.append(σ2/(μ*μ))
    return float(np.nanmean(Ls)) if Ls else np.nan

def tortuosity(skel):
    coords = np.column_stack(np.nonzero(skel))
    L      = skel.sum()
    if coords.size==0: return np.nan
    span   = np.hypot(*(coords.max(axis=0)-coords.min(axis=0)))
    return float(L/span) if span>0 else np.nan

def glcm_feats(mask):
    M    = (mask>0).astype(np.uint8)
    glcm = graycomatrix(M, [1], [0,np.pi/4,np.pi/2,3*np.pi/4],
                        levels=2, symmetric=True, normed=True)
    return {f"glcm_{p}": float(graycoprops(glcm,p).mean())
            for p in ("contrast","correlation","energy","homogeneity")}

def segment_stats(skel):
    lbl     = label(skel, connectivity=2)
    lengths = np.array([(lbl==i).sum() for i in np.unique(lbl) if i>0])
    if lengths.size==0:
        return {"n_segments":0,"mean_seg_len":0.0,"std_seg_len":0.0}
    return {"n_segments":   int(lengths.size),
            "mean_seg_len": float(lengths.mean()),
            "std_seg_len":  float(lengths.std())}

from vessel_features import (
    make_skel, area, density, skeleton_length,
    branch_end_counts, diameters, fractal_dimension,
    lacunarity, tortuosity, glcm_feats, segment_stats
)

def OneHotEncoder(*args, **kwargs):
    if "sparse" in kwargs:
        kwargs["sparse_output"] = kwargs.pop("sparse")
    return _OHE(*args, **kwargs)

_prep.OneHotEncoder = OneHotEncoder

# --------------------------------------
# OPTIONAL LIBRARIES
# --------------------------------------
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from neurocombat_sklearn import CombatModel
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

# Streamlit config
ImageFile.LOAD_TRUNCATED_IMAGES = True
st.set_page_config(page_title="Image Quality Dashboard", layout="wide")

# --------------------------------------
# DIRECTORIES
# --------------------------------------
OUTPUT_DIR = "results"
PASSED_DIR = os.path.join(OUTPUT_DIR, "passed_images")
FAILED_DIR = os.path.join(OUTPUT_DIR, "failed_images")
PREPROCESS_DIR = os.path.join(OUTPUT_DIR, "preprocessed_images")
for d in [OUTPUT_DIR, PASSED_DIR, FAILED_DIR, PREPROCESS_DIR]:
    os.makedirs(d, exist_ok=True)

# --------------------------------------
# METRICS & THRESHOLDS
# --------------------------------------
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
    "Batch_Effect_Index": "Batch Effect Index (BEI)",
    "illumination_uniformity": "Illumination Uniformity",
    "v_area": "Vessel area",
    "v_density": "Vessel density",
    "v_skel_len": "Skeleton length",
    "v_branch_pts": "Branch points",
    "v_end_pts": "End points",
    "v_mean_diam": "Mean vessel diameter",
    "v_std_diam": "Std vessel diameter",
    "v_fractal_dim": "Fractal dimension",
    "v_lacunarity": "Lacunarity",
    "v_tortuosity": "Tortuosity",
    **{f"v_glcm_{p}":f"GLCM {p}" for p in ["contrast","correlation","energy","homogeneity"]},
    "v_n_segments": "Number of segments",
    "v_mean_seg_len": "Mean segment length",
    "v_std_seg_len": "Std segment length",
}

def get_available_metrics(df, metric_dict):
    return [m for m in metric_dict.keys() if m in df.columns]



# Renamed to ALL_METRICS_THRESHOLDS (plural) for consistency.
ALL_METRICS_THRESHOLDS = {
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
    "edge_density": (0.05, 0.5),
    "Batch_Effect_Index": (0, 2.0),
    "illumination_uniformity": (0.6, float("inf")),
}

DEFAULT_CRITERIA = ["contrast", "psnr", "entropy", "blur"]
COLOR_MAP = {"Low": "red", "Medium": "yellow", "High": "green", "Failed": "gray"}

for k in ALL_METRICS:
    if k.startswith("v_"):
        ALL_METRICS_THRESHOLDS[k] = (0, float("inf"))



def metrics_threshold_table():
    rows = []
    for metric, desc in ALL_METRICS.items():
        low, high = ALL_METRICS_THRESHOLDS.get(metric, (None,None))
        rows.append({
            "Metric": metric,
            "Description": desc,
            "Acceptable Low": low,
            "Acceptable High": high,
            # you could also map which modality this applies to:
            "Modalities": "Fundus/OCTA/OCT"  # or split logic if some only apply to OCTA, OCT
        })
    return pd.DataFrame(rows)
# --------------------------------------
# NEW FUNCTIONS: IMAGE-LEVEL BATCH CORRECTION
# --------------------------------------
def compute_mean_std_for_images(images):
    """
    Compute per-channel mean and std across a list of images.
    Each image is assumed to be a NumPy array of shape (H, W, 3).
    """
    means = []
    stds = []
    for img in images:
        means.append(np.mean(img, axis=(0,1)))  # shape (3,)
        stds.append(np.std(img, axis=(0,1)))
    mean_all = np.mean(means, axis=0)
    std_all = np.mean(stds, axis=0)
    return mean_all, std_all

import tensorflow as tf

@st.cache_resource
def load_oct_models(model_paths: dict):
    """
    model_paths: dict of label→.h5 file
    Returns dict of label→tf.keras.Model
    """
    models = {}
    for label, path in model_paths.items():
        models[label] = tf.keras.models.load_model(path, compile=False)
    return models

# e.g.—adjust these paths to wherever your .h5 files live:
OCT_MODEL_PATHS = {
    "Fluid":   "models/best_Fluid.h5",
    "ILM_EZ":  "models/best_ILM_EZ.h5",
    "ILM_RPE": "models/best_ILM_RPE.h5",
    "EZ_RPE":  "models/best_EZ_RPE.h5",
}

oct_models = load_oct_models(OCT_MODEL_PATHS)


# Immediately after your v_* metrics:
for label in oct_models.keys():
    ALL_METRICS[f"{label}_thickness_std"] = f"{label} thickness σ"
    ALL_METRICS_THRESHOLDS[f"{label}_thickness_std"] = (0.0, 50.0)  # choose a clinically plausible upper bound


def prepare_oct_input(orig_bgr: np.ndarray, height=256, width=256):
    # convert BGR→GRAY
    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    # resize and normalize
    img = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(np.expand_dims(img, 0), -1)  # shape (1,H,W,1)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Compute per-pixel gradient magnitude using Sobel."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx*gx + gy*gy)

def compute_gmsd(ref: np.ndarray, dist: np.ndarray, T: float = (0.0026*255)**2) -> float:
    """
    Gradient Magnitude Similarity Deviation between two grayscale images.
    T is a small constant to avoid instability (≈(0.0026*L)^2 for 8-bit L=255).
    """
    g_ref  = gradient_magnitude(ref)
    g_dist = gradient_magnitude(dist)
    # pixel-wise similarity map
    sim_map = (2 * g_ref * g_dist + T) / (g_ref**2 + g_dist**2 + T)
    # GMSD is the standard deviation of that map
    return float(np.std(sim_map))

def compute_additional_iqa_metrics(img_gray: np.ndarray) -> dict:
    """
    Compute SSIM, GMSD, plus NIQE and BRISQUE if installed.
    """
    results = {}
    results['ssim_self']  = compare_ssim(img_gray, img_gray,
                                         data_range=img_gray.max() - img_gray.min())
    results['gmsd_self']  = compute_gmsd(img_gray, img_gray)

    # NIQE
    if NIQE_AVAILABLE:
        try:
            results['niqe'] = niqe.niqe(img_gray)
        except Exception:
            results['niqe'] = None
    else:
        results['niqe'] = None

    # BRISQUE
    if BRISQUE_AVAILABLE:
        try:
            results['brisque'] = brisque.score(img_gray)
        except Exception:
            results['brisque'] = None
    else:
        results['brisque'] = None

    return results


# --------------------------------------
# EXPLAINABILITY: SHAP
# --------------------------------------

def explain_model_with_shap(df_metrics, model, feature_cols):
    """
    Fit SHAP Explainer on regression or classifier model.
    Returns SHAP values and summary plot.
    """
    explainer = shap.Explainer(model.predict, df_metrics[feature_cols])
    shap_values = explainer(df_metrics[feature_cols])
    return shap_values

# --------------------------------------
# CLINICAL REPORT GENERATOR
# --------------------------------------

def generate_clinical_report(df_final, shap_values=None, report_path='report.html'):
    """
    Generate an HTML summary including key metrics, group distributions, and SHAP plots.
    Then export to PDF via WeasyPrint.
    """
    html = []
    html.append("<h1>Clinical Image Quality Report</h1>")
    html.append(f"<p>Generated on: {pd.Timestamp.now()}</p>")
    # Summary table
    html.append(df_final.describe().to_html(classes='table table-striped'))
    # Quality group pie
    pie_fig = px.pie(df_final, names='Quality_Group', title='Quality Group Distribution')
    pie_html = pie_fig.to_html(full_html=False, include_plotlyjs='cdn')
    html.append(pie_html)
    # SHAP summary
    if shap_values is not None:
        shap_fig = shap.plots.beeswarm(shap_values, show=False)
        shap_path = 'shap_summary.png'
        plt.savefig(shap_path, bbox_inches='tight')
        html.append(f"<h2>SHAP Summary</h2><img src='{shap_path}' width='700'>")
    # Save HTML
    with open(report_path, 'w') as f:
        f.write('<html><head><style>.table{width:100%;} </style></head><body>')
        f.write(''.join(html))
        f.write('</body></html>')
    # Export PDF
    HTML(report_path).write_pdf(report_path.replace('.html', '.pdf'))
    return report_path, report_path.replace('.html', '.pdf')

# --------------------------------------
# LOCAL LLM RAG SETUP
# --------------------------------------

def search_web_data(query, max_results=3):
    docs = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get('href')
            resp = requests.get(url, timeout=5)
            if resp.ok:
                text = '\n'.join(p for p in BeautifulSoup(resp.text, 'html.parser').stripped_strings)
                docs.append(Document(page_content=text, metadata={'source': url}))
    return docs

@st.cache_resource
def build_local_rag(doc_folder='clinical_docs'):
    # load existing clinical docs
    local_docs = []
    for fn in os.listdir(doc_folder):
        if fn.endswith('.txt'):
            with open(os.path.join(doc_folder, fn)) as f:
                local_docs.append(Document(page_content=f.read(), metadata={'source': fn}))
    if not local_docs:
        st.warning("🔍 No documents found in clinical_docs/ – RAG interface will be disabled.")
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.from_documents(local_docs, embeddings)
    llm = ChatOllama(model="llama3.2", temperature=0.7)
    system_prompt = """You are a clinical assistant specialized in ophthalmic image quality
    When answering, you should:
    1) Consult any retrieved passages from our local document set.
    2) But also freely draw on your own pretrained medical and imaging knowledge 
     to fill in gaps or add safe, evidence-based context.
     Always cite the local sources when you quote them, but you may also supplement with your own expertise."""
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(),
        chain_type="stuff",
        condense_question_prompt=ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Convert follow-up questions into standalone questions."
            ),
            HumanMessagePromptTemplate.from_template(
                "Chat History:\n{chat_history}\nFollow-up Input:\n{question}\nStandalone question:"
            ),
        ]),
        return_source_documents=True
    )
    qa_chain.combine_docs_chain.llm_chain.prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{question}")
    ])
    return qa_chain

rag_chain = build_local_rag()

def apply_image_batch_correction(uploaded_files, modality, batch_ids):
    """
    Reads all uploaded images and groups them by batch.
    Then computes a global reference (mean, std) and per-batch (mean, std).
    Each image is corrected by:
      corrected = (image - batch_mean) / (batch_std + epsilon) * global_std + global_mean
    Returns a dict mapping filename to corrected image (RGB, uint8).
    """
    images_by_batch = {}
    original_images = {}  # filename -> image
    for uf in uploaded_files:
        b = batch_ids.get(uf.name, "Batch_1")
        if modality == "OCT":
            img = read_oct_dicom(uf)
            if img is None:
                continue
        else:
            pil_image = Image.open(uf).convert("RGB")
            img = np.array(pil_image)
        original_images[uf.name] = img
        images_by_batch.setdefault(b, []).append(img)
    
    # Compute global statistics over all images
    all_images = [img.astype(np.float32) for img in original_images.values()]
    global_mean, global_std = compute_mean_std_for_images(all_images)
    
    # Compute per-batch statistics
    batch_stats = {}
    for b, imgs in images_by_batch.items():
        imgs_float = [img.astype(np.float32) for img in imgs]
        batch_mean, batch_std = compute_mean_std_for_images(imgs_float)
        batch_stats[b] = (batch_mean, batch_std)
    
    # Correct each image
    corrected_images = {}
    for fname, img in original_images.items():
        b = batch_ids.get(fname, "Batch_1")
        batch_mean, batch_std = batch_stats[b]
        img_float = img.astype(np.float32)
        corrected = (img_float - batch_mean) / (batch_std + 1e-8) * global_std + global_mean
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        corrected_images[fname] = corrected
    return corrected_images

# --------------------------------------
# METRIC FUNCTION: ILLUMINATION UNIFORMITY
# --------------------------------------
def compute_illumination_uniformity(gray_img: np.ndarray) -> float:
    h, w = gray_img.shape
    center_crop = gray_img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    full_mean = np.mean(gray_img)
    center_mean = np.mean(center_crop)
    if full_mean == 0:
        return 0.0
    return center_mean / (full_mean + 1e-8)

# --------------------------------------
# LOCAL METRIC FUNCTIONS (Patch-level)
# --------------------------------------
LOCAL_METRIC_FUNCS = {
    "mean": lambda patch: np.mean(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)),
    "var": lambda patch: np.var(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)),
    "range": lambda patch: np.ptp(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)),
    "cv": (lambda patch: (np.std(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)) /
                           max(np.mean(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)), 1e-6) * 100)),
    "entropy": lambda patch: shannon_entropy(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)),
    "contrast": lambda patch: ((float(np.max(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY))) -
                                float(np.min(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)))) /
                               max(float(np.max(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY))) +
                                   float(np.min(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY))), 1e-6)),
    "blur": lambda patch: np.var(cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY), cv2.CV_64F)),
    "edge_density": lambda patch: (np.count_nonzero(cv2.Canny(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY), 100, 200)) /
                                   (patch.shape[0] * patch.shape[1])),
}


# --------------------------------------
# OCT (DICOM) READING
# --------------------------------------
def read_oct_dicom(uploaded_file):
    if not DICOM_AVAILABLE:
        st.error("pydicom is not installed. OCT DICOM files cannot be processed.")
        return None
    try:
        ds = pydicom.dcmread(uploaded_file)
        img_array = ds.pixel_array
        if img_array.dtype != np.uint8:
            img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if len(img_array.shape) == 2:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_array
        return img_rgb
    except Exception as e:
        st.error(f"Error reading DICOM file: {e}")
        return None

# --------------------------------------
# FUNDUS & OCTA PREPROCESSING
# --------------------------------------
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
    cropped_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)[y : y + h, x : x + w]
    return cropped_img, mask

def detect_foreground_fundus(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def compute_edge_density(img_bgr, mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    fg_edges = edges[mask == 255]
    return np.count_nonzero(fg_edges) / fg_edges.size if fg_edges.size > 0 else -1

# --------------------------------------
# QUALITY METRIC COMPUTATION
# --------------------------------------
def compute_quality_metrics(img_bgr, mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fg_pixels = gray[mask == 255]
    bg_pixels = gray[mask == 0]
    if fg_pixels.size == 0 or bg_pixels.size == 0:
        ret = {m: -1 for m in ALL_METRICS if m not in ["Batch_Effect_Index", "illumination_uniformity"]}
        ret["illumination_uniformity"] = 0.0
        return ret
    mean_val = np.mean(fg_pixels)
    var_val = np.var(fg_pixels)
    range_val = np.ptp(fg_pixels)
    std_val = np.std(fg_pixels)
    cv_val = (std_val / mean_val) * 100 if mean_val != 0 else -1
    entropy_val = shannon_entropy(fg_pixels)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_val = np.var(lap[mask == 255])
    min_val, max_val = float(np.min(fg_pixels)), float(np.max(fg_pixels))
    contrast_val = (max_val - min_val) / max((max_val + min_val), 1e-6)
    mse = np.mean((fg_pixels - mean_val) ** 2)
    psnr_val = 10 * np.log10(255 ** 2 / max(mse, 1e-6))
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
    illum_uni = compute_illumination_uniformity(gray)
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
        "edge_density": edge_density_val,
        "illumination_uniformity": illum_uni,
    }

def normalize_df(df):
    norm_df = df.copy()
    for col in ALL_METRICS.keys():
        if col in norm_df.columns:
            valid_mask = (norm_df[col] != -1)
            if valid_mask.sum() > 1:
                vmin = norm_df.loc[valid_mask, col].min()
                vmax = norm_df.loc[valid_mask, col].max()
                if vmax > vmin:
                    norm_df.loc[valid_mask, col] = ((norm_df.loc[valid_mask, col] - vmin) /
                                                    (vmax - vmin))
                else:
                    norm_df.loc[valid_mask, col] = 0
            else:
                norm_df[col] = 0
    return norm_df

# --------------------------------------
# BATCH EFFECT CORRECTION ON FEATURES
# --------------------------------------
def compute_batch_effect_index(df, metric_cols, batch_col='batch'):
    scaler = StandardScaler()
    scaled_metrics = scaler.fit_transform(df[metric_cols])
    df_scaled = pd.DataFrame(scaled_metrics, columns=metric_cols, index=df.index)
    centroids = df_scaled.groupby(df[batch_col]).mean()
    bei_list = []
    for idx, row in df_scaled.iterrows():
        b = df.loc[idx, batch_col]
        centroid = centroids.loc[b].values.reshape(1, -1)
        distance = cdist(row.values.reshape(1, -1), centroid, metric='euclidean')[0][0]
        bei_list.append(distance)
    return bei_list



def correct_batch_effect_in_features(df, metric_cols, batch_col='batch'):
    # — only operate on metrics that actually exist in df —
    metric_cols = [m for m in metric_cols 
                   if m in df.columns and m != "Batch_Effect_Index"]
    if not metric_cols:
        # nothing to batch-correct: just return the original
        return df.copy()

    # first try NeuroCombat if available
    if COMBAT_AVAILABLE:
        try:
            data = df[metric_cols].values
            covars = df[[batch_col]].copy()
            covars.columns = ["batch"]
            covars["batch"] = covars["batch"].astype("category").cat.codes
            combat = CombatModel()
            data_corrected = combat.fit_transform(data, covars)

            df_corrected = df.copy()
            for i, col in enumerate(metric_cols):
                df_corrected[col] = data_corrected[:, i]
            df_corrected["Batch_Effect_Index"] = compute_batch_effect_index(
                df_corrected, metric_cols, batch_col
            )
            return df_corrected

        except Exception as e:
            st.error(f"NeuroCombat batch correction failed: {e}")

    # fallback: simple z-score alignment
    df_corrected = df.copy()
    overall_means = df[metric_cols].mean()
    overall_stds  = df[metric_cols].std()

    for b in df[batch_col].unique():
        idx = df[batch_col] == b
        batch_means = df.loc[idx, metric_cols].mean()
        batch_stds  = df.loc[idx, metric_cols].std()
        for col in metric_cols:
            if batch_stds[col] == 0:
                df_corrected.loc[idx, col] = overall_means[col]
            else:
                df_corrected.loc[idx, col] = (
                    (df.loc[idx, col] - batch_means[col]) / batch_stds[col]
                ) * overall_stds[col] + overall_means[col]

    df_corrected["Batch_Effect_Index"] = compute_batch_effect_index(
        df_corrected, metric_cols, batch_col
    )
    return df_corrected


def simple_batch_correct_metrics(df, batch_col="batch"):
    # only correct on the metrics you actually computed
    numeric_cols = [m for m in get_available_metrics(df, ALL_METRICS)
                    if m != "Batch_Effect_Index"]

    if COMBAT_AVAILABLE:
        try:
            data = df[numeric_cols].values
            covars = df[[batch_col]].copy()
            covars.columns = ['batch']
            covars['batch'] = covars['batch'].astype('category').cat.codes
            combat = CombatModel()
            data_combat = combat.fit_transform(data, covars)
            df_corrected = df.copy()
            for i, col in enumerate(numeric_cols):
                df_corrected[col] = data_combat[:, i]
            return df_corrected
        except Exception as e:
            st.error(f"NeuroCombat batch correction failed: {e}")
    df_corrected = df.copy()
    # if nothing to correct, just return original
    if not numeric_cols:
        return df.copy()

    overall_means = df[numeric_cols].mean()
    overall_stds  = df[numeric_cols].std()


    for b in df[batch_col].unique():
        idx = df[batch_col] == b
        batch_means = df.loc[idx, numeric_cols].mean()
        batch_stds = df.loc[idx, numeric_cols].std()
        for col in numeric_cols:
            if batch_stds[col] == 0:
                df_corrected.loc[idx, col] = overall_means[col]
            else:
                df_corrected.loc[idx, col] = ((df.loc[idx, col] - batch_means[col]) / batch_stds[col]) * overall_stds[col] + overall_means[col]
    return df_corrected

# --------------------------------------
# VISUALIZATIONS
# --------------------------------------
def plot_batch_effects(df, corrected=False):
    title_suffix = " (Corrected)" if corrected else " (Original)"
    st.markdown(f"#### PCA {title_suffix}")
    pca_model = PCA(n_components=2)
    metric_cols_for_pca = [m for m in get_available_metrics(df, ALL_METRICS)
                            if m != "Batch_Effect_Index"]
    features = df[metric_cols_for_pca].fillna(0).values
    pcs = pca_model.fit_transform(features)
    plot_df = df.copy()
    plot_df["PC1"] = pcs[:, 0]
    plot_df["PC2"] = pcs[:, 1]
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="batch",
                     title=f"Batch Effects {title_suffix}", template="plotly_white")
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(legend_font_size=14)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Clinical Insight**:  
    Batch effects may reflect scanner or date-based differences.
    After correction, each batch should cluster less distinctly.
    """)

def plot_metric_distributions(df):
    st.markdown("### Quality Metric Distributions by Batch")
    for metric in get_available_metrics(df, ALL_METRICS):
        if metric in df.columns and metric != "Batch_Effect_Index":
            fig = px.box(df, x="batch", y=metric,
                         title=f"{metric} distribution by Batch", template="plotly_white")
            fig.update_traces(marker=dict(size=12))
            fig.update_layout(legend_font_size=14)
            st.plotly_chart(fig, use_container_width=True)



def batch_effects_3d(df):
    st.markdown("## Batch Effects in 3D (PCA)")
    st.markdown("### Before Correction")
    pca_model = PCA(n_components=3)
    
    metric_cols_for_pca = [m for m in get_available_metrics(df, ALL_METRICS)
                            if m != "Batch_Effect_Index"]
    features = df[metric_cols_for_pca].fillna(0).values
    pcs = pca_model.fit_transform(features)
    df_3d = df.copy()
    df_3d["PC1"], df_3d["PC2"], df_3d["PC3"] = pcs[:, 0], pcs[:, 1], pcs[:, 2]
    fig_before = px.scatter_3d(df_3d, x="PC1", y="PC2", z="PC3", color="batch",
                               title="3D PCA Before Batch Correction", template="plotly_white")
    fig_before.update_traces(marker=dict(size=12))
    fig_before.update_layout(legend_font_size=14)
    st.plotly_chart(fig_before, use_container_width=True)
    df_corrected = simple_batch_correct_metrics(df, batch_col="batch")
    pca_model2 = PCA(n_components=3)
    features_corr = df_corrected[metric_cols_for_pca].fillna(0).values
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
    **Clinical Insight**:  
    If the clusters remain distinct even after correction,
    it may imply more severe differences in acquisition or additional confounders.
    """)

def explain_failure(metrics_row, threshold_dict):
    explanations = []
    for m, desc in ALL_METRICS.items():
        if m not in metrics_row:
            continue
        val = metrics_row.get(m, -1)
        low, high = threshold_dict.get(m, (None, None))
        if low is not None and high is not None:
            if val == -1:
                explanations.append(f"'{m}': Not computed (value = -1).")
            elif val < low:
                explanations.append(f"'{m}' too low ({val:.2f} < {low}).")
            elif val > high:
                explanations.append(f"'{m}' too high ({val:.2f} > {high}).")
    return "\n".join(explanations)

def local_contrast(patch_rgb):
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    return np.ptp(gray)

def layer_thickness_std(mask: np.ndarray):
    """
    mask: H×W binary mask of one layer (0 or 255)
    returns thickness = std of (max_row - min_row) across columns
    """
    rows = []
    bin_mask = (mask > 0).astype(int)
    for col in range(bin_mask.shape[1]):
        ys = np.where(bin_mask[:,col]==1)[0]
        if ys.size:
            rows.append(ys.max() - ys.min())
    return float(np.std(rows)) if rows else 0.0

def compute_local_metric(img_rgb, patch_size=50, metric_func=None):
    h, w = img_rgb.shape[:2]
    heatmap = np.zeros((h, w))
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img_rgb[i : i + patch_size, j : j + patch_size]
            val = metric_func(patch) if metric_func else np.ptp(patch)
            heatmap[i : i + patch_size, j : j + patch_size] = val
    hm_min, hm_max = heatmap.min(), heatmap.max()
    if hm_max > hm_min:
        heatmap = (heatmap - hm_min) / (hm_max - hm_min + 1e-6)
    else:
        heatmap[:] = 0
    return heatmap

def radar_failure_chart(metrics_row, threshold_dict, selected_metrics):
    categories = []
    norm_values = []
    for m in selected_metrics:
        val = metrics_row.get(m, -1)
        low, high = threshold_dict[m]
        mid = (low + high) / 2.0
        range_half = (high - low) / 2.0 if (high - low) != 0 else 1.0
        norm_val = (val - mid) / range_half
        categories.append(m)
        norm_values.append(norm_val)
    norm_values.append(norm_values[0])
    categories.append(categories[0])
    fig = go.Figure(data=go.Scatterpolar(r=norm_values, theta=categories, fill='toself', name="Normalized Deviation"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-2, 2], dtick=1, tickvals=[-2, -1, 0, 1, 2])),
                      showlegend=False,
                      title="Failure Analysis Radar Chart")
    return fig

# --------------------------------------
# IMAGE PROCESSING PIPELINE
# --------------------------------------
def process_image_for_metrics(uploaded_file, image_number, selected_metrics, threshold_dict,
                              batch_id="default", modality="Fundus", img_array_override=None):
    if img_array_override is not None:
        if modality == "OCT":
            img_rgb = img_array_override
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            img_array = img_array_override
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            mask = detect_foreground_fundus(img_bgr)
    else:
        if modality == "OCT":
            metrics_dict = compute_quality_metrics(img_bgr, mask)
            img_rgb = read_oct_dicom(uploaded_file)
            if img_rgb is None:
                return None
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif modality == "Fundus":
            metrics_dict = compute_quality_metrics(img_bgr, mask)

            pil_image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(pil_image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            mask = detect_foreground_fundus(img_bgr)
        elif modality == "OCTA":
            metrics_dict = compute_quality_metrics(img_bgr, mask)

            pil_image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(pil_image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            import opsfaz
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            faz_mask, faz_area_val, cnt = opsfaz.detectFAZ(gray, mm=3, prof=0, precision=0.7)
            mask = (faz_mask * 255).astype(np.uint8)


            # obtain your binary vessel mask from the UNet inference:
            inp = prepare_oct_input(img_bgr) 
            v_pred = vessel_unet.predict(inp)[0,0]
            vessel_mask = (v_pred > 0.5).astype(np.uint8)
            # Now compute all the features:
            m = load_mask_array(vessel_mask)
            s = make_skel(m)
            feats = {}
            feats["v_area"]         = area(m)
            feats["v_density"]      = density(m)
            feats["v_skel_len"]     = skeleton_length(s)
            feats.update(branch_end_counts(s))
            feats.update(diameters(m, s))
            feats["v_fractal_dim"]  = fractal_dimension(s)
            feats["v_lacunarity"]   = lacunarity(m)
            feats["v_tortuosity"]   = tortuosity(s)
            feats.update(glcm_feats(m))
            feats.update(segment_stats(s))
            # merge into your metrics dict:
            metrics_dict.update(feats)
        elif modality == "OCT":
            # orig_img is RGB or BGR array from your pipeline
            # assume img_bgr is your BGR OCT frame
            inp = prepare_oct_input(img_bgr)
            cols = st.columns(len(oct_models)+1)
            cols[0].image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original OCT")
            # Run each layer model
            for i, (label, model) in enumerate(oct_models.items(), start=1):
                pred = model.predict(inp)[0,:,:,0]        # shape (H,W)
                mask = (pred > 0.5).astype(np.uint8) * 255 # binary mask
                # optionally resize back to original
                mask_disp = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
                cols[i].image(mask_disp, caption=label, use_container_width=True)
                # in your loop, after mask_disp created:
                th = layer_thickness_std(mask_disp)
                metrics_dict[f"{label}_thickness_std"] = th

        else:
            pil_image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(pil_image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            mask = detect_foreground_fundus(img_bgr)
    metrics_dict = compute_quality_metrics(img_bgr, mask)
    if modality == "OCTA":
   
        #metrics_dict["faz_area"] = float(area(faz_mask.astype(np.uint8)))
        metrics_dict["faz_area"] = float(faz_area_val)


    filename = f"{image_number}_{uploaded_file.name}"
    metrics_dict["filename"] = filename
    metrics_dict["batch"] = batch_id
    pass_count = 0
    for m in selected_metrics:
        val = metrics_dict.get(m, -1)
        if val == -1:
            continue
        low, high = threshold_dict[m]
        if low <= val <= high:
            pass_count += 1
    is_pass = (pass_count >= 0.75 * len(selected_metrics))
    metrics_dict["is_pass"] = is_pass
    segmented_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    mask_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    save_dir = PASSED_DIR if is_pass else FAILED_DIR
    cv2.imwrite(os.path.join(save_dir, filename), img_bgr)
    return {
        "original_bgr": img_bgr,
        "mask_bgr": mask_disp,
        "segmented_bgr": segmented_img,
        "metrics": metrics_dict,
        "is_pass": is_pass,
    }

# --------------------------------------
# QUALITY GROUP ANALYSIS
# --------------------------------------
def assign_quality_group(df):
    df = df.copy()
    df["Pass_Status"] = df["is_pass"].apply(lambda x: "Passed" if x else "Failed")
    passed = df[df["is_pass"]]
    if not passed.empty:
        passed = passed.copy()
        passed["Quality_Score"] = (passed.get("psnr", 0) +
                                   passed.get("snr1", 0) +
                                   passed.get("contrast", 0) -
                                   passed.get("blur", 0))
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
    all_metric_cols = list(ALL_METRICS.keys())
    avg_metrics = df.groupby("Quality_Group")[all_metric_cols].mean().reset_index()
    fig_bar = px.bar(avg_metrics, x="Quality_Group", y=avg_metrics.columns[1:],
                     title="Average Quality Metrics by Group",
                     barmode="group", color="Quality_Group",
                     color_discrete_map=COLOR_MAP, template="plotly_white")
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

# --------------------------------------
# CLINICAL DASHBOARD VISUALIZATIONS
# --------------------------------------
def overall_quality_gauge(df):
    passed = df[df["is_pass"]]
    if passed.empty:
        st.info("No passed images to display overall quality gauge.")
        return
    passed = passed.copy()
    passed["Quality_Score"] = (passed.get("psnr", 0) +
                               passed.get("snr1", 0) +
                               passed.get("contrast", 0) -
                               passed.get("blur", 0))
    median_score = passed["Quality_Score"].median()
    fig = go.Figure(go.Indicator(mode="gauge+number", value=median_score,
                                 title={"text": "Median Quality Score"},
                                 gauge={"axis": {"range": [None, passed["Quality_Score"].max()]}}))
    st.plotly_chart(fig, use_container_width=True)

def clinical_summary(df):
    df = assign_quality_group(df)
    group_counts = df["Quality_Group"].value_counts().reset_index()
    group_counts.columns = ["Quality_Group", "Count"]
    fig = px.pie(group_counts, values="Count", names="Quality_Group", title="Quality Group Distribution")
    st.plotly_chart(fig, use_container_width=True)

def custom_3d_scatter(df):
    st.markdown("## Custom 3D Scatter Plot")
    norm_df = normalize_df(df)

    available_cols = get_available_metrics(df, ALL_METRICS)
    x_axis = st.selectbox("X-axis metric:", available_cols, index=0)
    y_axis = st.selectbox("Y-axis metric:", available_cols, index=1)
    z_axis = st.selectbox("Z-axis metric:", available_cols, index=2)
    if "Quality_Group" not in norm_df.columns:
        norm_df["Quality_Group"] = assign_quality_group(df)["Quality_Group"]
    fig = px.scatter_3d(norm_df, x=x_axis, y=y_axis, z=z_axis, color="Quality_Group",
                          color_discrete_map=COLOR_MAP, title="Custom 3D Scatter Plot (Quality Groups)",
                          template="plotly_white")
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(legend_font_size=14)
    st.plotly_chart(fig, use_container_width=True)

def interactive_3d_analysis(df):
    st.markdown("## Interactive 3D Analysis of Quality Metrics")
    method_options = ["PCA", "t-SNE"]
    if UMAP_AVAILABLE:
        method_options.append("UMAP")
    method = st.selectbox("Select Dimensionality Reduction Method:", options=method_options)
    
    avail = get_available_metrics(df, ALL_METRICS)
    feats = normalize_df(df)[avail].fillna(0).values
    if method == "PCA":
        dr_model = PCA(n_components=3)
        X_dr = dr_model.fit_transform(feats)
    elif method == "t-SNE":
        n_samples = feats.shape[0]
        perplexity = 5 if n_samples > 5 else max(1, n_samples - 1)
        dr_model = TSNE(n_components=3, perplexity=perplexity, random_state=42)
        X_dr = dr_model.fit_transform(feats)
    else:
        dr_model = UMAP(n_components=3)
        X_dr = dr_model.fit_transform(feats)
    display_df = normalize_df(df).copy()
    display_df["Dim1"], display_df["Dim2"], display_df["Dim3"] = X_dr[:, 0], X_dr[:, 1], X_dr[:, 2]
    if "Quality_Group" not in display_df.columns:
        display_df["Quality_Group"] = assign_quality_group(df)["Quality_Group"]
    fig_dr = px.scatter_3d(display_df, x="Dim1", y="Dim2", z="Dim3", color="Quality_Group",
                           hover_data=display_df.columns, title=f"{method} 3D Interactive Plot (Colored by Quality Group)",
                           template="plotly_white", color_discrete_map=COLOR_MAP)
    fig_dr.update_traces(marker=dict(size=12))
    fig_dr.update_layout(legend_font_size=14)
    st.plotly_chart(fig_dr, use_container_width=True)
    st.markdown("""
    **Clinical Insight**:  
    Dimensionality reduction can reveal underlying clusters in image quality.
    Post-batch correction, data from different sites should be more intermixed.
    """)

def surface_correlation_plot(df):
    st.markdown("## 3D Surface Plot of Metric Correlation")
    
    metrics_names = get_available_metrics(df, ALL_METRICS)
    corr = df[metrics_names].corr().values
    x = np.arange(len(metrics_names))
    y = np.arange(len(metrics_names))
    fig = go.Figure(data=[go.Surface(z=corr, x=x, y=y)])
    fig.update_layout(title="Correlation Surface",
                      scene=dict(xaxis=dict(title='Metric', tickmode='array', tickvals=x, ticktext=metrics_names),
                                 yaxis=dict(title='Metric', tickmode='array', tickvals=y, ticktext=metrics_names),
                                 zaxis=dict(title='Correlation')),
                      legend_font_size=14)
    st.plotly_chart(fig, use_container_width=True)

def static_visualizations(df):
    st.markdown("## Static Visualizations")
    display_metric = normalize_df(df)
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
    st.markdown("### Violin Plot (Normalized)")
    fig, ax = plt.subplots(figsize=(10, 5))

    numeric_cols = get_available_metrics(display_metric, ALL_METRICS)
    sns.violinplot(data=display_metric[numeric_cols], scale="width", inner="quartile", palette="coolwarm", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    st.pyplot(fig)
    plt.close('all')
    st.markdown("### Correlation Heatmap")
    corr = display_metric[numeric_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
    plt.close('all')
    st.markdown("### Radar Chart of Average Metrics by Quality Group")
    df_qg = assign_quality_group(df)
    avg_qg = df_qg.groupby("Quality_Group")[numeric_cols].mean().reset_index()
    categories = numeric_cols
    fig_radar = go.Figure()
    for group in avg_qg["Quality_Group"].unique():
        group_data = avg_qg[avg_qg["Quality_Group"] == group]
        values = group_data[categories].values.flatten().tolist()
        if len(values) == 0:
            continue
        values += values[:1]
        fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories + [categories[0]], fill='toself', name=str(group)))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True,
                            title="Radar Chart of Average Metrics by Quality Group", legend_font_size=14)
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown("""
    **Clinical Insight**:  
    These static plots can show anomalies or relationships among metrics
    (e.g., correlation between blur & PSNR) that might prompt re-imaging.
    """)

def documentation_tab():
    st.markdown("# Documentation & Clinical Explanations")
    st.markdown("## Overview")
    st.markdown("""
    This tool adapts the MRI-based batch effect correction from the MRQy paper (Sadri et al. 2020)
    to ophthalmic images (OCT, OCTA, Fundus). It identifies site or scanner-specific variations (batch effects)
    and corrects them at both the feature and image levels.
    
    **Image-Level Correction**:  
    Before computing quality metrics, images can be corrected by a linear intensity normalization—
    aligning each batch’s mean and standard deviation to a global reference. This ensures that
    any future feature extraction operates on batch-homogenized images.
    """)
    st.markdown("## Batch Effect Correction Workflow")
    st.markdown("""
    1. **Extract Metrics**: Compute standard quality metrics (PSNR, entropy, blur, etc.) from images.
    2. **Image-Level Correction (Optional)**:  
       - Compute per-batch and global intensity statistics (per channel).  
       - Correct each image so that its intensity distribution matches the global reference.
    3. **Feature-Level Correction**:  
       - For the computed metrics, adjust for batch effects using either Combat (if available) or a Z-score alignment.
       - Recompute the Batch Effect Index (BEI).
    4. **Final Quality Score**:  
       - Combine normalized metrics and penalize high BEI to compute an overall score.
    """)
    st.markdown("## Overall Quality Score Integration")
    st.markdown(r"""
    We penalize images with high BEI via a final score:
    \[
       Q = \bar{M} - w \times BEI_{\mathrm{norm}}
    \]
    where \( \bar{M} \) is the mean of normalized metrics and \( BEI_{\mathrm{norm}} \) is the normalized BEI.
    """)
    st.markdown("## Clinical Considerations")
    st.markdown("""
    - **High BEI** suggests unusual scanner parameters or severe artifacts.
    - **Image-Level Correction** minimizes batch-specific intensity differences, making downstream feature extraction more robust.
    """)
    st.markdown("## Summary")
    st.markdown("""
    By integrating both image-level and feature-level batch corrections, this pipeline aims to reduce
    batch effects so that subsequent analyses and any additional feature extractions are less biased
    by differences in acquisition conditions.
    """)
    st.markdown("# 👩‍⚕️ Clinician’s Guide")
    st.markdown("### Modalities & Applicable Metrics")
    st.markdown("""
        - **Fundus**  
          • Intensity & contrast: `mean`, `var`, `range`, `cv`, `entropy`  
          • Noise / signal: `snr1–4`, `psnr`, `cnr`  
          • Sharpness: `blur` (Laplacian variance)  
          • Illumination uniformity  

        - **OCTA** (all of the above, plus):  
          • Vessel geometry: `v_area`, `v_density`, `v_skel_len`, `v_branch_pts`, `v_end_pts`  
          • Vessel size & shape: `v_mean_diam`, `v_std_diam`, `v_fractal_dim`, `v_lacunarity`, `v_tortuosity`  
          • Vessel texture: `v_glcm_contrast`, `v_glcm_correlation`, `v_glcm_energy`, `v_glcm_homogeneity`  
          • Segment counts & lengths: `v_n_segments`, `v_mean_seg_len`, `v_std_seg_len`  
          • FAZ area: `faz_area`  

        - **OCT** (all of the above base metrics, plus):  
          • Layer-segmentation stability (σ of layer thickness):  
            `Fluid_thickness_std`, `ILM_EZ_thickness_std`, `ILM_RPE_thickness_std`, `EZ_RPE_thickness_std`  

        - **Batch metrics** (across all modalities):  
          • `Batch_Effect_Index` (BEI)  
          • `Overall_Quality_Score` (combines normalized metrics & BEI penalty)
        """)
    st.markdown("### Recommended Thresholds")
 
    st.markdown("### Features & Functionality")
    st.markdown("""
        - **Image-level batch correction** aligns each batch’s intensity distribution to a global reference.  
        - **Feature-level batch correction** (Combat or Z-score) removes scanner/site effects on metrics.  
        - **Interactive plots**: 2D/3D PCA, t-SNE, UMAP, parallel coordinates, radar charts, local heatmaps.  
        - **Explainability** via SHAP for your overall quality model.  
        - **Clinical report**: downloadable HTML/PDF summary with tables, Pie charts, SHAP.  
        """)

# --------------------------------------
# FINAL SCORE: INTEGRATING BEI
# --------------------------------------
def integrate_quality_score(df, metric_cols=None, bei_weight=0.3):
    if metric_cols is None:
        metric_cols = [m for m in ALL_METRICS.keys() if m != "Batch_Effect_Index"]
    subset = df[metric_cols].fillna(0)
    scaler = StandardScaler()
    scaled_metrics = scaler.fit_transform(subset)
    quality_raw = scaled_metrics.mean(axis=1)
    if 'Batch_Effect_Index' not in df.columns:
        df['Batch_Effect_Index'] = 0.0
    bei = df['Batch_Effect_Index'].fillna(0)
    bei_norm = (bei - bei.min()) / (bei.max() - bei.min() + 1e-8)
    overall = quality_raw - bei_weight * bei_norm
    df['Overall_Quality_Score'] = overall
    return df

def identify_outliers(df, bei_threshold=2.5):
    if 'Batch_Effect_Index' not in df.columns:
        return pd.DataFrame()
    threshold_val = df['Batch_Effect_Index'].mean() + bei_threshold * df['Batch_Effect_Index'].std()
    df['Outlier_Flag'] = df['Batch_Effect_Index'] > threshold_val
    return df[df['Outlier_Flag']]

def visualize_bei_quality(df):
    if 'Batch_Effect_Index' not in df.columns or 'Overall_Quality_Score' not in df.columns:
        st.write("Batch Effect Index or Overall Quality Score not found.")
        return
    fig = px.scatter(df, x='Batch_Effect_Index', y='Overall_Quality_Score', color='batch',
                     hover_data=['filename'], title='Batch Effect Index vs Overall Quality Score')
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------
# MAIN STREAMLIT APP
# --------------------------------------
def main():

    import shap
    from weasyprint import HTML
    # Additional IQA metrics
    import brisque
    import niqe

    Image.MAX_IMAGE_PIXELS = None

# Explainability
    import shap

# Local LLM & RAG
    from langchain_community.vectorstores import FAISS
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate)
    from langchain_ollama import ChatOllama
    from duckduckgo_search import DDGS

# Export to PDF
    from weasyprint import HTML
    st.subheader("Clinical Q&A 🤖")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

   # render past messages
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # input box
    user_q = st.chat_input("Ask about image quality or workflow…")
    if user_q:
       st.session_state.chat_history.append({"role":"user","content":user_q})
       #resp = rag_chain({"question": user_q, "chat_history": []})
       resp = rag_chain.invoke({"question": user_q, "chat_history": []})

       answer = resp["answer"]
       st.session_state.chat_history.append({"role":"assistant","content":answer})
       st.chat_message("assistant").markdown(answer)
    # optionally show sources
       if resp.get("source_documents"):
          st.markdown("**Sources:**")
          for doc in resp["source_documents"]:
            st.write(f"- {doc.metadata['source']}")

    st.title("Image Quality Dashboard")
    st.markdown("""
    **Enhanced for OCT/OCTA/Fundus** with ideas from MRQy:
     - Batch Effect Correction at both the feature and image levels.
     - Additional metrics (illumination uniformity, etc.).
     - Interactive dashboards & documentation.
    """)
    modality = st.sidebar.selectbox("Select Image Modality", ["Fundus", "OCTA", "OCT"])
    st.sidebar.title("Upload Images")

    # 1) Ask how many batches the user needs
    num_batches = st.sidebar.number_input(
        "How many batches?", min_value=1, max_value=20, value=1, step=1
    )

    # 2) Create one uploader per batch
    uploaded_batches = {}
    for i in range(1, num_batches + 1):
        label = f"Batch {i} images"
        if modality == "OCT":
            files = st.sidebar.file_uploader(
                label, type=["dcm"], accept_multiple_files=True, key=f"batch_{i}"
            )
        else:
            files = st.sidebar.file_uploader(
                label, type=["png", "jpg", "jpeg"], accept_multiple_files=True, key=f"batch_{i}"
            )
        if files:
            uploaded_batches[f"Batch_{i}"] = files

    # 3) Flatten into a single list and build batch_ids map
    all_files = []
    batch_ids = {}
    for batch_name, files in uploaded_batches.items():
        all_files.extend(files)
        for uf in files:
            batch_ids[uf.name] = batch_name

     
    # 4) From here on, use `all_files` instead of `uploaded_files`
    uploaded_files = all_files

    process_mode = st.sidebar.radio("Select Processing Mode", ("Quality Metrics", "Preprocessing"))

    # Option to apply image-level batch correction
    apply_img_correction = st.sidebar.checkbox("Apply image-level batch correction", value=True)
    if process_mode == "Quality Metrics":
        st.sidebar.markdown("### Select Metrics for Pass/Fail")
        available_metrics = list(ALL_METRICS.keys())
        selected_metrics = st.sidebar.multiselect("Choose metrics:", available_metrics, default=DEFAULT_CRITERIA)
        if not selected_metrics:
            st.sidebar.warning("No metrics selected. Using default set.")
            selected_metrics = DEFAULT_CRITERIA
        st.sidebar.markdown("### Define Thresholds for Each Selected Metric")
        threshold_dict = {}
        for m in selected_metrics:
            default_low, default_high = ALL_METRICS_THRESHOLDS.get(m, (0, 300))
            if default_low == float('-inf'):
                default_low = 0
            if default_high == float('inf'):
                default_high = 300
            low_high = st.sidebar.slider(f"{m} ({ALL_METRICS[m]})", 0.0, 300.0,
                                         (float(default_low), float(default_high)), key=f"thresh_{m}")
            threshold_dict[m] = (low_high[0], low_high[1])
        bei_weight = st.sidebar.slider("Penalty weight for BEI in overall quality score:", 0.0, 2.0, 0.3, 0.05)
        corrected_images = {}
        if uploaded_files and apply_img_correction:
            st.info("Applying image-level batch correction...")
            corrected_images = apply_image_batch_correction(uploaded_files, modality, batch_ids)
        if uploaded_files:
            results = []
            passed_images_data = []
            failed_images_data = []
            for idx, uf in enumerate(uploaded_files, start=1):
                bid = batch_ids.get(uf.name, "Batch_1")
                img_override = corrected_images.get(uf.name) if apply_img_correction else None
                processed = process_image_for_metrics(uf, idx, selected_metrics, threshold_dict,
                                                      batch_id=bid, modality=modality, img_array_override=img_override)
                
                if processed is None:
                   continue

                metrics = processed["metrics"]

                # right after `processed = process_image_for_metrics(...)`
                orig_bgr = processed["original_bgr"]
                gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
                add_iqa = compute_additional_iqa_metrics(gray)
                # merge those into your metrics dict
                metrics.update(add_iqa)

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
        
        
            metric_cols = get_available_metrics(df, ALL_METRICS)
            df_corrected = correct_batch_effect_in_features(df, metric_cols, batch_col='batch')


            df_final = integrate_quality_score(df_corrected, metric_cols=metric_cols, bei_weight=bei_weight)
            outliers_df = identify_outliers(df_final, bei_threshold=2.5)

            # ---- NEW: SHAP explainability ----
            # select only the numeric columns
            shap_features = [c for c in df_final.columns
                 if c in ALL_METRICS.keys()
                 or c in ["ssim_self","gmsd_self","niqe","brisque"]]

            # build a clean feature matrix
            feature_df = (df_final[shap_features].fillna(0).astype(float))

           # train a simple ridge to predict your overall score
            from sklearn.linear_model import Ridge
            target     = df_final["Overall_Quality_Score"]
            reg_model  = Ridge(alpha=1.0).fit(feature_df, target)

            # use the linear explainer (no masking of object columns)
            import shap
            explainer  = shap.LinearExplainer(reg_model, feature_df)
            shap_values = explainer(feature_df)

            st.subheader("Feature Importance (SHAP)")
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(bbox_inches="tight")


            
            # 3. Clinical report
    
            html_report, pdf_report = generate_clinical_report(df_final, shap_values)

            st.sidebar.markdown("## Clinical Report")
            st.sidebar.download_button("Download HTML Report", open(html_report,'rb'), file_name=os.path.basename(html_report))
            st.sidebar.download_button("Download PDF Report", open(pdf_report,'rb'), file_name=os.path.basename(pdf_report))


            tabs = st.tabs([
                "Metrics Table", "Passed Images", "Failed Images", "Interactive Plots",
                "Interactive 3D Analysis", "Custom 3D Scatter", "Batch Effects (3D)",
                "Static Visualizations", "Batch Effects (2D)", "Quality Group Analysis",
                "Clinical Dashboard", "Batch Effect Analysis", "Clinical Documentation",
            ])
            with tabs[0]:
                st.title("Image Quality Dashboard")
                with st.expander("🔎 Reference: All metrics & thresholds"):
                     st.dataframe(metrics_threshold_table(), use_container_width=True)
                st.subheader("Quality Metrics Table (Post-Correction)")
                st.dataframe(df_final)
                df_thresh = metrics_threshold_table()
                st.markdown("### Metrics & Thresholds Reference")
                st.dataframe(df_thresh, use_container_width=True)
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
                            st.markdown("#### Failure Analysis Radar Chart")
                            radar_fig = radar_failure_chart(metrics_row, threshold_dict, selected_metrics)
                            st.plotly_chart(radar_fig, use_container_width=True)
                else:
                    st.info("No failed images found.")
            with tabs[3]:
                st.subheader("Interactive Plots (All Metrics)")
                norm_df = normalize_df(df_final)
                st.markdown("### Parallel Coordinates")
                color_choices = [m for m in df_final.columns if m in ALL_METRICS.keys()]
                if not color_choices:
                    st.info("No metrics available for parallel coordinates.")
                else:
                    color_metric = st.selectbox("Select metric for coloring:", color_choices, index=0)
                    dims = color_choices
                    fig_px = px.parallel_coordinates(norm_df, dimensions=dims, color=color_metric,
                                                     color_continuous_scale=px.colors.sequential.Plasma,
                                                     color_continuous_midpoint=norm_df[color_metric].mean(),
                                                     template="plotly_white")
                    fig_px.update_layout(legend_font_size=14)
                    st.plotly_chart(fig_px, use_container_width=True)
                st.markdown("""
                **Clinical Insight**:  
                Parallel coordinates let us see multiple metrics at once, highlighting outliers.
                """)
            with tabs[4]:
                interactive_3d_analysis(df_final)
            with tabs[5]:
                custom_3d_scatter(df_final)
            with tabs[6]:
                st.subheader("Batch Effects (3D)")
                batch_effects_3d(df_final)
                st.markdown("### 3D Surface of Metric Correlation")
                surface_correlation_plot(df_final)
            with tabs[7]:
                static_visualizations(df_final)
            with tabs[8]:
                st.subheader("Batch Effects (2D)")
                st.markdown("#### Before Correction")
                plot_batch_effects(df, corrected=False)
                st.markdown("#### After Correction")
                plot_batch_effects(df_final, corrected=True)
                plot_metric_distributions(df_final)
            with tabs[9]:
                display_quality_group_analysis(df_final, selected_metrics, threshold_dict)
            with tabs[10]:
                st.markdown("## Clinical Dashboard")
                st.markdown("### Overall Quality Score Gauge")
                overall_quality_gauge(df_final)
                st.markdown("### Quality Group Distribution")
                clinical_summary(df_final)
            with tabs[11]:
                st.markdown("## Batch Effect Analysis")
                st.write("### Outlier Images Based on Batch Effect Index")
                if outliers_df.empty:
                    st.write("No outliers detected (based on threshold=2.5 std dev).")
                else:
                    st.dataframe(outliers_df[["filename", "batch", "Batch_Effect_Index", "Overall_Quality_Score"]])
                st.write("\n### Visualize Batch Effect vs. Quality")
                visualize_bei_quality(df_final)
            with tabs[12]:
                documentation_tab()
            csv_path = os.path.join(OUTPUT_DIR, "quality_metrics.csv")
            df_final.to_csv(csv_path, index=False)
            with open(csv_path, "rb") as f:
                st.sidebar.download_button("Download Quality Metrics CSV", f, file_name="quality_metrics.csv", mime="text/csv")
            st.markdown("""
            ---
            **Summary of Clinical Insights**  
            - **Batch Effects & BEI**: Both image-level and feature-level corrections are applied.
            - **Failure Analysis**: Radar chart + local heatmaps illustrate which metrics are problematic.
            - **Illumination Uniformity**: Newly included to address lighting issues in Fundus/OCT.
            - **Penalty Weight**: Adjust how strongly BEI influences the final quality score.
            ---
            """)
        else:
            st.info("📂 Upload at least one image to analyze.")
    else:
        if uploaded_files:
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
                    proc_img = cv2.bitwise_and(img_array, img_array, mask=(faz_mask * 255).astype(np.uint8))
                    mask_img = (faz_mask * 255).astype(np.uint8)
                elif modality == "OCT":
                    proc_img = img_array
                    mask_img = np.ones(img_array.shape[:2], dtype=np.uint8) * 255
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
                    cols[0].image(proc_img, caption="Preprocessed Image", use_container_width=True)
                    mask_3ch = np.stack([mask_img] * 3, axis=-1) if mask_img.ndim == 2 else mask_img
                    cols[1].image(mask_3ch, caption="Mask", use_container_width=True)
        else:
            st.info("📂 Upload at least one image to preprocess.")


main()
