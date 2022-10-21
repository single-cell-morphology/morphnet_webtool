import streamlit as st
import anndata
import scanpy as sc
import scvi
import pandas as pd
import numpy as np
import pickle
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_latent_expression, predict_image
import dnnlib
import legacy
import copy
from pathlib import Path
from scipy import sparse

st.write("""
# Generating Neuronal Morphology from scRNAseq data

This web tool predicts the neuronal morphology for a given scRNAseq data.
""")

st.sidebar.header("User Input")

# TODO: Make sure the app runs with an uploaded data; for testing, you can create a simple csv with a single cell

# Collets user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload AnnData (.h5ad) file for raw scRNAseq expression", type=["h5ad"])
adata = None
scvi_model = None
latents = None

if uploaded_file is not None:
    # input_df = pd.read_csv(uploaded_file)
    adata = anndata.read_h5ad(uploaded_file)
else:
    if st.sidebar.button("Use sample Patchseq data (102 cells)"):
        adata = anndata.read_h5ad("./data/patchseq_sample.h5ad")
    else:
        st.sidebar.write("Please upload an h5ad file or use sample Patchseq data.")

### Display Input Data
st.subheader("Step 1. Upload scRNA-seq data")
st.write("""
Please upload an .h5ad file of your scRNA-seq data. The scRNA-seq expression should be an AnnData object, with a `cell_source` column in obs of your AnnData.""")

if adata:
    st.write(f"Your AnnData: {adata}")
else:
    st.write("WARNING: Please upload your own data or click on `Use sample Patchseq data (102 cells)`.")

def preprocess(adata):
    adata.layers["counts"] = adata.X.copy() # preserve counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata # freeze the state in `.raw`

    return adata

def match_genes(source_adata, target_adata):
    target_adata = target_adata[:, target_adata.var_names.isin(source_adata.var_names)]
    # st.write(f"target_adata: {target_adata}")
    unmatched_genes = set(source_adata.var_names).difference(set(target_adata.var_names))
    num_cells = target_adata.shape[0]
    X = sparse.csr_matrix((num_cells, len(unmatched_genes)))
    unmatched_adata = anndata.AnnData(X = X, var = pd.DataFrame(index=unmatched_genes, columns=["genes"]), obs = target_adata.obs.copy())

    matched_adata = anndata.concat([target_adata, unmatched_adata], axis=1)
    matched_adata.obs = target_adata.obs.copy()
    return matched_adata

def get_latent_expression(target_adata):
    # scvi.model.SCVI.setup_anndata(adata)
    path = Path("./models/gene_expression/patchseq_scVI")
    scvi_model = scvi.model.SCVI.load(path)
    source_adata = anndata.read_h5ad(path / "adata.h5ad")
    target_adata = match_genes(source_adata, target_adata)

    scvi.data.setup_anndata(target_adata, batch_key="cell_source")
    latents = scvi_model.get_latent_representation(target_adata)

    return scvi_model, latents

### Preprocess
@st.cache
def get_trained_umap(latent):
    reducer = umap.UMAP(random_state=42).fit(latent)
    return reducer

st.subheader("Step 2. Preprocessing and Obtaining Latent Gene Expression")
if adata:
    adata = preprocess(adata)
    scvi_model, latents = get_latent_expression(adata)
    source_latents = scvi_model.get_latent_representation()
    umap_fn = copy.deepcopy(get_trained_umap(source_latents))
    source_patchseq_embedding = umap_fn.transform(source_latents)
    user_embedding = umap_fn.transform(latents)

    all_adata = scvi_model.adata
    target_ids = adata.obs.index
    source_adata = all_adata[~all_adata.obs.index.isin(target_ids)]
    source_latents = scvi_model.get_latent_representation(source_adata)

    umap_fn = get_trained_umap(source_latents)
    source_patchseq_embedding = umap_fn.embedding_
    user_embedding = umap_fn.transform(latents)

### Plot UMAP with Patchseq UMAP
if adata and scvi_model:
    # Plotting UMAP
    fig, ax = plt.subplots()
    c = np.array(scvi_model.adata.obs["celltype"].values)
    ax.scatter(x = source_patchseq_embedding[:,0], y = source_patchseq_embedding[:,1], s=0.2, alpha=0.4)
    ax.scatter(user_embedding[:,0], user_embedding[:,1], c="red", s=10, alpha=1.0)
    st.write("Below is a combined UMAP plot of your data and Patchseq training data.")
    st.pyplot(fig)
else:
    st.write("Waiting for data and UMAP embeddings...")

### Predict Morphology
st.subheader("Step 3. Generating Morphology")

def get_trained_morphgan():
    device = "cpu"
    network_pkl = "./models/morphology/patchseq/morphgan.pkl"
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    return G

if latents is not None:
    model = get_trained_morphgan()
    img = predict_image(model, np.expand_dims(latents[0], axis=0))

    fig, ax = plt.subplots()
    ax.imshow(img)
    st.pyplot(fig)
else:
    st.write("Waiting for latent gene expresion...")
# # TODO: Add a Streamlit button for users to download their morphological image(s).
st.button("Download Images")