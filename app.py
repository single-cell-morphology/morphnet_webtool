import streamlit as st
import anndata
import scanpy as sc
import scvi
import pandas as pd
import numpy as np
import pickle
import umap
import matplotlib.pyplot as plt
from utils import get_latent_expression, predict_image
import dnnlib
import legacy
import copy
from pathlib import Path

st.write("""
# Generating Neuronal Morphology from scRNAseq data

This web tool predicts the neuronal morphology for a given scRNAseq data.

You can either upload your raw scRNAseq matrix (.csv) or input a scVI latent representation.
""")

st.sidebar.header("User Input")

# TODO: Make sure the app runs with an uploaded data; for testing, you can create a simple csv with a single cell

# Collets user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload AnnData (.h5ad) file for raw scRNAseq expression", type=["h5ad"])
if uploaded_file is not None:
    # input_df = pd.read_csv(uploaded_file)
    adata = anndata.read_h5ad(uploaded_file)
else:
    if st.sidebar.button("Use sample Patchseq data"):
        adata = anndata.read_h5ad("./data/patchseq_sample.h5ad")
    else:
        st.sidebar.write("Please upload an h5ad file or use sample Patchseq data.")

### Display Input Data
st.subheader("User Input Features")

try:
    st.write(adata)
except:
    st.write("Please upload or use sample data.")

# TODO: Eventually, user should upload raw scRNAseq data. Add a Step here to run scVI to get latent gene expression.
def preprocess(adata):
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy() # preserve counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata # freeze the state in `.raw`

    return adata

def match_genes(source_adata, target_adata):
    target_adata = target_adata[:, target_adata.var_names.isin(source_adata.var_names)]
    unmatched_genes = set(source_adata.var_names).difference(set(target_adata.var_names))
    st.write(target_adata)
    return target_adata

def get_latent_expression(target_adata):
    # scvi.model.SCVI.setup_anndata(adata)
    path = Path("./models/gene_expression/patchseq_scVI")
    scvi_model = scvi.model.SCVI.load(path)
    source_adata = anndata.read_h5ad(path / "adata.h5ad")
    target_adata = match_genes(source_adata, target_adata)
    latents = scvi_model.get_latent_representation(target_adata)
    return scvi_model, latents

### Preprocess
st.subheader("Preprocessed scRNAseq")
adata = preprocess(adata)
st.write(adata)

# ValueError: Number of vars in adata_target not the same as source. Expected: 22534 Received: 3552

### scVI Model
st.subheader("scVI model")
scvi_model, latents = get_latent_expression(adata)
st.write(scvi_model)
st.write(latents.shape)

### Plot UMAP with Patchseq UMAP
st.subheader("User Input on Patchseq UMAP")
st.write("Your data projected to latent gene expression space of Patchseq")

@st.cache
def get_trained_umap(latent):
    reducer = umap.UMAP(random_state=42).fit(latent)
    return reducer

patchseq_latent = np.load("./data/patchseq_latents.npy")
umap_fn = copy.deepcopy(get_trained_umap(patchseq_latent))

patchseq_embedding = umap_fn.transform(patchseq_latent)
user_embedding = umap_fn.transform(input_df.values)

fig, ax = plt.subplots()
ax.scatter(patchseq_embedding[:,0], patchseq_embedding[:,1], s=0.2, alpha=0.4)
ax.scatter(user_embedding[:,0], user_embedding[:,1], c="red", s=10, alpha=1.0)
st.pyplot(fig)

### Predict Morphology
st.subheader("Predicted Morphology")

def get_trained_morphgan():
    device = "cpu"
    network_pkl = "./models/morphology/patchseq/morphgan.pkl"
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    return G

# TODO: Have two models - one more nuclear image and one for neuronal image
model = get_trained_morphgan()
img = predict_image(model, input_df.values)

fig, ax = plt.subplots()
ax.imshow(img)
st.pyplot(fig)

# TODO: Add a Streamlit button for users to download their morphological image(s).