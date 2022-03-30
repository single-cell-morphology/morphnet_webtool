import streamlit as st
import pandas as pd
import numpy as np
import pickle
import umap
import matplotlib.pyplot as plt
from utils import get_latent_expression, predict_image
import dnnlib
import legacy
import copy

st.write("""
# Generating Neuronal Morphology from scRNAseq data

This web tool predicts the neuronal morphology for a given scRNAseq data.

You can either upload your raw scRNAseq matrix (.csv) or input a scVI latent representation.
""")

st.sidebar.header("User Input Features")

# TODO: Make sure the app runs with an uploaded data; for testing, you can create a simple csv with a single cell

# Collets user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        dims = {}
        for i in range(10):
            dims[i] = st.sidebar.slider(f"Latent Dimension {i}")
        features = pd.DataFrame(dims, index=[0])
        return features

    input_df = user_input_features()

### Display Input Data
st.subheader("User Input Features")

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write("From Sliders")
    st.write(input_df)

# TODO: Eventually, user should upload raw scRNAseq data. Add a Step here to run scVI to get latent gene expression.

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

model = get_trained_morphgan()
img = predict_image(model, input_df.values)

fig, ax = plt.subplots()
ax.imshow(img)
st.pyplot(fig)

# TODO: Add a Streamlit button for users to download their morphological image(s).