import scvi
import anndata
import umap
import numpy as np
from pathlib import Path

def df2anndata(df):
    adata = anndata.AnnData(df)
    return adata

def load_scVI_model():
    path = Path("./models/gene_expression/patchseq_scVI")
    model = scvi.model.SCVI.load(path)
    adata = anndata.read_h5ad(path / "adata.h5ad")

    return model, adata

def get_patchseq_latent(model, adata):
    latent = model.get_latent_representation(adata)
    return latent

def umap_patchseq(latent):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(latent)
    return embedding

if __name__=="__main__":
    model, adata = load_scVI_model()
    latent = get_patchseq_latent(model, adata)
    np.save("./data/patchseq_latents.npy", latent)
