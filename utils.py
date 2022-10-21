from _utils import load_scVI_model, df2anndata
import torch
import numpy as np

def predict_image(G, label):
    device = "cpu"
    label = torch.from_numpy(label).to(device)
    z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)

    img = G(z, label, force_fp32=True)
    img = ((img.permute(0, 2, 3, 1) * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
    img = img[0].cpu().numpy()

    return img

def get_latent_expression(df):
    model, _ = load_scVI_model()
    adata = df2anndata(df)
    latent = model.get_latent_representation(adata)

    return latent
