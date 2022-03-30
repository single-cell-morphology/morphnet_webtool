# MorphGAN Web Tool

This is a [Streamlit](https://streamlit.io/) app where users can upload scRNAseq data to obtain predicted _nuclear_ or _neuronal_ morphological images.

### How to Run
1. Git clone this repo, i.e. `git clone https://github.com/welch-lab/morphgan_webtool.git`
2. Make sure you have the required packages (see below). It is highly recommended to use a conda environment (tested with Python 3.8).
3. In the `morphgan_webtool` directory, run the following command:
```
streamlit run app.py
```

### Requirements
- anndata
- scvi-tools
- umap-learn
- torch
- streamlit
- pandas
- numpy
- matplotlib
