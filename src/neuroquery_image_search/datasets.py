from pathlib import Path
import tarfile

import requests
import numpy as np
from scipy import sparse
import pandas as pd
from nilearn import input_data
from neuroquery.datasets import get_neuroquery_data_dir


def _download_data(data_dir="/tmp/nqimg"):
    print("Downloading Neuroquery image search data ...")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    url = "https://osf.io/rvm78/download"
    data = requests.get(url).content
    archive = data_dir / "neuroquery_image_search_data.tar.gz"
    archive.write_bytes(data)
    with tarfile.open(archive, "r:gz") as f:
        f.extractall(data_dir)
    print("Done")


def fetch_data():
    data_dir = Path(get_neuroquery_data_dir()).joinpath(
        "extra", "neuroquery_image_search_data"
    )
    if not data_dir.is_dir():
        _download_data(data_dir.parent)
    result = {}
    result["masker"] = input_data.NiftiMasker(
        str(data_dir / "mask.nii.gz")
    ).fit()
    result["atlas_maps"] = sparse.load_npz(str(data_dir / "difumo_maps.npz"))
    result["atlas_inv_covar"] = np.load(
        str(data_dir / "difumo_inverse_covariance.npy")
    )
    result["studies_loadings"] = np.load(str(data_dir / "projections.npy"))
    result["studies_info"] = pd.read_csv(str(data_dir / "articles-info.csv"))
    return result
