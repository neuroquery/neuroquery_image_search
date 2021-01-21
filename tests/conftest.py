from pathlib import Path
import tempfile

import pytest
import numpy as np
import pandas as pd
from scipy import sparse
import nibabel
from nilearn.datasets import _testing

from nilearn.datasets._testing import request_mocker   # noqa: F401


@pytest.fixture()
def fake_img():
    rng = np.random.default_rng(0)
    img = rng.random(size=(4, 3, 5))
    return nibabel.Nifti1Image(img, np.eye(4))


def make_fake_data():
    n_voxels, n_components, n_studies = 23, 8, 12
    rng = np.random.default_rng(0)
    difumo_maps = rng.random((n_components, n_voxels))
    difumo_maps[rng.binomial(1, 0.3, size=difumo_maps.shape).astype(int)] = 0
    difumo_inverse_covariance = np.linalg.pinv(difumo_maps.dot(difumo_maps.T))
    difumo_maps = sparse.csr_matrix(difumo_maps)
    projections = rng.random((n_studies, n_components))
    articles_info = pd.DataFrame({"pmid": np.arange(n_studies) + 100})
    articles_info["title"] = [
        f"title {pmid}" for pmid in articles_info["pmid"]
    ]
    articles_info["pubmed_url"] = [
        f"url {pmid}" for pmid in articles_info["pmid"]
    ]
    mask = np.zeros(4 * 3 * 5, dtype=int)
    mask[:n_voxels] = 1
    mask = mask.reshape((4, 3, 5))
    mask_img = nibabel.Nifti1Image(mask, np.eye(4))
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        sparse.save_npz(temp_dir / "difumo_maps.npz", difumo_maps)
        np.save(
            temp_dir / "difumo_inverse_covariance.npy",
            difumo_inverse_covariance,
        )
        np.save(temp_dir / "projections.npy", projections)
        articles_info.to_csv(temp_dir / "articles-info.csv", index=False)
        mask_img.to_filename(str(temp_dir / "mask.nii.gz"))
        archive = _testing.dict_to_archive(
            {"neuroquery_image_search_data": temp_dir}
        )
    return archive


@pytest.fixture(autouse=True)
def temp_data_dir(tmp_path_factory, monkeypatch):
    home_dir = tmp_path_factory.mktemp("temp_home")
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("USERPROFILE", str(home_dir))
    data_dir = home_dir / "neuroquery_data"
    data_dir.mkdir()
    monkeypatch.setenv("NEUROQUERY_DATA", str(data_dir))


@pytest.fixture(autouse=True, scope="function")
def osf_mocker(request_mocker):
    request_mocker.url_mapping[
        "https://osf.io/rvm78/download"
    ] = make_fake_data()
    return request_mocker
