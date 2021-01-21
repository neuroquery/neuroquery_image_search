import numpy as np
import pandas as pd
from neuroquery_image_search import image_search


def test_image_search(tmp_path, fake_img):
    img_path = str(tmp_path / "img.nii.gz")
    fake_img.to_filename(img_path)
    results_path = str(tmp_path / "results.csv")
    image_search.image_search([img_path, "-o", results_path, "-n", "7"])
    results = pd.read_csv(results_path)
    assert results.shape == (7, 4)
    assert np.allclose(results.at[0, "similarity"], 1.0)
