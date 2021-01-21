import numpy as np
import pandas as pd
from nilearn import image

from neuroquery_image_search import image_search


def test_image_search(tmp_path, fake_img):
    img_path = str(tmp_path / "img.nii.gz")
    fake_img.to_filename(img_path)

    results_path = str(tmp_path / "results.csv")
    image_search.image_search([img_path, "-o", results_path, "-n", "7"])
    results = pd.read_csv(results_path)
    assert results.shape == (7, 4)
    assert np.allclose(results.at[0, "similarity"], 1.0)

    results_path = str(tmp_path / "results.tsv")
    image_search.image_search([img_path, "-o", results_path, "-n", "7"])
    results = pd.read_csv(results_path, sep="\t")
    assert results.shape == (7, 4)

    results_path = tmp_path / "results.html"
    image_search.image_search([img_path, "-o", str(results_path), "-n", "1"])
    results = results_path.read_text()
    assert results.strip().startswith("<!DOCTYPE html>")

    image_search.image_search(["-o", str(results_path), "-n", "7"])
    results = results_path.read_text()
    assert "Image" in results
    image_search.image_search([])


def test_find_similar_studies(fake_img):
    results = image_search.find_similar_studies(
        fake_img, 20, transform="identity"
    )
    neg_img = image.new_img_like(fake_img, image.get_data(fake_img) * -1.0)
    neg_results = image_search.find_similar_studies(
        neg_img, 20, transform="identity"
    )
    assert (neg_results["pmid"].values == results["pmid"].values[::-1]).all()

    results = image_search.find_similar_studies(
        fake_img, 20, transform="absolute_value"
    )
    neg_results = image_search.find_similar_studies(
        neg_img, 20, transform="absolute_value"
    )
    assert (neg_results["pmid"].values == results["pmid"]).all()
    pos_img = image.new_img_like(
        fake_img, np.maximum(0, image.get_data(fake_img))
    )
    results = image_search.find_similar_studies(
        fake_img, 20, transform="positive_part"
    )
    pos_results = image_search.find_similar_studies(
        pos_img, 20, transform="identity"
    )
    assert (pos_results["pmid"].values == results["pmid"]).all()
