import numpy as np
import pandas as pd
from nilearn import image
import json

import pytest

from neuroquery_image_search import _searching, _datasets


def test_image_search(tmp_path, fake_img):
    img_path = str(tmp_path / "img.nii.gz")
    fake_img.to_filename(img_path)

    results_path = tmp_path / "results.json"
    _searching.image_search(
        f"{img_path} -o {results_path} --n_studies 7 --n_terms 3".split()
    )
    results = json.loads(results_path.read_text())
    study_results = pd.DataFrame(results["studies"])
    assert study_results.shape == (7, 4)
    assert np.allclose(study_results.reset_index().at[0, "similarity"], 1.0)

    results_path = tmp_path / "results.html"
    _searching.image_search(
        [img_path, "-o", str(results_path), "--n_studies", "1"]
    )
    results = results_path.read_text()
    assert results.strip().startswith("<!DOCTYPE html>")

    _searching.image_search(["-o", str(results_path), "--n_studies", "7"])
    results = results_path.read_text()
    assert "Image" in results
    _searching.image_search([])


def test_json_encoder():
    df = pd.DataFrame({"A": [2, 3]}, index=list("ab"))
    data = {"a": {"B": 3.3}, "b": df}
    as_json = json.dumps(data, cls=_searching._JSONEncoder)
    loaded = json.loads(as_json)
    loaded_df = pd.DataFrame(loaded["b"])
    assert (df == loaded_df).all().all()
    with pytest.raises(TypeError):
        json.dumps({"a": json}, cls=_searching._JSONEncoder)


def test_neuroquery_image_search(fake_img):
    search = _searching.NeuroQueryImageSearch()

    results = search(fake_img, 20, transform="identity", rescale_similarities=False)
    assert results["studies"]["similarity"].max() != pytest.approx(1.)

    results = search(fake_img, 20, transform="identity")
    assert results["terms"]["similarity"].min() == pytest.approx(0.)
    assert results["studies"]["similarity"].max() == pytest.approx(1.)
    results = results["studies"]
    neg_img = image.new_img_like(fake_img, image.get_data(fake_img) * -1.0)
    neg_results = search(neg_img, 20, transform="identity")["studies"]
    assert (neg_results["pmid"].values == results["pmid"].values[::-1]).all()

    results = search(fake_img, 20, transform="absolute_value")["studies"]
    neg_results = search(neg_img, 20, transform="absolute_value")["studies"]
    assert (neg_results["pmid"].values == results["pmid"]).all()
    pos_img = image.new_img_like(
        fake_img, np.maximum(0, image.get_data(fake_img))
    )
    results = search(fake_img, 20, transform="positive_part")["studies"]
    pos_results = search(pos_img, 20, transform="identity")["studies"]
    assert (pos_results["pmid"].values == results["pmid"]).all()
    data = _datasets.fetch_data()
    assert (search.data["studies_info"] == data["studies_info"]).all().all()
    assert (
        (search.data["document_frequencies"] == data["document_frequencies"])
        .all()
        .all()
    )
