from neuroquery_image_search import _datasets


def test_fetch_data(request_mocker):

    data = _datasets.fetch_data()
    assert data.keys() == {
        "masker",
        "atlas_maps",
        "atlas_inv_covar",
        "studies_loadings",
        "terms_loadings",
        "studies_info",
        "document_frequencies"
    }
    data = _datasets.fetch_data()
    assert request_mocker.url_count == 1
