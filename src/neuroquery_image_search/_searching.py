from pathlib import Path
from string import Template
import json

import numpy as np
from nilearn import plotting, datasets, image

from neuroquery_image_search._datasets import fetch_data


def studies_to_html_table(studies):
    """Transform DataFrame of similar studies to an HTML table.

    Parameters
    ----------
    studies : pandas DataFrame, as returned by
        `NeuroQueryImageSearch()(img)["studies"]`

    Returns
    -------
    table : a `str` containing an HTML table.

    """
    studies["Title"] = [
        f"<a href='{link}' target='_blank'>{text}</a>"
        for link, text in studies.loc[:, ["pubmed_url", "title"]].values
    ]
    studies = studies.loc[
        :, ["Title", "similarity", "pmid", "title", "pubmed_url"]
    ]
    studies.rename(columns={"similarity": "Similarity"}, inplace=True)
    table = (
        studies.style.bar(
            subset=["Similarity"], color="lightgreen", width=98, vmin=0.0
        )
        .hide_index()
        .hide_columns(["pmid", "title", "pubmed_url"])
        .format({"Similarity": "{:.2f}"})
        .set_table_attributes('class="studies-table"')
        .render()
    )
    return table


def terms_to_html_table(terms):
    """Transform DataFrame of similar terms to an HTML table.

    Parameters
    ----------
    terms : pandas DataFrame, as returned by
        `NeuroQueryImageSearch()(img)["terms"]`

    Returns
    -------
    table : a `str` containing an HTML table.

    """
    nq_url = (
        "<a href='https://neuroquery.org/query?text={}' target='_blank'>{}</a>"
    )
    terms["Term"] = [
        nq_url.format(t.replace(" ", "+"), t) for t in terms["term"]
    ]
    terms = terms.loc[:, ["Term", "similarity"]]
    terms.rename(columns={"similarity": "Similarity"}, inplace=True)
    table = (
        terms.style.bar(
            subset=["Similarity"], color="lightgreen", width=95, vmin=0.0
        )
        .hide_index()
        .format({"Similarity": "{:.2f}"})
        .set_table_attributes('class="terms-table"')
        .render()
    )
    return table


def results_to_html(results, title="NeuroQuery Image Search"):
    """Create an HTML page displaying results of NeuroQueryImageSearch

    Parameters
    ----------
    results : dict returned by `NeuroQueryImageSearch()(img)`

    title : str, title of the resulting page

    Returns
    -------
    html : nilearn.plotting.html_document.HTMLDocument
       An object representing an HTML page; methods of interest are
       `save_as_html` and `open_in_browser`.

    """
    studies_table = studies_to_html_table(results["studies"])
    terms_table = terms_to_html_table(results["terms"])
    img_display = plotting.view_img(
        results["image"], threshold="95%"
    ).get_iframe()
    template = (
        Path(__file__)
        .parent.joinpath("data", "search_results_template.html")
        .read_text()
    )
    html = Template(template).safe_substitute(
        {
            "title": title,
            "img_display": img_display,
            "studies_table": studies_table,
            "terms_table": terms_table,
        }
    )
    return plotting.html_document.HTMLDocument(html)


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)


class NeuroQueryImageSearch:
    """Search for studies and terms with activation maps similar to an image.

    Searches the NeuroQuery dataset (see https://neuroquery.org)

    How to use:
    >>> search = NeuroQueryImageSearch()
    >>> results_a = search("image_a.nii.gz")
    >>> results_b = search("image_b.nii.gz")

    Returns a dict containing "terms" and "studies" DataFrames as well as
    "image" (`nibabel.Nifti1Image` containing the input image).

    """
    def __init__(self):
        self.data = fetch_data()

    def __call__(
        self, query_img, n_studies=50, n_terms=20, transform="absolute_value"
    ):
        """Search for studies and terms with activation maps similar to an image

        Parameters
        ----------
        query_img : path to a .nii.gz image or `nibabel.Nifti1Image`; the input
            image

        n_studies : number of similar studies to return

        n_terms : number of similar terms to return

        transform : {"absolute_value", "positive_part" or "identity"}
            Transformation to apply to the input image. As NeuroQuery
            activation coordinates don't have a sign, often results are more
            helpful when comparing to the absolute value of the input image.
            "absolute_value" is the default.

        Returns
        -------
        results : dictionary with keys "image", "studies", "terms".
           - "image" is the input image

           - "studies" is a pandas DataFrame containing similar studies with
             columns: ['pmid', 'title', 'pubmed_url', 'similarity']

           - "terms" is a pandas DataFrame containing similar terms with
             columns: ['term', 'document_frequency', 'similarity'].

        """
        total_n_studies = self.data["studies_loadings"].shape[0]
        total_n_terms = self.data["terms_loadings"].shape[0]
        print(
            f"Searching in {total_n_studies:,} studies "
            f"and {total_n_terms:,} terms "
            "for similar activation patterns"
        )
        query_img = image.load_img(query_img)
        masked_query_img = self.data["masker"].transform(query_img).ravel()
        if transform == "absolute_value":
            masked_query_img = np.abs(masked_query_img)
        elif transform == "positive_part":
            masked_query_img = np.maximum(0, masked_query_img)
        query = self.data["atlas_inv_covar"].dot(
            self.data["atlas_maps"].dot(masked_query_img)
        )

        results = {}

        similarities = self.data["studies_loadings"].dot(query)
        most_similar = np.argsort(similarities)[::-1][:n_studies]
        if (similarities > 0).any():
            similarities /= similarities.max()
        study_results = (
            self.data["studies_info"]
            .iloc[most_similar]
            .reset_index(drop=True, inplace=False)
        )
        study_results["similarity"] = similarities[most_similar]
        results["studies"] = study_results

        similarities = self.data["terms_loadings"].dot(query)
        similarities *= np.log(
            1 + self.data["document_frequencies"]["document_frequency"].values
        )
        most_similar = np.argsort(similarities)[::-1][:n_terms]
        if (similarities > 0).any():
            similarities /= similarities.max()
        term_results = (
            self.data["document_frequencies"]
            .iloc[most_similar]
            .reset_index(drop=True, inplace=False)
        )
        term_results["similarity"] = similarities[most_similar]
        results["terms"] = term_results
        results["image"] = query_img

        return results


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "query_img",
        nargs="?",
        type=str,
        default=None,
        help="Nifti image with which to query the dataset. "
        "If not provided, an example image is downloaded from neurovault.org.",
    )
    parser.add_argument(
        "--n_studies",
        type=int,
        default=50,
        help="Number of similar studies returned",
    )
    parser.add_argument(
        "--n_terms",
        type=int,
        default=20,
        help="Number of similar terms returned",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="File in which to store the output. If not specified, "
        "output is displayed in a web browser. Output format depends "
        "on the filename extension (.html or .json)",
    )
    parser.add_argument(
        "--transform",
        type=str,
        choices=["absolute_value", "identity", "positive_part"],
        default="absolute_value",
        help="Transform to apply to the image. As NeuroQuery ignores the "
        "direction of activations by default the absolute value of the "
        "input map is compared to activation patterns in the literature.",
    )
    return parser


def image_search(args=None):
    parser = _get_parser()
    args = parser.parse_args(args=args)
    img = args.query_img
    if img is None:
        img = datasets.fetch_neurovault_motor_task()["images"][0]
    try:
        image_name = Path(img).name
    except Exception:
        image_name = "Image"
    search = NeuroQueryImageSearch()
    results = search(
        img,
        n_studies=args.n_studies,
        n_terms=args.n_terms,
        transform=args.transform,
    )
    if args.output is None:
        results_to_html(results, image_name).open_in_browser()
        print("Displaying results in web browser")
        print("Use '--output' to write results in a file")
        return
    output_file = Path(args.output)
    print(f"Saving results in {output_file}")
    if output_file.suffix in [".html", ".htm"]:
        results_to_html(results, image_name).save_as_html(output_file)
        return
    results.pop("image")
    output_file.write_text(json.dumps(results, cls=_JSONEncoder))
