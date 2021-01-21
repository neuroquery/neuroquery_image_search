from pathlib import Path

import numpy as np
from nilearn import plotting

from neuroquery_image_search.datasets import fetch_data


def _find_similar_studies(
    query_img,
    masker,
    atlas_maps,
    atlas_inv_covar,
    studies_loadings,
    studies_info,
    n_results,
):
    print(
        f"Searching in {studies_loadings.shape[0]:,} studies "
        "for similar activation patterns"
    )
    masked_query_img = masker.transform(query_img).ravel()
    # query = np.abs(atlas_maps.dot(masked_query_img))
    query = atlas_inv_covar.dot(atlas_maps.dot(masked_query_img))
    similarities = studies_loadings.dot(query)
    similarities /= similarities.max()
    most_similar = np.argsort(similarities)[::-1][:n_results]
    results = studies_info.iloc[most_similar].copy()
    results["similarity"] = similarities[most_similar]
    return results


def results_to_html_table(results):
    results["Title"] = [
        f"<a href={link} target='_blank'>{text}</a>"
        for link, text in results.loc[:, ["pubmed_url", "title"]].values
    ]
    results = results.loc[
        :, ["Title", "similarity", "pmid", "title", "pubmed_url"]
    ]
    results.rename(columns={"similarity": "Similarity"}, inplace=True)
    table = (
        results.style.bar(subset=["Similarity"], color="lightgreen", width=95)
        .hide_index()
        .hide_columns(["pmid", "title", "pubmed_url"])
        .format({"Similarity": "{:.2f}"})
        .render()
    )
    return table


def results_to_html(results, title, query_img):
    table = results_to_html_table(results)
    img_display = plotting.view_img(query_img, threshold="95%").get_iframe()
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>{title}</title>
    <meta charset="UTF-8" />
    <style>
    tr:nth-child(even) {{background: #EEE}}
    tr:nth-child(odd) {{background: #FFF}}
    </style>
    </head>
    <body>
    <h1>{title}</h1>
    {img_display}
    <h2>Most similar studies</h2>
    {table}
    </body>
    </html>
    """
    return plotting.html_document.HTMLDocument(html)


def find_similar_studies(query_img, n_results=50):
    data = fetch_data()
    results = _find_similar_studies(query_img, n_results=n_results, **data)
    return results


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "query_img",
        type=str,
        help="The Nifti image with which to query the dataset",
    )
    parser.add_argument(
        "-n",
        "--n_results",
        type=int,
        default=50,
        help="The number of similar studies returned",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="The file in which to store the output. If not specified, "
        "output is displayed in a web browser. Output format depends "
        "on the filename extension (.html, .csv or .tsv)",
    )
    return parser


def image_search(args=None):
    parser = _get_parser()
    args = parser.parse_args(args=args)
    img = args.query_img
    results = find_similar_studies(img, n_results=args.n_results)
    if args.output is None:
        results_to_html(results, Path(img).name, img).open_in_browser()
        print("Displaying results in web browser")
        print("Use '--output' to write results in a file")
        return
    output_file = Path(args.output)
    print(f"Saving results in {output_file}")
    if output_file.suffix in [".html", ".htm"]:
        results_to_html(results, Path(img).name, img).save_as_html(
            output_file
        )
        return
    if output_file.suffix == ".csv":
        results.to_csv(str(output_file), index=False)
        return
    results.to_csv(str(output_file), index=False, sep="\t")
