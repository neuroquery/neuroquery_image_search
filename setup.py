#!/usr/bin/env python

from pathlib import Path

from setuptools import setup, find_packages

version = (
    Path(__file__)
    .parent.joinpath("src", "neuroquery_image_search", "data", "VERSION.txt")
    .read_text()
    .strip()
)
description = (
    "Search the NeuroQuery dataset for studies with "
    "activation patterns similar to an input image."
)
setup(
    name="neuroquery_image_search",
    description=description,
    version=version,
    url="https://github.com/neuroquery/neuroquery_image_search",
    maintainer="Jerome Dockes",
    maintainer_emain="jerome@dockes.org",
    license="BSD 3-Clause License",
    classifiers=["Programming Language :: Python :: 3"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"neuroquery_image_search.data": ["*"]},
    install_requires=[
        "nilearn",
        "numpy",
        "scipy",
        "pandas",
        "requests",
        "matplotlib",
        "jinja2",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "neuroquery_image_search = "
            "neuroquery_image_search._searching:image_search"
        ]
    },
)
