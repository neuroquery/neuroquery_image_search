#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="neuroquery_image_search",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={},
    install_requires=[
        "neuroquery",
        "nilearn",
        "numpy",
        "scipy",
        "pandas",
        "requests",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "neuroquery_image_search = "
            "neuroquery_image_search.image_search:image_search"
        ]
    },
)
