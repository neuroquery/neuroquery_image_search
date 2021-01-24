#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="neuroquery_image_search",
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
