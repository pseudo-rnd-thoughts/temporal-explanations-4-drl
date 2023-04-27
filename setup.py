"""Setups the project."""

from setuptools import setup

setup(
    name="TemporalExplanation4XRL",
    version="1.0.0",
    author="Mark Towers",
    author_email="mt5g17@soton.ac.uk",
    description="Implementation of 'Temporal Explanations for Explainable Reinforcement Learning'",
    license="MIT",
    keywords=["Reinforcement Learning", "Explanability AI"],
    python_requires=">=3.7",
    packages=["temporal_explanations_4_xrl"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
