# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 11:37:39 2016

@author: Amina Asif
"""
import setuptools
with open("README.md", "r",encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cafemap",
    version="0.2",
    author="Dr. Fayyaz Minhas, Amina Asif, Muhammad Arif",    
    description="Python Implementation of CAFÉ-Map: Context Aware Feature Mapping for mining high dimensional biomedical data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/foxtrotmike/cafemap",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
