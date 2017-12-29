#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

__author__ = 'James W. Browne, j@jwwwb.com'

setup(
    name="layercake",
    version="pre-0.1",
    description="automatically differentiating numpy based machine learning library",
    author='James W. Browne',
    author_email='j@jwwwb.com',
    license="BSD",
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
    install_requires=['numpy'],  # external packages as dependencies
)

