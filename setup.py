#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='lit-server',
    version='0.0.1',
    description='Lit server',
    author='Lightning AI',
    author_email='luca@lightning.ai',
    url='https://github.com/Lightning-AI/lit-server',
    install_requires=['fastapi>=0.100'],
    packages=find_packages(),
)