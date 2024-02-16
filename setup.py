#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='litserve',
    version='0.0.1',
    description='Lightweight AI server',
    author='Lightning AI',
    author_email='luca@lightning.ai',
    url='https://github.com/Lightning-AI/litserve',
    install_requires=['fastapi>=0.100'],
    packages=find_packages(),
)
