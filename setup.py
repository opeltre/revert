#!/usr/bin/env python
import os
from setuptools import setup

with open('./requirements.txt', 'r') as f:
    requirements = f.read().strip().split('\n')

setup(
    name    ='revert',
    version ='0.1',
    description ="revert",
    author      ="Olivier Peltre",
    author_email='opeltre@gmail.com',
    url     ='https://github.com/opeltre/revert',
    license ='MIT',
    install_requires=requirements,
    packages = ['revert', 
                'revert.transforms']
)
