#!/usr/bin/env/python

from setuptools import setup, find_packages

setup(
    name='esh_dynamics',
    author='Greg Ver Steeg',
    author_email='gversteeg@gmail.com',
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
    url='https://github.com/gregversteeg/esh_dynamics',
    description='Hamiltonian Dynamics with Non-Newtonian Momentum for Rapid Sampling',
    keywords=['sampling, energy-based models, Hamiltonian dynamics'],
    license='MIT',
    packages=find_packages(),
    version="0.1"
)
