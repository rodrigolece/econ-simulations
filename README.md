# Analysis of games for economic development

This repository hosts a series of Jupyter notebooks to answer different research questions derived from the model presented in *Ernst, Ekkehard. ‘The Evolution of Time Horizons for Economic Development’, 2004.*  \[**Ernst04**\] [(DOI)](https://doi.org/10.13140/RG.2.2.34593.15204). 

Most of the analysis and questions answered in the notebooks use the package ```kala```. The instructions to install it can be found below. 

The notebooks and the present repository, although initially thought as a first step for exploration, can be further developed to accompany a future publication as part of the open code policies of the journals.

## Index 

Each of the notebooks has as title the question that wants to answer. 

- 1. How does the wealth of a group evolve with respect to the network structure?
- 2. What is the effect of memory and update rules when the network and the parameters are kept fixed?
- 3. How does the wealth of a group evolve with respect to different parameters?
- 4. How do different types of shocks with different magnitude affect the evolution of the game?

## Installation of ```kala```

The package is written in Python (minimal version: 3.10). We recommend that the installation is made inside a virtual environment and to do this, one can use either `conda` (recommended in order to control the Python version) or the Python builtin `venv` (if the system's version of Python is compatible).

### Create a virtual environment using Python's builtin `venv`

The first step is running

```bash
$ python -m venv kala
```

This creates a folder that contains the virtual environment `kala` (a different name can be used; change below as appropriate). We activate it using


```bash
$ source kala/bin/activate
```

### Using conda (recommended)

The tool `conda`, which comes bundled with Anaconda has the advantage that it lets us specify the version of Python that we want to use. Python>=3.10 is required.

A new environment can be created with

```bash
$ conda create -n kala python=3.10 -y
```

Like before, the environment's name can be anything else instead of `kala` (simply change the name below). We activate it using

```bash
$ conda activate kala
```

### Local install of the package

Once we are working inside an active virtual environment, we install (the dependencies and) the package by running

```bash
[$ pip install -r requirements.txt]
$ pip install -e .
```