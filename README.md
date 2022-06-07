[revert]: http://revertproject.org 

<img alt="Reversible dementia" height="150px"
    src="img/revert-logo.png"> 

# REVERT

This repository contains code for analysing flux and pressure recordings of
the [revert][revert] project.

## Install

Install in editable mode via pip for changes to take effect without needing to reinstall:
```
git clone https://github.com/opeltre/revert
cd revert && pip install -e ./
```
Omit the `-e` option if you don't plan to change the code. 

## Environment variables

The library will look for a few environment variables that should be set first. 

__Dataset locations:__
Path to dataset directories should be given via the `$INFUSION_DATASETS` and `$PCMRI_DATASETS` environment variables, 
which should point to directories containing datasets as subdirectories (use symlinks to clone or filter the full dataset). 

__Model states and logs:__
Checkpoints and tensorboard traces will be stored in the `$REVERT_MODELS` and `$REVERT_LOGS` directories. 

__Example:__ 
Consider appending these lines to your shell configuration file, before moving/symlinking datasets to the appropriate locations:


```sh
# ~/.bashrc 
export INFUSION_DATASETS='$HOME/infusion_datasets'
export PCMRI_DATASETS='$HOME/pcmri_datasets'
export REVERT_MODELS='$HOME/revert/_models'     # paths starting with _* 
export REVERT_LOGS='$HOME/revert/_logs'         # are gitignored by default

# terminal
mkdir ~/infusion_datasets 
cp -rs /abs/path/to/dset $INFUSION_DATASETS/full
```

## Dataset scripts 

Some important pieces of code in [scripts-infusion](scripts-infusion) and [scripts-pcmri](scripts-pcmri) 
are intended to be run just once, e.g to transform the dataset or to extract segmented pulses from the recordings or to extract timestamps and regression results from XML files contained in the HDF5s to more convenients formats such as JSON or CSV.

__Infusion:__
See the [parse-infusion.sh](parse-infusion.sh) master script, which should run the following steps:
1. Extract timestamps from hdf5 files (3 sources), output `$INFUSION_DATASETS/periods-{dbname}.json`
2. Extract analysis results (elastance, etc.) from hdf5 files (results.xml), output `$INFUSION_DATASETS/results-{dbname}.json`
3. Extract pulses from specified intervals ('Baseline', 'Infusion' and 'Plateau' by default), output `$INFUSION_DATASETS/{tag}-{dbname}.pt`. 

__PCMRI:__
The raw text files are parsed to produce either a 6-channel tensor aggregating flows by type, or a json file with key-value pairs keeping all channels untouched. 

## Running tests

Having a look at the tests defined in [revert/test](revert/test) is a good way to understand the library components and interface. Tests can be run by:

```
cd test && python -m unittest
```

Most of the functions defined in [revert/transforms](revert/transforms) are tested. 
They include segmentation algorithms, spectral and spatial filters, and other differential calculus tools, implemented as sparse torch matrices. 

Model architectures from [revert/models](revert/models) are also tested for a good part, with optional `.fit(dset, ...)` attempts skipped by default, 
using the `@test.skipFit(name)` decorator. If the `$REVERT_TEST_FIT` environment variable is set to `"true"`, then the test module will look in `test/config.toml` for which tests to run. 

## Using notebooks (outdated)

__N.B.__ Notebooks have been deleted to avoid polluting the source repository. We considered creating a `revert-notebooks` repository instead if notebooks become useful again, e.g. as examples or experiment notebooks. 

For jupyter to look for locally installed packages you _might_ need to 
build a kernel:

```
pip install ipykernel
python -m ipykernel install --user --name revert
```
Otherwise you can also add paths to the repository with `sys.path.insert`. 

# Data Loaders 

## Infusion tests

The `Dataset` constructor accepts relative paths w.r.t. the `$INFUSION_DATASETS` environment variable. 
Either define this path in your shell configuration or supply absolute paths to the directory containing `.hdf5` files 
compressed by ICM+. 

```py 
>>> from revert im17e3e0d88b7c458d82e314968b752b04ad084897port infusion
>>> db = infusion.Dataset("full")  # Dataset instance
>>> file = db.get(0)               # File instance <=> db.get(db.ls()[0])
>>> icp_full = file.icp()          # Full ICP signal
>>> icp = file.icp(0, 1000)        # First 10 seconds at 100 Hz
```
See help for the `infusion.Dataset` and `infusion.File` for more information, or have a look at the source in the [revert/infusion](revert/infusion) directory.

## PCMRI exams

The `Dataset` constructor accepts relative paths w.r.t. the `$PCMRI_DATASETS` environment variable. 

```py
>>> from revert import pcmri
>>> db = pcmri.Dataset("full")     # Dataset instance
>>> file = db.get(0)               # File instance
```
See help for the `pcmri.Dataset` and `pcmri.File` for more information, or have a look at the source in the [revert/pcmri](revert/pcmri) directory.

# Blood and Cerebrospinal Fluid Dynamics

Intracranial pressure (ICP) shows a pulsatile dynamic due
to cardiac excitations, inducing periodic brain expansions 
inside an inextensible cranial bone. Cerebrospinal fluid (CSF), 
bathing the brain and spine, is periodically flushed in and out
to the softer spinal sac through the occipital 
foramen and cervical vertebras 
to accomodate for blood volume changes. 

Normal pressure hydrocephalus (NPH) is a CSF formation and absorption 
disorder that causes gait and other cognitive impairments in the 
elderly population. Believed to be largely misdiagnosed as Alzheimer, 
NPH is a _reversible dementia_ as symptoms can sometimes
quickly disappear by shunt drainage. Improving practice in NPH diagnosis 
and gaining finer characterisations of CSF disorders is the goal of the 
revert project. 

<img alt="brain PCMRI and infusion exams" height="300px"
    src="img/infusionPCMRI.png"> 
