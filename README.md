[revert]: http://revertproject.org 

<img alt="Reversible dementia" height="150px"
    src="img/revert-logo.png"> 

# REVERT

This repository contains code for analysing flux and pressure recordings of
the [revert][revert] project.

## Install

```
git clone https://github.com/opeltre/revert
pip install revert
```

## Running tests

Most of the functions defined in [revert/transforms](revert/transforms) are tested for now. 
They include segmentation algorithms, spectral and spatial filters, and other differential calculus tools, implemented as sparse torch matrices. 

```
cd test && python -m unittest
```

## Using notebooks

For jupyter to look for locally installed packages you might need to 
build a kernel:

```
pip install ipykernel
python -m ipykernel install --user --name revert
```
Otherwise you can also add paths to the repository with `sys.path.insert`. 

## Scripts 

Some important pieces of code are intended to be run just once, e.g to transform the dataset or to extract segmented pulses from the recordings or to extract timestamps and regression results from XML files contained in the HDF5s to more convenients formats such as JSON or CSV. 

For instance, the `infusion.Dataset` class will look for timestamps in a file called `periods-{dbname}.json` and one might want to run first:
```
cd scripts-infusion
python extract_timestamps.py full $INFUSION_DATASETS/periods-full.json
```

# Data Loaders 

## Infusion tests

The `Dataset` constructor accepts relative paths w.r.t. the `$INFUSION_DATASETS` environment variable. 
Either define this path in your shell configuration or supply absolute paths to the directory containing `.hdf5` files 
compressed by ICM+. 

```py 
>>> from revert import infusion
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
