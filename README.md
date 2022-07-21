[revert]: http://revertproject.org 

<img alt="Reversible dementia" height="150px"
    src="img/revert-logo.png"> 

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

# REVERT

This repository contains code for analysing flux and pressure recordings of
the [revert][revert] project.

## Install

Using revert requires python 3.9 or later.
Install in editable mode via pip for changes to take effect without needing to reinstall:
```
git clone https://github.com/opeltre/revert
cd revert && pip install -e ./
```
Omit the `-e` option if you don't plan to change the code. 

## Running tests

Most of the functions defined in [revert/transforms](revert/transforms) are tested for now. 
They include segmentation algorithms, spectral and spatial filters, and other differential calculus tools, implemented as sparse torch matrices. 

```
cd test && python -m unittest
```

## Using notebooks

For jupyter to look for locally installed packages you _might_ need to 
build a kernel:

```
pip install ipykernel
python -m ipykernel install --user --name revert
```
Otherwise you can also add paths to the repository with `sys.path.insert`. 

# Scripts 

## Infusion exams

With revert, you can process raw data to make them more conveniant fo work. For example, infusion recordings look a bit like cardiac recordings with regular pulses as CSF pressure increase when blood is pushed into the patient's head. You may want to segment all infusion pulses from all patients to run deep learning algorithms on this dataset. To do so, once you have downloaded infusion data, you might have the hdf5 files stored in a directory with a path like this `/.../infusion_datasets/full`. Export a new environment variable :
```
export INFUSION_DATASETS="/.../infusion_datasets"
```
The `INFUSION_DATASETS` variable must target the parent directory of the directory containing hdf5 files.
Some important pieces of code are intended to be run just once, e.g to transform the dataset or to extract segmented pulses from the recordings or to extract timestamps and regression results from XML files contained in the HDF5s to more convenients formats such as JSON or CSV. 

To segment pulses, you can use `extract_pulses.py` scripts in `scripts_infusion` directory but it uses `infusion.Dataset` class. This latter will look for timestamps in a file called `periods-{dbname}.json` and you might want to run first:
```
cd scripts-infusion
python extract_timestamps.py full
```
to generate this file. If you have your own dataset with infusion recordings of your patients in hdf5 files stored in a directory `/.../infusion_datasets/mypatients`, just replace `full` by `mypatients` in the previous line and in futur occurences of `full`. The program will print at some point in the output a list of labels that looks like this :
```
--- Keys encountered in icmtests:
[
  "Plateau",
  "Overdrainage test",
  "CSF Infusion",
  "Overdrainage baseline",
  "Baseline",
  "Infusion",
  "Transition"
]
```
We will use these labels for pulse segmentation. Once this extraction is done, you can explore the infusion dataset manually with python like that :
```py 
>>> from revert import infusion
>>> db = infusion.Dataset("full")  # Dataset instance
>>> file = db.get(0)               # File instance <=> db.get(db.ls()[0])
>>> icp_full = file.icp()          # Full ICP signal
>>> icp = file.icp(0, 1000)        # First 10 seconds at 100 Hz
```
(See help for the `infusion.Dataset` and `infusion.File` for more information, or have a look at the source in the [revert/infusion](revert/infusion) directory.)
And finally you can segment pulses, using this command :
```
python extract_pulses.py full Baseline
```
This will create a file `baseline-full.pt` containing pulses marked with `Baseline` label. If you want a file containing pulses marked with `Plateau` label, replace the argument `Baseline` by `Plateau`, you can give any label that appears in the list `Keys encountered in icmtests` that we encountered earlier during timestamps extraction. If you omit this argument, default is `Baseline`. You shall get the following in your terminal :
```
No events metadata found: /.../infusion_datasets/results-full.json
Run scripts-infusion/extract_results.py
model = Id
filtering files with 'Baseline' timestamps
extracting pulses from 2312 recordings
100%|██████████████████████████████████████████████████████████████████| 2312/2312 [05:33<00:00,  6.93it/s]
saving output as '/.../infusion_datasets/baseline-full.pt'
  + masks	: [1742, 64, 128] tensor
  + pulses	: [1742, 64, 128] tensor
  + means	: [1742, 64] tensor
  + slopes	: [1742, 64] tensor
  + keys	: 1742 list string
extracted 64 pulses from 1742 recordings
  - 7 bad Y-quantizations encountered
  - 547 low amplitudes encountered
  - 16 errors encountered

```
The output file `{label}-full.pt` is meant to be loaded in python with `torch` like this :
```
dataset = torch.load("/.../infusion_datasets/Baseline.pt")
```
After that, `dataset` is a dictionnary with the keys just listed above. `dataset['pulses']` is a tensor such that `dataset['pulses'][i,j]` is the j-th pulse of the i-th patient encoded as a recording of length 128 hundredths of a second. As different pulses in the original recording have different lengths, segmented pulses are padded with their final values to reach a length of 128. Distinction between real part and padding part is encoded in `dataset['masks']`. Pulses are slightly different from their original shapes to make the dataset more homegenous : each of them has been soustracted its mean value and mean slope and those values are stored in `dataset['means']` and `dataset['slopes']`. `dataset['keys']` identifies each patient by a character string.

## PCMRI exams

The `Dataset` constructor accepts relative paths w.r.t. the `$PCMRI_DATASETS` environment variable. 

```py
>>> from revert import pcmri
>>> db = pcmri.Dataset("full")     # Dataset instance
>>> file = db.get(0)               # File instance
```
See help for the `pcmri.Dataset` and `pcmri.File` for more information, or have a look at the source in the [revert/pcmri](revert/pcmri) directory.

# Deep Learning Algorithms

Revert includes a library that is a layer on top of torch. It enables the conceptions of many architectures with very simple descriptions in terms of `Pipe`s that chain together modules of your choice like neural networks, concatenating modules, splitting modules, etc.
