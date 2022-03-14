[revert]: http://revertproject.org 

<img alt="Reversible dementia" height="150px"
    src="img/revert-logo.png"> 

# REVERT

This repository contains code for analysing flux and pressure recording of
the [revert][revert] project.

## Install

```
git clone https://github.com/opeltre/revert
pip install revert
```

## Running tests

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

## Flux 

Blood and CSF fluxes over a typical cardiac cycle (CC) 
are acquired during a PCMRI examination, in which fluid velocities 
are integrated accross chosen sections of interest, 
such as large cerebral veins and arteries (blood),
the inter-ventricular aqueduct and cervical vertebras (CSF). 

<img alt="intracranial volume change" height="200px"
    src="img/volumeChange.png">

## ICP 

ICP is measured over time by lumbar puncture during infusion tests. 
The acquisition lasts ~45 minutes so that the resulting signal contains 
thousands of cardiac cycles, first recorded at baseline ICP and during
a constant-rate injection of saline water in CSF space. 

<img alt="intracranial pressure segmentation" height="200px"
    src="img/icpSegmentation.png">
    
## Images 

Vascular model [2], accounting for blood action on CSF pressure:  
<img alt='csf vascular model' width="200px" src='img/vascularModel.svg'>


Anatomical model, accounting for circulation inside CSF space:  
<img alt='csf anatomical model' width="200px" src='img/anatomicalModel.jpg'>


## References 

[1] Marmarou A. 
_A theoretical model and experimental evaluation of  the 
cerebrospinal  fluid  system._
Thesis, Drexel University, Philadelphia, PA, 1973

[2] Czosnyka M, Piechnik S, Richards HK, Kirkpatrick P, 
Smielewski P, Pickard JD. 
_Contribution of mathematical modelling to the bedside tests of
 cerebrovascular autoregulation._ 
Journal of Neurology, Neurosurgery, and Psychiatry 1997; 63:721-731 
