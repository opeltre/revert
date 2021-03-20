[revert]: /

# Blood and Cerebrospinal Fluid Dynamics

This repository contains code for analysing flux and pressure measures 
acquired for the [revert][revert] project.
Intracranial pressure (ICP) shows a pulsatile dynamic due
to cardiac excitations, inducing periodic brain expansions 
inside an inextensible cranial bone. Cerebrospinal fluid (CSF), 
bathing the brain and spine, is periodically flushed in and out
to the softer spinal sac through the occipital 
foramen and cervical vertebras 
to accomodate for blood volume changes. 

## Flux 

Blood and CSF fluxes over a typical cardiac cycle (CC) 
are acquired during a PCMRI examination, in which fluid velocities 
are integrated accross chosen sections of interest, 
such as large cerebral veins and arteries (blood),
the inter-ventricular aqueduct and cervical vertebras (CSF). 


## Images 

Vascular model [2], accounting for blood action on CSF pressure:  
![csf vascular model](img/vascularModel.svg)


Anatomical model, accounting for circulation inside CSF space:  
![csf anatomical model](img/anatomicalModel.jpg)

Because CSF fluxes taken accross different route sections are 
desynchronised, it is important to account for pressure gradients 
putting the fluid in motion. 


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
