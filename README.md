# proxbias

Code to support the publication "High-resolution genome-wide mapping of chromosome-arm-scale truncations induced by 
CRISPR-Cas9 editing". The preprint is on bioRxiv [here](https://www.biorxiv.org/content/10.1101/2023.04.15.537038v1.article-metrics)

# Installation

This package is installable via [Nexus](nexus.rxrx.io). 

```bash
pip install proxbias
```

# Notebooks

Notebooks to reproduce plots from public datasets are included in the `notebooks` directory. Note that this doesn't 
include rxrx3 for IP reasons. 

- `cpg0016_loading.ipynb` - Load the JUMP CP data, apply PCA and proximity bias correction and save data.
- `cpg0016_plots.ipynb` - Create whole-genome plots from cpg0016, calculate Brunner-Munzel statistics and make bar plots.
- `shinyDepMap_benchmark.ipynb` - Load DepMap 19Q3 data and create plots showing an enrichment for within-chromosome arm relationships.
- `DepMap_PB_Drivers.ipynb` - Use DepMap 22Q4 data to look at differential proximity bias when TP53 and other genes are wild-type vs loss/gain of function.

# TODO:
- Create whole-genome plots for the DepMap data
- Code to search DepMap for drivers of proximity bias
