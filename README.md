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
- Add in the u2os expression data
- Load in DepMap 22Q4 data
- Load in DepMap shRNA data
- Add code for gene-level BM stats and plots comparing that to chromosome arm relative position
- Create whole-genome plots for the DepMap data
- Demonstrate spitting DepMap cell lines by gene WT/LOF/GOF and assessing proximity bias
- Link to Recursion's benchmarking repo and show benchmarking results for cpg0016 before and after proximity bias correction
- Add in code to analyze RNA sequencing data

