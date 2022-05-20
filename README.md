# README #

Analysis code accompanying kappel et al. 2022 'Visual recognition of social signals by a tecto-thalamic neural circuit'.

This repository is intended to provide a snapshot of our analysis scripts 'as is'. The python notebooks and standalone scripts are somewhat commented but have not been optimized for modularity or recycling in other contexts.

To re-run the analysis and generate the data figure panels, follow these steps:



## cFos and behavior data

make sure to set paths first:

download zipped raw data 'behavior', 'HCR', and 'masks' from here: https://doi.org/10.17617/3.2QCFQP

unzip the rawData files into a root folder such as 'kappelData', like so:

- kappelData
   - Raw Data
       - behavior
       - HCR
       - masks

and adjust the 3 paths to the data in the props.csv file in the 'notebooks' folder from the repository.

then run the master notebook: kappelEtAl2022AllFigurePanels_cfosAndBehavior.ipynb




## 2photon calcium imaging: