
NIOP-S: Spatio-Temporal Nature Investment Optimisation and Planning
==================================================================

[![DOI](https://zenodo.org/badge/1160444138.svg)](https://doi.org/10.5281/zenodo.18675992)

Code for modelling spatio-temporal restoration investment portfolios under climate risk
and nature market dynamics.

This repository implements the Spatio-Temporal Nature Investment Optimisation and
Planning (NIOP-S) framework developed for the spatio-temporal optimisation chapter of
the associated thesis/paper. The model extends temporal stochastic dynamic programming
(SDP) into a spatial portfolio framework, allowing restoration investments to vary across
regions, sites, and time while accounting for correlated climate disturbances and spatial
dependencies.

Key ideas
---------
- Restoration sites differ in cost, sequestration potential, and climate risk.
- Climate disturbances (e.g., cyclones) may be spatially correlated across regions.
- Investment decisions can be adapted year-by-year using a finite-horizon MDP solved by SDP.
- Outcomes are assessed using financial and environmental metrics (e.g., NPV, PI, area, carbon).

Typical workflow
----------------
1) Prepare or generate input data (master + case studies):
   - Plot_Parameters.xlsx
   - Plot_Dependency_Matrix.xlsx
   - Regional_Dependency_Matrix.xlsx

2) Generate Markov transition matrices:
   - Writes CSV transition matrices to: Data/Markov_Matrix/<Case_Study_X>/

3) Run SDP optimisation:
   - Writes policy + value outputs to: Data/SDP_Outputs/<Case_Study_X>/

4) Run forward simulations (optional):
   - Generates cyclone seed data with inter-region spillover
   - Applies optimal policies through time and records outcomes
   - Writes results to: Data/Simulation_Data/...

Repository structure (typical)
------------------------------
Scripts/
  - Spatio_Temporal_Model.py          Main pipeline script (run this)
  - Define_Parameters.py              Create master/case study input files
  - Spatio_Temporal_Markov.py         Markov matrix generation
  - SDP_Run_Spatio_temporal.py        SDP optimisation (finite-horizon MDP)
  - Spatio_Temporal_Simulations.py    Cyclone seed generation + simulations

Data/ (auto-created; usually gitignored)
  - Plots_Data/       inputs + case study folders
  - Markov_Matrix/    generated transition matrices
  - SDP_Outputs/      optimal policies + value functions
  - Simulation_Data/  cyclone seeds + simulation outputs

Requirements
------------
Python 3.9+ recommended

Core packages:
- numpy
- pandas
- pymdptoolbox (mdptoolbox)


Outputs
-------
This repository contains code only. Outputs (CSV, NPY, figures) are written to Data/
and should be excluded from version control via .gitignore.

Citation
--------
If you use the NIOP-S Framework please cite the archived software release:

Citation metadata is provided in [`CITATION.cff`](./CITATION.cff).

