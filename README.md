**Overview**
This repository provides an implementation of the analysis described in [LePage’s lectures on Lattice QCD](https://arxiv.org/abs/hep-lat/0506036). The project explores the numerical formulation Quantum Chromodynamics (QCD) on a discretized space-time lattice, focusing on Monte Carlo methods, path integrals, and gauge field computations. 


**Files Description**
We present a brief description of the classes implemented to obtain all the results presented in LePage's work. To run the simulations use the associated main. Please refer to the report for a deeper discussion of each class.

- "NumericalPathIntegral.py": class provides a numerical framework for evaluating quantum mechanical path integrals using Monte Carlo methods. It supports different potential models (harmonic and quartic) and implements the VEGAS algorithm for efficient integration. The class enables computation of propagators, ground state wavefunction, ground state energy (these last two only for the harmonic case), and visualization of results.
