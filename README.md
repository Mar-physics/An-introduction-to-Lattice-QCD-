**Overview**

This repository provides an implementation of the analysis described in [LePage’s lectures on Lattice QCD](https://arxiv.org/abs/hep-lat/0506036). The project explores the numerical formulation Quantum Chromodynamics (QCD) on a discretized space-time lattice, focusing on Monte Carlo methods, path integrals, and gauge field computations. 


**Files Description**

We present a brief description of the classes implemented to obtain all the results presented in LePage's work. To run the simulations use the associated main. Please refer to the report (An Introduction to Lattice QCD) for a deeper discussion of each class.

- "NumericalPathIntegral.py": This class is the starting point and provides a numerical framework for evaluating quantum mechanical path integrals. It supports different potential models (harmonic and quartic) and implements the VEGAS algorithm for efficient integration. The class enables computation of propagators, ground state wavefunction, ground state energy (these last two only for the harmonic case), and visualization of results.
- "PathIntegralMonteCarlo.py": This class implements a Monte Carlo algorithm for numerically evaluating first-state excitation energy from the propagator. The code supports both harmonic and anharmonic potentials and employs a Metropolis-based algorithm for sampling path configurations. Various action discretizations, including unimproved, improved, and ghost-free formulations, are implemented to analyze their impact on both the results and statistical errors, which are handled using binning and bootstrap methods.
- "EOMSolver.py": This class implements a numerical solver for the equation of motion (EOM) for the harmonic oscillator in a discretized one-dimensional system. It supports different discretizations of the action (the same ones present in the previously described class). The solver constructs the corresponding coefficient matrices, applies Dirichlet boundary conditions, and numerically extracts the squared frequency of the fundamental mode. Additionally, the ghost mode frequency is computed for the improved action and fitted against theoretical expectations. The implementation is optimized with Numba for efficient matrix construction and solution.
- 
