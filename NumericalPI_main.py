from NumericalPathIntegral import NumericalPathIntegral

N = 8  # Number of points in the path
a = 0.5  # Lattice spacing
m = 1  # Mass of the particle
potential = 'harmonic'  # Potential type

 # Create an instance of the PathIntegralQuantum class
NPI = NumericalPathIntegral(N, a, m, potential)
 
 # Perform the analysis
analysis_data = NPI.compute_analysis()
 
 # Plot the results
NPI.plot_results(analysis_data)