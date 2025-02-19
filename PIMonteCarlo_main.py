from PathIntegralMonteCarlo import PathIntegralMonteCarlo
import numpy as np
import matplotlib.pyplot as plt
    
# Initialize 3 instances with different methods
methods = ['n', 'y', 'noghost']
labels = {
         'n': 'not improved',
        'y': 'improved',
        'noghost': 'no ghost'
        }
results = {}
    
    # Run simulations for each method
for method in methods:
    sim = PathIntegralMonteCarlo(potential = 'harmonic', imp=method)
    sim.run()
    avgE, sdevE = sim.bootstrap_deltaE(sim.G, nbstrap=10000)
    results[method] = (avgE, sdevE)
        
    # Plot the comparison
plt.figure(figsize=(10, 6))
        
for method in methods:
    avgE, sdevE = results[method]
    t = np.arange(len(avgE)) * sim.a
    plt.errorbar(t, avgE, yerr=sdevE, label=labels[method], fmt='o')

    # Add details to the plot
plt.title('Comparison between discretization methods')
plt.xlabel('Time $t$')
plt.ylabel('Excited energy $\Delta E$')
plt.legend()
plt.grid(True)
plt.show()