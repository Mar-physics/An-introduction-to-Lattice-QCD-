import numpy as np
import matplotlib.pyplot as plt
from Exercise1 import NumericalPathIntegral
from Exercise2 import PathIntegralMonteCarlo

class StatisticalErrorAnalysis:
    def __init__(self, monte_carlo_instance):
        """
        Initialize the statistical error analysis class.
        
        Args:
            monte_carlo_instance (PathIntegralMonteCarlo): Instance of the Monte Carlo class.
        """
        self.mc = monte_carlo_instance

    def analyze_statistical_errors(self, num_bootstrap=100, binsize=10):
        """
        Perform analysis of statistical errors using bootstrap and binning methods.
        
        Args:
            num_bootstrap (int): Number of bootstrap samples.
            binsize (int): Size of bins for binning the data.
        """
        # Step 1: Generate Monte Carlo averages
        self.mc.MCaverage(self.mc.x, self.mc.G)

        # Step 2: Compute standard deviations
        std_deviation = self.mc.sdev(self.mc.G)

        # Step 3: Apply binning
        binned_G = self.mc.bin(self.mc.G, binsize)
        std_dev_binned = self.mc.sdev(binned_G)

        # Step 4: Apply bootstrap method for energy differences
        avgE, sdevE = self.mc.bootstrap_deltaE(binned_G, nbstrap=num_bootstrap)

        # Step 5: Visualize results
        t = np.arange(self.mc.N - 1) * self.mc.a
        plt.figure(figsize=(10, 6))
        plt.title("Statistical Error Analysis")
      # plt.errorbar(t, avgE, yerr=sdevE, fmt='o', label='Bootstrap Estimate') # Plot with error bars doesn't make much sense
        plt.plot(t, avgE, 'o', label='Bootstrap Estimate')  # Plot without error bars
        plt.plot(t, np.ones(len(t)), label='Exact Energy')
        plt.xlabel('Time (t)')
        plt.ylabel('Energy Difference (ΔE)')
        plt.xlim(-0.1, 3.2)  # Set x-axis domain to match Exercise2
        plt.ylim(0, 2)  # Match typical energy range as in Exercise2
        plt.legend()
        plt.grid()
        plt.show()
        
        # Prepare data for the table
        binned_means = binned_G.mean(axis=0)[:self.mc.N - 1]
        table = np.column_stack((
            t,                  # Time
            avgE,               # Bootstrap average energy
            sdevE,              # Bootstrap errors
            std_deviation[:self.mc.N - 1],   # Raw standard deviation
            std_dev_binned[:self.mc.N - 1],  # Binned standard deviation
            binned_means        # Mean of binned G
            ))

        # Only select the first 7 rows for printing
        table_to_print = table[:7]  # Slicing to get the first 7 rows

        # Print table
        print("\nTable of Results (Time, Avg Energy, Error, Std Dev (Raw), Std Dev (Binned), Mean Binned G):")
        print(f"{'Time':<10}{'Avg Energy':<15}{'Error':<15}{'Std Dev (Raw)':<18}{'Std Dev (Binned)':<20}{'Mean Binned G':<15}")
        for row in table_to_print:
            print(f"{row[0]:<10.3f}{row[1]:<15.6f}{row[2]:<15.6f}{row[3]:<18.6f}{row[4]:<20.6f}{row[5]:<15.6f}")

    ''' # Output key results for debugging
        print("Standard Deviation (Raw):", std_deviation)
        print("Standard Deviation (Binned):", std_dev_binned)
        print("Bootstrap Averages:", avgE)
        print("Bootstrap Errors:", sdevE) '''

if __name__ == '__main__':
    # Parameters for the simulation
    numerical_integral = NumericalPathIntegral(N=20, a=0.5, m=1, potential='harmonic')
    monte_carlo = PathIntegralMonteCarlo(numerical_integral)

    # Perform statistical error analysis
    analysis = StatisticalErrorAnalysis(monte_carlo)
    analysis.analyze_statistical_errors(num_bootstrap=1000, binsize=20)
