import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import vegas

class NumericalPathIntegral:
    def __init__(self, N, a, m, potential='harmonic'):
        self.N = N  # Number of points in the path
        self.a = a  # Lattice spacing
        self.m = m  # Mass of the particle
        self.potential = potential  # Potential type ('harmonic' or 'quartic')

        # Define the potential function
        if potential == 'harmonic':
            self.V = lambda x: 0.5 * x ** 2  # Harmonic potential
        elif potential == 'quartic':
            self.V = lambda x: 0.5 * x ** 4  # Quartic potential
        else:
            raise ValueError("Potential not recognized. Available: 'harmonic', 'quartic'")
    
    def integrand(self, x, x0=None, fixed=True):
        """
        Computes the integrand for the path integral.

        Parameters:
        - x: Array of positions
        - x0: Fixed starting/ending point for the path
        - fixed: Boolean indicating if the path starts and ends at x0

        Returns:
        - The integrand value for the given path
        """
        S_lat = 0  # Initialize the lattice action
        if fixed:
            # Add contributions from fixed boundary conditions
            S_lat = (self.m / (2 * self.a)) * (x[0] - x0) ** 2 + self.a * self.V(x0)
            S_lat += (self.m / (2 * self.a)) * (x0 - x[-1]) ** 2 + self.a * self.V(x[-1])
        else:
            # Add contributions from periodic boundary conditions
            S_lat = (self.m / (2 * self.a)) * (x[0] - x[-1]) ** 2 + self.a * self.V(x[-1])
            
        # Add contributions from the bulk of the path
        S_lat += (self.m / (2 * self.a)) * np.sum((x[1:] - x[:-1]) ** 2) + self.a * np.sum(self.V(x[:-1]))
        A = (self.m / (2 * np.pi * self.a)) ** (self.N / 2)  # Normalization factor

        return A * np.exp(-1.0 * S_lat)  # Exponentiate the action
    
    def vegas_integrate(self, x0=None, fixed=True, x_min=None, x_max=None):
        """
        Performs the numerical integration using the VEGAS algorithm.

        Parameters:
        - x0: Fixed boundary condition value
        - fixed: Boolean indicating fixed or periodic boundary conditions
        - x_min, x_max: Integration limits

        Returns:
        - The mean of the numerical integration
        """
        if x_min is None and x_max is None:
            x_min, x_max = -5, 5  # Default integration limits
        
        if x_min > x_max:
            raise ValueError("x_min cannot be greater than x_max.")
            
        if fixed and x0 is None:
            raise ValueError("x0 must be provided when fixed=True")

        # Define the integration limits and integrand
        if fixed:
            lims = [[x_min, x_max]] * (self.N - 1)  # Integrate over N-1 points for fixed paths
            func = partial(self.integrand, x0=x0, fixed=True)
        else:
            lims = [[x_min, x_max]] * self.N  # Integrate over all N points for periodic paths
            func = partial(self.integrand, fixed=False)

        integ = vegas.Integrator(lims)  # Create VEGAS integrator
        S_lat = integ(func, nitn=10, neval=100000)  # Perform integration

        return S_lat.mean  # Return the mean value of the integral
    
    def analytic(self, x):
        """
        Computes the analytical solution for the harmonic potential.

        Parameters:
        - x: Positions at which to evaluate the solution

        Returns:
        - Analytical propagator values
        """
        if self.potential == 'harmonic':
            x = np.array(x)
            return (np.exp(-1.0 * x ** 2) / np.sqrt(np.pi)) * np.exp(-1.0 * 1 / 2 * self.a * self.N)
    
    def compute_analysis(self, analytical=True):
        """
        Computes the numerical and (optional) analytical propagators.

        Parameters:
        - analytical: Boolean indicating whether to compute analytical solutions

        Returns:
        - A dictionary containing numerical and analytical results
        """
        x0_list = np.linspace(0, 2, 10)  # List of boundary values
        prop_numerical = np.empty(x0_list.shape)  # Initialize numerical propagators
        for i, x0 in enumerate(x0_list):
            prop_numerical[i] = self.vegas_integrate(x0=x0_list[i], fixed=True)
        
        # Compute analytical solutions for harmonic potential
        if self.potential == 'harmonic':
            expTE = self.vegas_integrate(fixed=False)  # Calculate trace of the exponential
            E0 = -1.0 * np.log(expTE) / (self.a * self.N)  # Ground state energy
            psi2_analytical = np.exp(-x0_list ** 2) / np.sqrt(np.pi) if analytical else None
            prop_analytical = self.analytic(x0_list) if analytical else None
            return {
                "x0_list": x0_list,
                "prop_numerical": prop_numerical,
                "psi2_numerical": prop_numerical / expTE,
                "prop_analytical": prop_analytical,
                "psi2_analytical": psi2_analytical,
                "E0": E0,
            }
        else:
            return {
                "x0_list": x0_list,
                "prop_numerical": prop_numerical,
                "psi2_numerical": None,
            }

    def plot_results(self, analysis_data):
        """
        Plots the results of the numerical and analytical propagators.

        Parameters:
        - analysis_data: Dictionary containing analysis results
        """
        x0_list = analysis_data["x0_list"]
        prop_numerical = analysis_data["prop_numerical"]

        # Plot propagator
        plt.figure(figsize=(8, 6), dpi=100)
        plt.scatter(x0_list, prop_numerical, color="green", label="Numerical Solution")
        if self.potential == 'harmonic' and "prop_analytical" in analysis_data:
            plt.plot(x0_list, analysis_data["prop_analytical"], color="black", linestyle="--", label="Analytical Solution")
        plt.xlabel("x_0 = x_N")
        plt.title(f"Closed Euclidean Propagator, a = {self.a}, {self.potential} potential")
        plt.xlim(0, 2)
        plt.ylim(0, 0.1)
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot squared wavefunction
        if self.potential == 'harmonic' and "psi2_numerical" in analysis_data:
            plt.figure(figsize=(8, 6), dpi=100)
            plt.scatter(x0_list, analysis_data["psi2_numerical"], color="green", label="Numerical Solution")
            if "psi2_analytical" in analysis_data:
                plt.plot(x0_list, analysis_data["psi2_analytical"], color="black", linestyle="--", label="Analytical Solution")
            plt.xlabel("x_0 = x_N")
            plt.ylabel(r"$|\psi(x)_0|^2$")
            plt.title(f"a = {self.a}, {self.potential} potential")
            plt.xlim(0, 2)
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True)
            plt.show()

        # Print ground state energy if computed
        if self.potential == 'harmonic' and "E0" in analysis_data:
            print("Ground State Energy (E0):", analysis_data["E0"])

if __name__ == "__main__":
    N = 8  # Number of points in the path
    a = 0.5  # Lattice spacing
    m = 1  # Mass of the particle
    potential = 'quartic'  # Potential type

    # Create an instance of the PathIntegralQuantum class
    NPI = NumericalPathIntegral(N, a, m, potential)
    
    # Perform the analysis
    analysis_data = NPI.compute_analysis()
    
    # Plot the results
    NPI.plot_results(analysis_data)
