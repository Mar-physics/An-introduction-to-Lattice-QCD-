import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import vegas

class NumericalPathIntegral:
    def __init__(self, N, a, m, potential='harmonic'):
        self.N = N
        self.a = a
        self.m = m
        self.potential = potential

        # Definizione del potenziale
        if potential == 'harmonic':
            self.V = lambda x: 0.5 * x ** 2
        elif potential == 'anharmonic':
            self.V = lambda x: 0.5 * x ** 4
        else:
            raise ValueError("Potential not recognized. Available: 'harmonic', 'anharmonic'")
    
    def integrand(self, x, x0=None, fixed=True):
        S_lat = 0
        if fixed:
            S_lat = (self.m / (2 * self.a)) * (x[0] - x0) ** 2 + self.a * self.V(x0)
            S_lat += (self.m / (2 * self.a)) * (x0 - x[-1]) ** 2 + self.a * self.V(x[-1])
        else:
            S_lat = (self.m / (2 * self.a)) * (x[0] - x[-1]) ** 2 + self.a * self.V(x[-1])
            
        S_lat += (self.m / (2 * self.a)) * np.sum((x[1:] - x[:-1]) ** 2) + self.a * np.sum(self.V(x[:-1]))
        A = (self.m / (2 * np.pi * self.a)) ** (self.N / 2)

        return A * np.exp(-1.0 * S_lat)
    
    def vegas_integrate(self, x0=None, fixed=True, x_min=None, x_max=None):
        if x_min is None and x_max is None:
            x_min, x_max = -5, 5
        
        if x_min > x_max:
            raise ValueError("x_min cannot be greater than x_max.")
            
        if fixed and x0 is None:
            raise ValueError("x0 must be provided when fixed=True")

        if fixed:
            lims = [[x_min, x_max]] * (self.N - 1)
            func = partial(self.integrand, x0=x0, fixed=True)
        else:
            lims = [[x_min, x_max]] * self.N
            func = partial(self.integrand, fixed=False)

        integ = vegas.Integrator(lims)
        S_lat = integ(func, nitn=10, neval=100000)

        return S_lat.mean
    
    def analytic(self, x):
        if self.potential == 'harmonic':
            x = np.array(x)
            return (np.exp(-1.0 * x ** 2) / np.sqrt(np.pi)) * np.exp(-1.0 * 1 / 2 * self.a * self.N)
    
    def compute_analysis(self, analytical=True):
        x0_list = np.linspace(0, 2, 10)
        prop_numerical = np.empty(x0_list.shape)
        for i, x0 in enumerate(x0_list):
            prop_numerical[i] = self.vegas_integrate(x0=x0_list[i], fixed=True)
        
        # Soluzioni analitiche (solo per potenziale armonico)
        if self.potential == 'harmonic':
            expTE = self.vegas_integrate(fixed=False)
            E0 = -1.0 * np.log(expTE) / (self.a * self.N)
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
        x0_list = analysis_data["x0_list"]
        prop_numerical = analysis_data["prop_numerical"]

        # Plot propagatore
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

        # Plot funzione d'onda al quadrato
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

        # Stampa \(E_0\) solo se calcolato
        if self.potential == 'harmonic' and "E0" in analysis_data:
            print("Ground State Energy (E0):", analysis_data["E0"])

if __name__ == "__main__":
    N = 8
    a = 0.5
    m = 1
    potential = 'anharmonic'

    # Create an instance of the PathIntegralQuantum class
    NPI = NumericalPathIntegral(N, a, m, potential)
    
    # Perform the analysis
    analysis_data = NPI.compute_analysis()
    
    # Plot the results
    NPI.plot_results(analysis_data)