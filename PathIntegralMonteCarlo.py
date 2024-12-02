import numpy as np
import matplotlib.pyplot as plt


class PathIntegralMonteCarlo:
    def __init__(self, N = 20, a = 0.5, m=1, potential = 'harmonic', N_cf = 10000, w=1, eps=1.4, N_cor=20, source='x', imp='noghost'):
        """
        Initializes the PathIntegralMonteCarlo instance.

        Args:
            numerical_integral (NumericalPathIntegral): Instance of NumericalPathIntegral containing mass (m), time step (a), and number of time slices (N).
            w (float): Frequency of the harmonic oscillator. Defaults to 1.
            eps (float): Step size parameter for the Metropolis update. Defaults to 1.4.
            N_cor (int): Number of correlation steps between updates. Defaults to 20.
            N_cf (int): Number of configurations for Monte Carlo sampling. Defaults to 1000.
            source (str): Source type for the propagator calculation ('x' or 'x3'). Defaults to 'x'.
            imp (str): Type of improvement used ('n', 'y', 'noghost'). Defaults to 'noghost'.
        """
        valid_imp = ['n', 'y', 'noghost']
        if imp not in valid_imp:
            raise ValueError(f"Invalid option '{imp}'. Valid options are: {', '.join(valid_imp)}")
        
        valid_pot = ['harmonic', 'anharmonic']
        if potential not in valid_pot:
            raise ValueError(f"Invalid option '{potential}'. Valid options are: {', '.join(valid_pot)}")
            
        self.m = m
        self.a = a
        self.N = N
        self.potential = potential
        self.w = w
        self.eps = eps
        self.N_cor = N_cor
        self.N_cf = N_cf
        self.source = source
        self.imp = imp
        
        self.x = np.zeros(self.N, dtype=np.float64)
        self.G = np.zeros((self.N_cf, self.N), dtype=np.float64)

    
    def update(self):
        """
        Metropolis update for the path configuration.
        This function performs an update step for each element of the path configuration,
        using the Metropolis criterion to accept or reject the change.
        Inspired by the update algorithm described in LePage's article, page 8.
        """
        accepted_moves = 0  # Counter for accepted moves
        total_moves = 0    # Counter for total moves
        
        for j in range(self.N):
            old_x = self.x[j]
            old_Sj = self.S(j, self.x)
            self.x[j] += np.random.uniform(-self.eps, self.eps)  # Propose a new value for x[j]
            dS = self.S(j, self.x) - old_Sj  # Compute the change in the action
            total_moves += 1  # Increment the total moves counter
            if dS <= 0 or np.exp(-dS) >= np.random.uniform(0, 1):
                accepted_moves += 1  # Increment the accepted moves counter if accepted
            else:
                self.x[j] = old_x  # Revert to the previous value if not accepted
        
        acceptance_rate = accepted_moves / total_moves if total_moves > 0 else 0

        return acceptance_rate


    def S(self, j, x):
        """
        Computes the local contribution to the action for the given path configuration,
        both for the harmonic and the anharmonic case.

        Args:
            j (int): Index of the current path point.
            x (ndarray): Array representing the path configuration.

        Returns:
            float: Local contribution to the action.
        
        Inspired by the S(j,x) algorithm described in LePage's article, page 8.
        Improved discretized methods are also included.
        
        """
        jp = (j + 1) % self.N
        jm = (j - 1) % self.N
        jm2 = (j - 2) % self.N
        jp2 = (j + 2) % self.N
        
        if self.potential == 'harmonic':
    
            if self.imp == 'n':
                return self.a * self.m * (self.w ** 2) * x[j] ** 2 / 2 + self.m * x[j] * (x[j] - x[jp] - x[jm]) / self.a
            elif self.imp == 'y':
                return self.a * self.m * (self.w ** 2) * x[j] ** 2 / 2 - (self.m / (2 * self.a)) * x[j] * (-(x[jm2] + x[jp2]) / 6 + (x[jm] + x[jp]) * (8 / 3) - x[j] * (5 / 2))
            elif self.imp == 'noghost':
                return self.a * self.m * (self.w ** 2) * (1 + (self.a * self.w) ** 2 / 12) * x[j] ** 2 / 2 + self.m * x[j] * (x[j] - x[jp] - x[jm]) / self.a
        
        elif self.potential == 'anharmonic':
            c = 2  # anharmonicity constant
            if self.imp == 'n':
                return 0.5 * self.a * self.m * (self.w ** 2) * (x[j]**2) * (1 + c * self.m * self.w * x[j]**2) + self.m * x[j] * (x[j]-x[jp]-x[jm])/self.a
            elif self.imp == 'y':
                S_kinetic = (
                    - (self.m / (2 * self.a)) *
                    x[j] * (
                        - (x[jm2] + x[jp2]) / 6 +
                        (8 / 3) * (x[jm] + x[jp]) -
                        (5 / 2) * x[j]
                    )
                )
                S_potential = (
                    self.a * self.m * (self.w**2 / 2) * x[j]**2 * (1 + c * self.m * self.w * x[j]**2)
                )
                return S_potential + S_kinetic
            elif self.imp == 'noghost':
                S_correction = (
                    (self.a**2 / 24) * (x[j] + 2 * c * self.m * self.w * x[j]**3)**2  # Correzione per i modi ghost
                    - (self.a**2 / 4) * c * self.m * self.w**3 * x[j]**2              # Termine correttivo quadratico
                    + (self.a**3 / 2) * (0.25 * c * self.m * self.w**3 * x[j]**2)**2  # Termine correttivo quartico
                    )
                S_kinetic = self.m * x[j] * (x[j] - x[jp] - x[jm]) / self.a           # Termine cinetico
                S_potential = (
                    self.a * self.m * (self.w**2 / 2) * (1 + c * self.m * self.w**2 * x[j]**2) * x[j]**2  # Potenziale migliorato
                    )
                return S_potential + S_correction + S_kinetic
                
    def compute_G(self, x, n):
        """
        Computes the two-point correlation function G for a given path configuration.

        Args:
            x (ndarray): Array representing the path configuration.
            n (int): Time separation for correlation function.

        Returns:
            float: Computed value of the correlation function.
        
        Inspired by the compute_G(x,n) method described in LePage's article, page 8.
        """
        if self.source == 'x3':
            return np.sum((x ** 3) * (np.roll(x, -n) ** 3)) / self.N
        elif self.source == 'x':
            return np.sum(x * np.roll(x, -n)) / self.N

    def MCaverage(self, x, G):
        """
        Computes the Monte Carlo average of the two-point correlation function G,
        and prints the average acceptance rate over all updates.
        
        Args:
            x (ndarray): Array representing the path configuration.
            G (ndarray): Array to store the computed correlation functions for each configuration.
        """
        x.fill(0)
        total_acceptance_rate = 0  # Accumulator for acceptance rates
        num_updates = 0  # Counter for the number of updates
        
        # Thermalization step
        for _ in range(10 * self.N_cor):
            total_acceptance_rate += self.update()
            num_updates += 1
            
            # Compute configurations
        for alpha in range(self.N_cf):
            for _ in range(self.N_cor):
                total_acceptance_rate += self.update()
                num_updates += 1
            for n in range(self.N):
                G[alpha][n] = self.compute_G(x, n)

        # Calculate the average acceptance rate
        avg_acceptance_rate = total_acceptance_rate / num_updates

        # Print the average acceptance rate
        print(f"Average acceptance rate over all updates: {avg_acceptance_rate * 100:.2f}%")

    def deltaE(self, G_avgd_over_paths):
        """
        Calculates the excitation energy from the correlation function.

        Args:
            G_avgd_over_paths (ndarray): Array of averaged correlation function values.

        Returns:
            ndarray: Array of energy differences (excitation energies).
        
        The energy calculation is based on the formula described on page 9 of LePage's article.
        """
        return np.log(np.abs(G_avgd_over_paths[:-1] / G_avgd_over_paths[1:])) / self.a

    def avg(self, G):
        """
        Computes the average of the given ensemble of correlation functions.

        Args:
            G (ndarray): Array of correlation functions.

        Returns:
            ndarray: Averaged correlation function.
        """
        return np.mean(G, axis=0)

    def sdev(self, G):
        """
        Computes the standard deviation of the given ensemble of correlation functions.

        Args:
            G (ndarray): Array of correlation functions.

        Returns:
            ndarray: Standard deviation of the correlation function.
        """
        return np.std(G, axis=0)

    def bootstrap(self, G):
        """
        Generates a bootstrap copy of the given ensemble of correlation functions for error estimation.

        Args:
            G (ndarray): Array of correlation functions.

        Returns:
            ndarray: Bootstrap copy of the correlation function ensemble.
        
        The bootstrap method is inspired by the description on page 12 of LePage's article.
        """
        N_bs = len(G)
        return G[np.random.randint(0, N_bs, size=N_bs)]

    def bin(self, G, binsize):
        """
        Bins the given ensemble of correlation functions to reduce correlations.

        Args:
            G (ndarray): Array of correlation functions.
            binsize (int): Size of each bin.

        Returns:
            ndarray: Binned ensemble of correlation functions.
        
        Binning is used to reduce correlations between configurations, as discussed in LePage's article, page 13.
        """
        return np.array([np.mean(G[i:i + binsize], axis=0) for i in range(0, len(G), binsize)])

    def bootstrap_deltaE(self, G, nbstrap=100):
        """
        Calculates the bootstrap estimate for the excitation energy and its standard deviation.

        Args:
            G (ndarray): Array of correlation functions.
            nbstrap (int): Number of bootstrap samples. Defaults to 100.

        Returns:
            tuple: Averaged excitation energies and their standard deviations.
        """
        bsE = np.empty((nbstrap, self.N - 1))
        for i in range(nbstrap):
            g = self.bootstrap(G)
            bsE[i] = self.deltaE(self.avg(g))
        return self.avg(bsE), self.sdev(bsE)

    def run(self):
        """
        Runs the Monte Carlo simulation to calculate the energy levels of the system and plots the results.
        """
        self.MCaverage(self.x, self.G)

        stand_dev = self.sdev(self.G)
        binsize = 20
        binned_G = self.bin(self.G, binsize)
        stand_dev_binned = self.sdev(binned_G)

        print(stand_dev)
        print(stand_dev_binned)

        avgE, sdevE = self.bootstrap_deltaE(binned_G, nbstrap=10000)
        t = np.arange(self.N - 1) * self.a

        plt.title(f'$\epsilon$={self.eps}, a={self.a}, N={self.N}, N_cor={self.N_cor}, N_cf={self.N_cf}')
        plt.errorbar(t, avgE, yerr=sdevE, fmt="o", label='computed')
        if self.potential == 'harmonic':
            plt.plot(t, np.ones(len(t)), label='exact')
        elif self.potential == 'anharmonic':
            plt.plot(t, 1.933*np.ones(len(t)), label='exact')
        plt.legend()
        plt.axis([-0.1, 8, -5, 5])

        plt.show()

if __name__ == '__main__':
    
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


    


