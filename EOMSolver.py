import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

class EOMSolver:
    def __init__(self, N=100, w0=1, m=1, a=np.linspace(0.1, 0.5, 100), mode = 'n'):
        
        """
        Initialize the Equation of Motion (EOM) solver.

        Parameters:
        N (int): Number of points in the lattice.
        w0 (float): Frequency parameter.
        m (float): Mass parameter (currently unused).
        a (array): Array of lattice spacings.
        mode (str): Mode for the action ('n' for unimproved, 'y' for improved, 'no_ghost' for ghost-free improved).
        """
        
        valid_modes = ['n', 'y', 'no_ghost']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes are: {', '.join(valid_modes)}")
        
        self.N = N  # Number of points
        self.w0 = w0  # Frequency parameter
        self.m = m  # Mass (currently unused but kept for generality)
        self.a = a  # Array of lattice spacings
        self.T = self.N * self.a  # Total time for each lattice spacing
        self.mode = mode
        
        
    
    def build_matrix(self, a):
        
        matrix = np.zeros((self.N, self.N))  # Initialize an NxN matrix
        
        if self.mode == 'n':
            diagonal = -(2 + a**2 * self.w0**2)  # Main diagonal elements
            off_diagonal = 1  # Off-diagonal elements
            
            for i in range(self.N):
                matrix[i, i] = diagonal  # Main diagonal
                if i > 0:
                    matrix[i, i - 1] = off_diagonal  # Lower diagonal
                if i < self.N - 1:
                    matrix[i, i + 1] = off_diagonal  # Upper diagonal
            
        elif self.mode == 'y':
            coeff_j = -(5 / 2 + a**2 * self.w0**2)  # Main diagonal
            coeff_j_pm_1 = 4 / 3  # First off-diagonals
            coeff_j_pm_2 = -1 / 12  # Second off-diagonals

            for i in range(self.N):
                matrix[i, i] = coeff_j
                if i > 0:
                    matrix[i, i - 1] = coeff_j_pm_1
                if i < self.N - 1:
                    matrix[i, i + 1] = coeff_j_pm_1
                if i > 1:
                    matrix[i, i - 2] = coeff_j_pm_2
                if i < self.N - 2:
                    matrix[i, i + 2] = coeff_j_pm_2
                    
        elif self.mode == 'no_ghost':
            
            diagonal = -(2 + (a * self.w0)**2 * (1 + (a * self.w0)**2 / 12))  # Main diagonal elements
            off_diagonal = 1  # Off-diagonal elements
            
            for i in range(self.N):
                matrix[i, i] = diagonal  # Main diagonal
                if i > 0:
                    matrix[i, i - 1] = off_diagonal  # Lower diagonal
                if i < self.N - 1:
                    matrix[i, i + 1] = off_diagonal  # Upper diagonal
        
        return matrix
    
    
    def boundary_conditions(self, xi, xf, xi2, xf2):
        
        B = np.zeros(self.N)
        
        if self.mode in ['n', 'no_ghost']:
            B[0] = -xi
            B[-1] = -xf
        
        elif self.mode == 'y':
            B[0] = (1 / 12) * xi2 - (4 / 3) * xi
            B[1] = (1 / 12) * xi
            B[-2] = (1 / 12) * xf
            B[-1] = (1 / 12) * xf2 - (4 / 3) * xf
        
        return B
    
    @staticmethod
    def fit_solution(solution, t):
        
        """
        Fit the solution to an exponential function and extract the squared frequency.
       
        Parameters:
            solution (ndarray): The computed solution.
            t (ndarray): The time array.

        Returns:
            float: The squared frequency extracted from the fit.
       """
       
        def exp_func(t, A, w, phi):
            return A * np.exp(-w * t + phi)

        param, _ = optimize.curve_fit(exp_func, t, solution, p0=(0.1, 1, 0))
        return param[1] ** 2
    
    def perform_analysis(self):
        
        w2 = np.empty(len(self.a))
        for i, ai in enumerate(self.a):
            t = np.arange(0, self.N) * ai
            xi = np.exp(0)  # Boundary condition at t_0
            xf = np.exp(-self.T[i])  # Boundary condition at t_N
            xi2 = np.exp(-(-ai))  # Condition at t_{-2}
            xf2 = np.exp(-self.T[i] + ai)  # Condition at t_{N+1}

            A = self.build_matrix(ai)
            B = self.boundary_conditions(xi, xf, xi2, xf2)
            X = np.linalg.solve(A, B)

            w2[i] = self.fit_solution(X, t)

        return w2
    
    
    def expected_frequency_squared(self):
        
        if self.mode == 'n':
            return self.w0**2 * (1 - (self.a * self.w0)**2 / 12)
        
        if self.mode == 'y':
            return self.w0**2 * (1 + (self.a * self.w0)**4 / 90)
        
        if self.mode == 'no_ghost':
            return self.w0**2 * (1 - (self.a * self.w0)**4 / 360)
    
    
    def plot_results(self):
        
        w2 = self.perform_analysis()
        expected = self.expected_frequency_squared()
        plt.figure(figsize=(8, 6))
        
        if self.mode == 'n':
            plt.title('Results for not improved discretized S[x]')
            plt.scatter(self.a, w2, s=10, label='Computed Frequency Squared')
            plt.plot(self.a, expected, label='Expected')
            
        elif self.mode == 'y':
            plt.title('Results for improved discretized S[x]')
            plt.scatter(self.a, w2, s=10, color = 'orange', label='Computed Frequency Squared')
            plt.plot(self.a, expected, color = 'orange', label='Expected')
            
        elif self.mode == 'no_ghost':
            plt.title('Results for no-ghost discretized S[x]')
            plt.scatter(self.a, w2, s=10, color = 'green', label='Computed Frequency Squared')
            plt.plot(self.a, expected, color = 'green', label='Expected')
            
        plt.xlabel('Lattice spacing: a')
        plt.ylabel('$\\omega^2$')
        plt.legend()
        plt.show()
    
    def compute_ghost_mode(self):
        """
        Compute the ghost mode frequency for the improved action and fit it to the expected theoretical relation.
        
        Returns:
            dict: Contains lattice spacings, numerical ghost mode frequencies, theoretical frequencies, and fit results.
       """
        ghost = np.empty(len(self.a))

        for i, ai in enumerate(self.a):
            def f(w):
                # Ghost mode equation from the improved discretization
                return (
                    np.exp(-2 * ai * w) - 16 * np.exp(-ai * w) + 30
                    - 16 * np.exp(ai * w) + np.exp(2 * ai * w) 
                    )

            # Use a better initial guess close to the theoretical ghost mode
            initial_guess = 2.6 / ai

            # Solve numerically, using the improved guess
            sol = optimize.fsolve(f, initial_guess)
            
            ghost[i] = sol[0]  # Store the solution
        
        # Calculate w^2
        ghost_squared = ghost**2
        
        # Define the theoretical expectation for ghost mode
        def expectation(a, C):
            return (C / a)**2

        # Fit the numerical ghost modes to the theoretical form
        popt, _ = optimize.curve_fit(expectation, self.a, ghost_squared)
        C_fit = popt[0]  # Extract the fitted constant

        # Generate the fitted curve for w^2
        fit_squared = expectation(self.a, C_fit)
        
        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.scatter(self.a, ghost_squared, label='Numerical Ghost Mode', s=10)
        plt.plot(self.a, fit_squared, label=f'Fit: $C/a$, C={C_fit:.4f}', linestyle='--', color='red')
        plt.xlabel('Lattice spacing $a$')
        plt.ylabel('$\omega$')
        plt.title('Ghost Mode Analysis for Improved Action')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    
    """# Modalità da analizzare
    modes = ['n', 'y', 'no_ghost']

    # Colori per i grafici
    colors = {
        'n': 'blue',
        'y': 'orange',
        'no_ghost': 'green'
    }
    
    labels = {
        'n': 'not improved',
        'y': 'improved',
        'no_ghost': 'no ghost'
        }

    # Dati per il grafico combinato
    results = {}
    plt.figure(figsize=(10, 6))

    for mode in modes:
        solver = EOMSolver(mode=mode)
        w2 = solver.perform_analysis()
        expected = solver.expected_frequency_squared()

        # Salva i risultati per il grafico combinato
        results[mode] = (w2, expected)
        
        # Plot individuale
        plt.scatter(solver.a, w2, s=10, label=labels[mode], color=colors[mode])
        plt.plot(solver.a, expected, linestyle='--', label=f'Expected ({mode}) - {labels[mode]}', color=colors[mode])

    # Impostazioni del grafico combinato
    plt.xlabel('Lattice spacing $a$')
    plt.ylabel('$\\omega^2$')
    plt.title('Combined Results for All Modes')
    plt.legend()
    plt.grid(True)

    # Mostra il grafico combinato
    plt.show()"""
    solver = EOMSolver(mode = 'no_')
    solver.plot_results()
        
            
        
        
        