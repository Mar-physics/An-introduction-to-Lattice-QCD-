import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numba import njit

class EOMSolver:
    def __init__(self, N=100, w0=1, m=1, a=np.linspace(0.1, 0.5, 100)):
        
        """
        Initialize the Equation of Motion (EOM) solver.

        Parameters:
        N (int): Number of points in the lattice.
        w0 (float): Frequency parameter.
        m (float): Mass parameter (currently unused).
        a (array): Array of lattice spacings.
        """
        
        self.N = N  # Number of points
        self.w0 = w0  # Frequency parameter
        self.m = m  # Mass (currently unused but kept for generality)
        self.a = a  # Array of lattice spacings
        self.T = self.N * self.a  # Total time for each lattice spacing

    @staticmethod
    @njit
    def build_not_imp_matrix(N, a, w0):
        
        """
        Build the matrix for the unimproved action equation of motion.

        Parameters:
        N (int): Number of points in the lattice.
        a (float): Lattice spacing.
        w0 (float): Frequency parameter.

        Returns:
        ndarray: The coefficient matrix for the unimproved action.
        """
        
        diagonal = -(2 + a**2 * w0**2)  # Main diagonal elements
        off_diagonal = 1  # Off-diagonal elements

        matrix = np.zeros((N, N))  # Initialize an NxN matrix

        for i in range(N):
            matrix[i, i] = diagonal  # Main diagonal
            if i > 0:
                matrix[i, i - 1] = off_diagonal  # Lower diagonal
            if i < N - 1:
                matrix[i, i + 1] = off_diagonal  # Upper diagonal
        
        return matrix

    @staticmethod
    @njit
    def build_imp_matrix(N, a, w0):
        
        """
        Build the matrix for the improved action equation of motion.

        Parameters:
        N (int): Number of points in the lattice.
        a (float): Lattice spacing.
        w0 (float): Frequency parameter.

        Returns:
        ndarray: The coefficient matrix for the improved action.
        """
        
        coeff_j = -(5 / 2 + a**2 * w0**2)  # Main diagonal
        coeff_j_pm_1 = 4 / 3  # First off-diagonals
        coeff_j_pm_2 = -1 / 12  # Second off-diagonals

        matrix = np.zeros((N, N))

        for i in range(N):
            matrix[i, i] = coeff_j
            if i > 0:
                matrix[i, i - 1] = coeff_j_pm_1
            if i < N - 1:
                matrix[i, i + 1] = coeff_j_pm_1
            if i > 1:
                matrix[i, i - 2] = coeff_j_pm_2
            if i < N - 2:
                matrix[i, i + 2] = coeff_j_pm_2
        
        return matrix
    
    @staticmethod
    @njit
    def build_no_ghost_matrix(N, a, w0):
        
        """
        Build the matrix for the improved action equation of motion with no ghost states.

        Parameters:
        N (int): Number of points in the lattice.
        a (float): Lattice spacing.
        w0 (float): Frequency parameter.

        Returns:
        ndarray: The coefficient matrix for the improved action.
        """
        
        diagonal = -(2 + (a*w0)**2*(1+(a*w0)**2/12))  # Main diagonal elements
        off_diagonal = 1  # Off-diagonal elements
        
        matrix = np.zeros((N, N))  # Initialize an NxN matrix
        
        for i in range(N):
            matrix[i, i] = diagonal  # Main diagonal
            if i > 0:
                matrix[i, i - 1] = off_diagonal  # Lower diagonal
            if i < N - 1:
                matrix[i, i + 1] = off_diagonal  # Upper diagonal
                
        return matrix
        
        
    @staticmethod
    @njit
    def BC_not_improved(N, xi, xf):
        
        """
        Construct the Dirichlet boundary conditions vector for the unimproved action.

        Parameters:
        N (int): Number of points in the lattice.
        xi (float): Boundary condition at the initial point.
        xf (float): Boundary condition at the final point.

        Returns:
        ndarray: The boundary conditions vector.
        """
        
        B = np.zeros(N)
        B[0] = -xi
        B[-1] = -xf
        return B

    @staticmethod
    @njit
    def BC_imp(N, xi, xf, xi2, xf2):
        
        """
        Construct the boundary conditions vector for the improved action.

        Parameters:
        N (int): Number of points in the lattice.
        xi (float): Boundary condition at the initial point.
        xf (float): Boundary condition at the final point.
        xi2 (float): Boundary condition at the second initial point.
        xf2 (float): Boundary condition at the second final point.

        Returns:
        ndarray: The boundary conditions vector.
        """
        
        B = np.zeros(N)
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

    def perform_not_imp_analysis(self):
        
        """
        Perform the analysis using the unimproved action.

        Returns:
        ndarray: The squared frequencies for each lattice spacing.
        """
        
        w2_not_imp = np.empty(len(self.a))
        for i, ai in enumerate(self.a):
            t = np.arange(0, self.N) * ai
            xi = np.exp(0)  # Boundary condition at t_0
            xf = np.exp(-self.T[i])  # Boundary condition at t_N

            A = self.build_not_imp_matrix(self.N, ai, self.w0)
            B = self.BC_not_improved(self.N, xi, xf)
            X = np.linalg.solve(A, B)

            w2_not_imp[i] = self.fit_solution(X, t)

        return w2_not_imp

    def perform_imp_analysis(self):
        
        """
        Perform the analysis using the improved action.

        Returns:
        ndarray: The squared frequencies for each lattice spacing.
        """
        
        w2_imp = np.empty(len(self.a))
        for i, ai in enumerate(self.a):
            t = np.arange(0, self.N) * ai
            xi = np.exp(0)  # Boundary condition at t_0
            xf = np.exp(-self.T[i])  # Boundary condition at t_N
            xi2 = np.exp(-(-ai))  # Condition at t_{-2}
            xf2 = np.exp(-self.T[i] + ai)  # Condition at t_{N+1}

            A_imp = self.build_imp_matrix(self.N, ai, self.w0)
            B_imp = self.BC_imp(self.N, xi, xf, xi2, xf2)
            X_imp = np.linalg.solve(A_imp, B_imp)

            w2_imp[i] = self.fit_solution(X_imp, t)

        return w2_imp
    
    def perform_no_ghost_analysis(self):
        
        """
        Perform the analysis using the improved action with no ghost states.

        Returns:
        ndarray: The squared frequencies for each lattice spacing.
        """
        
        w2_no_ghost = np.empty(len(self.a))
        for i, ai in enumerate(self.a):
            t = np.arange(0, self.N) * ai
            xi = np.exp(0)  # Boundary condition at t_0
            xf = np.exp(-self.T[i])  # Boundary condition at t_N

            A = self.build_no_ghost_matrix(self.N, ai, self.w0)
            B = self.BC_not_improved(self.N, xi, xf)
            X = np.linalg.solve(A, B)

            w2_no_ghost[i] = self.fit_solution(X, t)

        return w2_no_ghost

    def expected_frequency_not_imp(self):
        
        """
        Calculate the expected squared frequencies for the unimproved action.

        Returns:
        ndarray: The expected squared frequencies.
        """
        
        return self.w0**2 * (1 - (self.a * self.w0)**2 / 12)

    def expected_frequency_imp(self):
        
        """
        Calculate the expected squared frequencies for the improved action.

        Returns:
        ndarray: The expected squared frequencies.
        """
        
        return self.w0**2 * (1 + (self.a * self.w0)**4 / 90)
    
    def expected_frequency_no_ghost(self):
        
        """
        Calculate the expected squared frequencies for the improved action with no ghost.

        Returns:
        ndarray: The expected squared frequencies.
        """
        
        return self.w0**2 * (1 - (self.a * self.w0)**4 / 360)

    def plot_results(self, mode='combined'):
        """
        Plot results for unimproved, improved, or combined actions.
        
        Parameters:
            mode (str): 'not_imp' for unimproved only, 
                'imp' for improved only,
                'combined' for both combined in one graph,
                'no_ghost' for ghost-free improved results,
                'all' for all individual and combined plots.
        """
        w2_not_imp = self.perform_not_imp_analysis()
        expected_not_imp = self.expected_frequency_not_imp()
        w2_imp = self.perform_imp_analysis()
        expected_imp = self.expected_frequency_imp()
        w2_no_ghost = self.perform_no_ghost_analysis()
        expected_no_ghost = self.expected_frequency_no_ghost()

        if mode in ['not_imp', 'all']:
            # Plot unimproved results
            plt.figure(figsize=(8, 6))
            plt.scatter(self.a, w2_not_imp, s=10, label='Unimproved Frequency')
            plt.plot(self.a, expected_not_imp, label='Expected Unimproved')
            plt.xlabel('Lattice spacing')
            plt.ylabel('$\\omega^2$')
            plt.title('Unimproved Results')
            plt.legend()
            plt.show()

        if mode in ['imp', 'all']:
            # Plot improved results
            plt.figure(figsize=(8, 6))
            plt.scatter(self.a, w2_imp, s=10, color = 'orange', label='Improved Frequency')
            plt.plot(self.a, expected_imp, color = 'orange', label='Expected Improved')
            plt.xlabel('Lattice spacing')
            plt.ylabel('$\\omega^2$')
            plt.title('Improved Results')
            plt.legend()
            plt.show()

        if mode in ['no_ghost', 'all']:
            # Plot ghost-free improved results
            plt.figure(figsize=(8, 6))
            plt.scatter(self.a, w2_no_ghost, s=10, color = 'green', label='No-Ghost Improved Frequency')
            plt.plot(self.a, expected_no_ghost, color = 'green', label='Expected No-Ghost Improved')
            plt.xlabel('Lattice spacing')
            plt.ylabel('$\\omega^2$')
            plt.title('No-Ghost Improved Results')
            plt.legend()
            plt.show()

        if mode in ['combined', 'all']:
            # Plot combined results
            plt.figure(figsize=(8, 6))
            plt.scatter(self.a, w2_not_imp, s=10, label='Unimproved Frequency')
            plt.plot(self.a, expected_not_imp, label='Expected Unimproved')
            plt.scatter(self.a, w2_imp, s=10, label='Improved Frequency')
            plt.plot(self.a, expected_imp, label='Expected Improved')
            plt.scatter(self.a, w2_no_ghost, s=10, label='No-Ghost Improved Frequency')
            plt.plot(self.a, expected_no_ghost, label='Expected No-Ghost Improved')
            plt.xlabel('Lattice spacing')
            plt.ylabel('$\\omega^2$')
            plt.title('Combined Results')
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
            