import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from WilsonLatticeUtils import WilsonLatticeUtils

class StaticPotentialUtils(WilsonLatticeUtils):
    def __init__(self, N, dim, N_matrix, eps, beta, beta_imp, u0, use_improved=False, smearing = True, a = 0.25, smeared_eps = 1. / 12.):
    
        """
        Initializes the StaticPotentialUtils class, inheriting from WilsonLatticeUtils.

        Parameters:
        - N: Lattice size.
        - dim: Number of dimensions.
        - N_matrix: Number of matrices (e.g., for SU(3)).
        - eps: Lattice spacing parameter.
        - beta: Coupling constant.
        - beta_imp: Improved coupling constant.
        - u0: Tadpole improvement factor.
        - use_improved: Whether to use the improved Wilson action.
        - smearing: Whether to apply smearing.
        - a: Lattice spacing in physical units.
        - smeared_eps: Smearing parameter.
        """
        
        super().__init__(N, dim, N_matrix, eps, beta, beta_imp, u0, use_improved)
        self.smearing = smearing
        self.a = a
        self.smeared_eps = smeared_eps
           
    def V_exact(self, r, sigma, b, c):
        
        """
        Computes the exact static quark potential using the Cornell potential formula:
        
        V(r) = sigma * r - b / r + c

        Parameters:
        - r: Quark separation distance.
        - sigma: String tension.
        - b: Coulomb coefficient.
        - c: Constant shift.

        Returns:
        - The exact static potential value at distance r.
        """
        
        return sigma*r - b/r + c
    
    def planar_loops_potential(self, lattice, point, length, duration):
        
        """
        Computes the expectation value of a planar Wilson loop, which contributes to the 
        static quark potential calculation.

        The Wilson loop is constructed by moving along a spatial direction and 
        a temporal direction, forming a rectangular loop.

        Parameters:
        - lattice: The gauge field lattice.
        - point: Starting position of the loop.
        - length: Spatial extent of the loop.
        - duration: Temporal extent of the loop.

        Returns:
        - The expectation value of the planar Wilson loop.
        """
        
        W_planar = 0
        
        # Iterate over spatial directions (1,2,3), excluding time (0)
        for space_direction in range(1, self.dim):  
            loop = np.identity(3, np.complex128)
        
            # Construct the loop in the time direction
            for time in range(duration):
                link = super().get_link(point, 0, lattice, dagger=False)
                link = np.ascontiguousarray(link)
                loop = loop @ link
                super().up(point, 0)
        
            # Construct the loop in the spatial direction
            for space in range(length):
                link = super().get_link(point, space_direction, lattice, dagger=False)
                link = np.ascontiguousarray(link)
                loop = loop @ link
                super().up(point, space_direction)
        
            # Close the loop by moving backward in time
            for time_reverse in range(duration):
                super().down(point, 0)  
                link = super().get_link(point, 0, lattice, dagger=True) 
                link = np.ascontiguousarray(link)
                loop = loop @ link
        
            # Close the loop by moving backward in space
            for space_reverse in range(length):
                super().down(point, space_direction) 
                link = super().get_link(point, space_direction, lattice, dagger=True) 
                link = np.ascontiguousarray(link)
                loop = loop @ link
        
            # Accumulate the contribution of the Wilson loop (averaged over SU(3) traces)
            W_planar += (1/3) * np.real(np.trace(loop))
    
        return W_planar / 3 
    
    def planar_loop_over_lattice(self, lattice, matrices, length, duration, N_cf, N_cor):
        
        """
        Computes the planar Wilson loop over the entire lattice, averaging over configurations.

        Parameters:
            - lattice: The gauge field lattice.
            - matrices: The gauge group matrices (SU(3)).
            - length: Spatial extent of the Wilson loop.
            - duration: Temporal extent of the Wilson loop.
            - N_cf: Number of configurations.
            - N_cor: Number of updates (correlation steps) between configurations.

        Returns:
            - W_planar: An array of Wilson loop values averaged over the lattice.
        """
        
        W_planar = np.zeros(N_cf, dtype=np.float64)
        for alpha in range(N_cf):
            for skip in range(N_cor):
                super().lattice_update(lattice, matrices)
            
            if self.smearing:
                lattice = self.smearings(lattice, number_of_smears=4)
    
            for t in range(self.N):
                for x in range(self.N):
                    for y in range(self.N):
                        for z in range(self.N):
                            point = np.array([t, x, y, z])
                            W_planar[alpha] += self.planar_loops_potential(lattice, point, length, duration)
            
            print(W_planar[alpha] / (self.N)**(self.dim))
    
        return W_planar / (self.N)**(self.dim)
    
    def Wilson(self, lattice, Ms, r_max, t_min, t_max, N_cf, N_cor):
        
        """
        Computes the Wilson loop expectation values as a function of spatial separation r and time t.

        Parameters:
            - lattice: The gauge field lattice.
            - Ms: The gauge group matrices (SU(3)).
            - r_max: Maximum spatial separation.
            - t_min: Minimum temporal separation.
            - t_max: Maximum temporal separation.
            - N_cf: Number of configurations.
            - N_cor: Number of updates between configurations.

        Returns:
            - W_planar_r_t: Wilson loop expectation values for each (t, r).
            - W_planar_r_t_err: Standard deviation of the Wilson loop values.
        """
    
        W_planar_r_t = np.zeros((t_max, r_max))
        W_planar_r_t_err = np.zeros((t_max, r_max))
        
        for t in range(t_min, t_max):
            
            for r in range(1, r_max):
                
                W_r = self.planar_loop_over_lattice(lattice, Ms, r, t, N_cf, N_cor)
                W_planar_r_t[t, r] = np.sum(W_r) / len(W_r)
                W_planar_r_t_err[t, r] = np.std(W_r)
    
        return W_planar_r_t, W_planar_r_t_err

    def gauge_covariant_derivative(self, lattice, point, starting_direction):
        
        """
        Computes the gauge covariant derivative using smeared links.

        Parameters:
            - lattice: The gauge field lattice.
            - point: The starting lattice site.
            - starting_direction: The direction in which the derivative is computed.

        Returns:
            - smeared_link: The smeared link matrix.
        """
        
        link_up = super().get_link(point, starting_direction, lattice, dagger=False)
        link_up = np.ascontiguousarray(link_up)
        
        smeared_link = np.zeros((3, 3), dtype=np.complex128)
        for direction in range(1, self.dim):#don't smear with time links
            if direction != starting_direction: #don't smear in the same direction as the link being smeared
    
                link_right = super().get_link(point, direction, lattice, dagger=False)
                link_right = np.ascontiguousarray(link_right)
                super().up(point, direction)
                link_right_up = super().get_link(point, starting_direction, lattice, dagger=False)
                link_right_up = np.ascontiguousarray(link_right_up)
                super().up(point, starting_direction)
                super().down(point, direction)
                link_right_up_left = super().get_link(point, direction, lattice, dagger=True)
                link_right_up_left = np.ascontiguousarray(link_right_up_left)
    
                super().down(point, direction)
                link_left_up_right = super().get_link(point, direction, lattice, dagger=False)
                link_left_up_right = np.ascontiguousarray(link_left_up_right)
                super().down(point, starting_direction)
                link_left_up = super().get_link(point, starting_direction, lattice, dagger=False)
                link_left_up = np.ascontiguousarray(link_left_up)
                link_left = super().get_link(point, direction, lattice, dagger=True)
                link_left = np.ascontiguousarray(link_left)
                super().up(point, direction)
    
                loop_right = link_right @ link_right_up @ link_right_up_left
                loop_left = link_left @ link_left_up @ link_left_up_right
    
                smeared_link = smeared_link + (1/(self.u0 * self.a)**2)*(loop_right - 2 * (self.u0**2) * link_up + loop_left)
    
        return smeared_link
    
    def smear_lattice(self, lattice):
        
        """
        Applies a single iteration of smearing to the lattice gauge field.

        Smearing is applied only to spatial links (not temporal links), which helps in
        reducing short-distance fluctuations while preserving long-distance physics.

        Parameters:
            - lattice: The gauge field lattice.

        Returns:
            - lattice: The smeared lattice.
        """
    
        for t in range(self.N):
            for x in range(self.N):
                for y in range(self.N):
                    for z in range(self.N):
                        point = np.array([t, x, y, z])
                        for direction in range(1, self.dim):  #Avoids smearing on temporal links
                            smeared_link = lattice[t, x, y, z, direction] + self.smeared_eps * (self.a ** 2) * self.gauge_covariant_derivative(lattice, point, direction)
                        
                            #SU(3) projection process
                            U, S, Vh = np.linalg.svd(smeared_link)  # SVD
                            lattice[t, x, y, z, direction] = np.dot(U, Vh)  
                        
                            detU = np.linalg.det(lattice[t, x, y, z, direction])
                            lattice[t, x, y, z, direction] /= detU**(1/3)
        return lattice
    
    def smearings(self, lattice, number_of_smears):
        
        """
        Applies multiple smearing steps to the lattice.

        Parameters:
            - lattice: The gauge field lattice.
            - number_of_smears: The number of times smearing should be applied.

        Returns:
            - repeatedly_smeared_lattice: The lattice after multiple smearing iterations.
        """
    
        repeatedly_smeared_lattice = lattice.copy()
        
        for i in range(number_of_smears):
            repeatedly_smeared_lattice = self.smear_lattice(repeatedly_smeared_lattice)
            
        return repeatedly_smeared_lattice
    
    def run_simulation(self, lattice, update_Ms, N_cf, N_cor, t_max, r_max):
        
        """
        Runs the full lattice QCD simulation.

       Parameters:
           - lattice: The gauge field lattice.
           - update_Ms: Gauge update matrices.
           - N_cf: Number of configurations.
           - N_cor: Number of updates between measurements.
           - t_max: Maximum temporal extent.
           - r_max: Maximum spatial separation.

       Returns:
           - W_planar_r_t: Wilson loop results for different r and t.
           - W_planar_r_t_err: Errors in Wilson loop measurements.
           - radius: Array of spatial separations.
       """
        
        super().thermalize_lattice(lattice, update_Ms, N_cor)
    
        print("Computing loops...")
        W_planar_r_t, W_planar_r_t_err = self.Wilson(lattice, update_Ms, r_max, 1, t_max, N_cf, N_cor)
        print("Loops computed.")
        
        radius = range(1, r_max)
    
        return W_planar_r_t, W_planar_r_t_err, radius
    
    def analyze_results(self, WL_avg, WL_avg_err, radius):
        
        """
        Computes the static quark potential V(r) from Wilson loop averages.

        The static quark potential is extracted from the asymptotic behavior of the 
        Wilson loops at large time separations.

        Parameters:
            - WL_avg: Average Wilson loop values.
            - WL_avg_err: Errors in the Wilson loop values.
            - radius: Array of spatial separations.

        Returns:
            - potential: Extracted static quark potential V(r).
            - potential_err_log: Logarithmic errors in potential values.
        """
    
        r_max = WL_avg.shape[1]
        t_max = WL_avg.shape[0]

        potential = np.zeros(r_max - 1)
        potential_err_ratio = np.zeros(r_max - 1)
        potential_err_log = np.zeros(r_max - 1)
        
        if self.smearing:
            for r in radius:
                potential[r-1] = np.log(np.abs(WL_avg[t_max-3, r]/WL_avg[t_max-2, r]))
                potential_err_ratio[r-1] = (((WL_avg_err[t_max-3, r]/WL_avg[t_max-3, r])**2 + (WL_avg_err[t_max-2, r]/WL_avg[t_max-2, r])**2)**(1/2))*potential[r-1]
                potential_err_log[r-1] = potential_err_ratio[r-1]/potential[r-1]
        else:
            for r in radius:
                potential[r-1] = np.log(np.abs(WL_avg[t_max-2, r]/WL_avg[t_max-1, r]))
                potential_err_ratio[r-1] = (((WL_avg_err[t_max-2, r]/WL_avg[t_max-2, r])**2 + (WL_avg_err[t_max-1, r]/WL_avg[t_max-1, r])**2)**(1/2))*potential[r-1]
                potential_err_log[r-1] = potential_err_ratio[r-1]/potential[r-1]
        
        return potential, potential_err_log
    
    def plot_static_potential(self, radius, potential, potential_err_log, filename):
        
        """
        Plots the static quark potential V(r) and fits it using the Cornell potential.

        Parameters:
            - radius: Array of spatial separations.
            - potential: Computed static potential values.
            - potential_err_log: Error estimates for V(r).
            - filename: File name to save the plot.
        """
    
    
        plt.errorbar(radius, potential, potential_err_log, fmt='o', label='simulation')
        plt.xlabel('r/a')
        plt.ylabel('aV(r)')
        plt.axis([0.1, 4.5, -0.5, 2.5])
        plt.title('Static quark potential')
        
        popt, pcov = optimize.curve_fit(self.V_exact, radius, potential, sigma=potential_err_log , p0=(0.5, 0.5, 0.5), bounds=([0, 0.1, 0],[1, 1, 1]))
        perr = np.sqrt(np.diag(pcov))
        radius = np.linspace(0.1, 4.5, 100)
        fit = [popt[0]*x -popt[1]/x + popt[2] for x in radius]
        plt.plot(radius, fit, label='fit')
        plt.plot(radius, np.zeros(len(radius)), linestyle='dashed', color='black')
        plt.legend()
        
        print('sigma =' + str(popt[0]) + f' plus/minus {perr[0]}')
        print('b =' + str(popt[1]) + f' plus/minus {perr[1]}')
        print('c =' + str(popt[2]) + f' plus/minus {perr[2]}')
        
        plt.show()
