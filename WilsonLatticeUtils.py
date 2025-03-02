import numpy as np
import math as math
from tqdm import tqdm

class WilsonLatticeUtils:
    def __init__(self, N, dim, N_matrix, eps, beta, beta_imp, u0, use_improved = False):
        
        """
        Initialize the Wilson lattice utility class.
        
        Parameters:
        N: int - Lattice size
        dim: int - Number of dimensions
        N_matrix: int - Number of matrices in the SU(3) pool
        eps: float - Small parameter for SU(3) matrix exponentiation
        beta: float - Coupling constant for the unimproved action
        beta_imp: float - Coupling constant for the improved action
        u0: float - Tadpole improvement factor
        use_improved: bool - Whether to use the improved action
        """
        
        self.N = N  # Lattice size
        self.dim = dim  # Dimensions
        self.N_matrix = N_matrix  # Number of matrices
        self.eps = eps  
        self.beta = beta
        self.beta_imp = beta_imp
        self.u0 = u0
        self.use_improved = use_improved


    def SU3(self, steps=30):
        
        """
        Generate a random SU(3) matrix using exponentiation of a random Hermitian matrix.
        
        Parameters:
        steps: int - Number of terms in the matrix exponential series
        
        Returns:
        U: ndarray - SU(3) matrix
        """
        
        ones = (np.random.rand(3, 3)*2 - 1) + 1j * (np.random.rand(3, 3)*2 - 1)
        H = (1/2)*(ones + np.conj(ones.T))
        U = np.zeros((3, 3), np.complex128)
        
        for i in range(steps):
            U += ((1j*self.eps)**i / math.factorial(i)) * np.linalg.matrix_power(H, i)
            
        return U / (np.linalg.det(U))**(1/3)


    def generate_matrices_pool(self):
        
        """
        Generate a pool of SU(3) matrices and their Hermitian conjugates.
        
        Returns:
        MatList: ndarray - Array of SU(3) matrices and their conjugates
        """
        
        MatList = np.zeros((self.N_matrix * 2, 3, 3), np.complex128)
        for i in range(self.N_matrix):
            MatList[i] = self.SU3()
            MatList[self.N_matrix + i] = MatList[i].conj().T
        return MatList
 
    
    def select_matrix(self, rand):
        
        """
        Select a matrix from the generated pool based on an index.
        
        Parameters:
        rand: int - Index of the matrix to select
        
        Returns:
        SU(3) matrix from the pool
        """
        
        pool = self.generate_matrices_pool()  # Genera il pool di matrici
        return pool[rand]


    def create_lattice(self):
        
        """
        Create an initial lattice where all links are identity matrices.
        
        Returns:
        lattice: ndarray - SU(3) lattice initialized with identity matrices
        """
        
        lattice = np.empty((self.N, self.N, self.N, self.N, self.dim, 3, 3), np.complex128)
        for t in range(self.N):
            for x in range(self.N):
                for y in range(self.N):
                    for z in range(self.N):
                        for mu in range(self.dim):
                            lattice[t, x, y, z, mu] = np.identity(3, np.complex128)
        print("Lattice creation finished")
        return lattice
  
    
    def get_link(self, point, direction, lattice, dagger):
        
        """
        Retrieve a link from the lattice in a given direction, optionally taking its Hermitian conjugate.
        
        Parameters:
        point: list - Lattice coordinates [t, x, y, z]
        direction: int - Direction index
        lattice: ndarray - Lattice field
        dagger: bool - Whether to return the Hermitian conjugate
        
        Returns:
        SU(3) link matrix
        """
        
        if dagger == False:
            return lattice[point[0], point[1], point[2], point[3], direction]
        elif dagger == True:
            return lattice[point[0], point[1], point[2], point[3], direction].conj().T


    def up(self, coordinate, direction):
        
        """
        Move the coordinate one step up in the given direction, applying periodic boundary conditions.
        
        Parameters:
        coordinate: list - Lattice coordinates [t, x, y, z]
        direction: int - Direction index
        
        Returns:
        Updated coordinate with periodic boundary conditions applied
        """
        
        coordinate[direction] = (coordinate[direction] + 1)% self.N
        return coordinate


    def down(self, coordinate, direction):
        
        """
        Move the coordinate one step down in the given direction, applying periodic boundary conditions.
        
        Parameters:
        coordinate: list - Lattice coordinates [t, x, y, z]
        direction: int - Direction index
        
        Returns:
        Updated coordinate with periodic boundary conditions applied
        """
        
        coordinate[direction] = (coordinate[direction] - 1)% self.N
        return coordinate
    
    def planar_loops(self, lattice, point, width, heigth):
        
        """
        Compute Wilson loops of given width and height at a specific lattice point.
        
        Parameters:
        lattice: ndarray - Lattice field
        point: list - Lattice coordinates [t, x, y, z]
        width: int - Loop width
        height: int - Loop height
        
        Returns:
        w_planar: float - Average Wilson loop value over all directions
        """
        
        w_planar=0
        for direction_1 in range(self.dim):
            for direction_2 in range(direction_1):
                loop = np.identity(3, np.complex128)
                for h in range(heigth):
                    link = self.get_link(point, direction_1, lattice, dagger=False)
                    link = np.ascontiguousarray(link)
                    loop = loop @ link
                    self.up(point, direction_1)
                for w in range(width):
                    link = self.get_link(point, direction_2, lattice, dagger=False)
                    link = np.ascontiguousarray(link)
                    loop = loop @ link
                    self.up(point, direction_2)
                for h_reverse in range(heigth):
                    self.down(point, direction_1)
                    link = self.get_link(point, direction_1, lattice, dagger=True)
                    link = np.ascontiguousarray(link)
                    loop = loop @ link
                for w_reverse in range(width):
                    self.down(point, direction_2)
                    link = self.get_link(point, direction_2, lattice, dagger=True)
                    link = np.ascontiguousarray(link)
                    loop = loop @ link
                w_planar += (1/3)*np.real(np.trace(loop))
        return w_planar/6


    def wilson_over_lattice(self, lattice, matrices, width, height, Ncf, Ncor):
        
        """
        Computes the Wilson loop over the lattice for a given set of configurations.

        Parameters:
            - lattice: The gauge field lattice.
            - matrices: Link matrices used for the update step.
            - width: The spatial extent of the Wilson loop.
            - height: The temporal extent of the Wilson loop.
            - Ncf: Number of configurations to sample.
            - Ncor: Number of updates (decorrelation steps) between configurations.

        Returns:
            - W_plaquettes: Array of Wilson loop values normalized over the lattice volume.
        """
    
        W_plaquettes = np.zeros(Ncf, dtype=np.float64)
        for alpha in range(Ncf):
            for skip in range(Ncor):
                self.lattice_update(lattice, matrices)
            for t in range(self.N):
                for x in range(self.N):
                    for y in range(self.N):
                        for z in range(self.N):
                            point = np.array([t, x, y, z])
                            W_plaquettes[alpha] += self.planar_loops(lattice, point, width, height)
            print(W_plaquettes[alpha] / (self.N)**(self.dim))
        return W_plaquettes/(self.N)**(self.dim)

            
    def plaquette(self, lattice, point, starting_direction):
        
        """
        Computes the plaquette at a given lattice point and direction.

        Parameters:
            - lattice: The gauge field lattice.
            - point: The coordinates of the lattice site.
            - starting_direction: The initial direction for constructing the plaquette.

        Returns:
            - gamma: The sum of the two traced plaquettes (clockwise and counterclockwise).
        """

        point_clockwise = point.copy()
        point_anticlockwise = point.copy()
    
        self.up(point_clockwise, starting_direction)                           #move up initial link
        self.up(point_anticlockwise, starting_direction)                           #move up initial link
    
        clockwise = np.zeros((3, 3), np.complex128)
        anticlockwise = np.zeros((3, 3), np.complex128)
        gamma = np.zeros((3, 3), np.complex128)
        for direction in range(self.dim):                                    #cycle over directions other than the starting_direction
            if direction != starting_direction:
                link_right = self.get_link(point_clockwise, direction, lattice, dagger=False)                  #take link pointing "right"
                link_right = np.ascontiguousarray(link_right)
                self.up(point_clockwise, direction)                                                             #move "up"
                self.down(point_clockwise, starting_direction)                                                  #move "down"
                link_right_down = self.get_link(point_clockwise, starting_direction, lattice, dagger=True)     #take link pointing "down"
                link_right_down = np.ascontiguousarray(link_right_down)
                self.down(point_clockwise, direction)                                                           #move "left"
                link_right_down_left = self.get_link(point_clockwise, direction, lattice, dagger=True)         #take link pointing "left"
                link_right_down_left = np.ascontiguousarray(link_right_down_left)
                self.up(point_clockwise, starting_direction)                                                    #back to initial position
    
                self.down(point_anticlockwise, direction)
                link_left = self.get_link(point_anticlockwise, direction, lattice, dagger=True)
                link_left = np.ascontiguousarray(link_left)
                self.down(point_anticlockwise, starting_direction)
                link_left_down = self.get_link(point_anticlockwise, starting_direction, lattice, dagger=True)
                link_left_down = np.ascontiguousarray(link_left_down)
                link_left_down_right = self.get_link(point_anticlockwise, direction, lattice, dagger=False)
                link_left_down_right = np.ascontiguousarray(link_left_down_right)
                self.up(point_anticlockwise, direction)
                self.up(point_anticlockwise, starting_direction)
    
                clockwise += (link_right @ link_right_down) @ link_right_down_left
                anticlockwise += (link_left @ link_left_down) @ link_left_down_right
    
        gamma = clockwise + anticlockwise
        
        return gamma

    def plaquette_improved(self, lattice, point, starting_direction):
        
        """
        Computes an improved version of the plaquette, which includes contributions 
        from larger loops to reduce discretization errors.

        Parameters:
            - lattice: The gauge field lattice.
            - point: The coordinates of the lattice site.
            - starting_direction: The initial direction for constructing the improved plaquette.

        Returns:
            - gamma: The sum of contributions from different orientations of the improved plaquette.
        """

        
        point_clockwise_vertical_down = point.copy()
        point_anticlockwise_vertical_down = point.copy()
        point_clockwise_vertical_up = point.copy()
        point_anticlockwise_vertical_up = point.copy()
        point_clockwise_horizontal = point.copy()
        point_anticlockwise_horizontal = point.copy()
    
        self.up(point_clockwise_vertical_down, starting_direction)                           #move up initial link
        self.up(point_clockwise_vertical_up, starting_direction)                           #move up initial link
        self.up(point_anticlockwise_vertical_down, starting_direction)                           #move up initial link
        self.up(point_anticlockwise_vertical_up, starting_direction)                           #move up initial link
        self.up(point_clockwise_horizontal, starting_direction)                           #move up initial link
        self.up(point_anticlockwise_horizontal, starting_direction)                           #move up initial link
    
        clockwise_vertical_up = np.zeros((3, 3), np.complex128)
        clockwise_vertical_down = np.zeros((3, 3), np.complex128)
        anticlockwise_vertical_up = np.zeros((3, 3), np.complex128)
        anticlockwise_vertical_down = np.zeros((3, 3), np.complex128)
        clockwise_horizonal = np.zeros((3, 3), np.complex128)
        anticlockwise_horizontal = np.zeros((3, 3), np.complex128)
        
        gamma = np.zeros((3, 3), np.complex128)
        for direction in range(self.dim):                                    #cycle over directions other than the starting_direction
            if direction != starting_direction:
                
                link_up = self.get_link(point_clockwise_vertical_up, starting_direction, lattice, dagger=False)                  #take link pointing "right"
                link_up = np.ascontiguousarray(link_up)
                
                #clockwise vertical up
                self.up(point_clockwise_vertical_up, starting_direction)
                link_up_right = self.get_link(point_clockwise_vertical_up, direction, lattice, dagger=False)
                link_up_right = np.ascontiguousarray(link_up_right)
                self.up(point_clockwise_vertical_up, direction)                                    #move "right"
                self.down(point_clockwise_vertical_up, starting_direction)
                link_up_right_down = self.get_link(point_clockwise_vertical_up, starting_direction, lattice, dagger=True)    #take link moving "down"
                link_up_right_down = np.ascontiguousarray(link_up_right_down)
                self.down(point_clockwise_vertical_up, starting_direction)                         #move "down"
                link_up_right_down_down = self.get_link(point_clockwise_vertical_up, starting_direction, lattice, dagger=True)    #take link moving "down"
                link_up_right_down_down = np.ascontiguousarray(link_up_right_down_down)
                self.down(point_clockwise_vertical_up, direction)
                link_up_right_down_down_left = self.get_link(point_clockwise_vertical_up, direction, lattice, dagger=True)
                link_up_right_down_down_left = np.ascontiguousarray(link_up_right_down_down_left)
                self.up(point_clockwise_vertical_up, starting_direction)
    
                #anticlockwise vertical up
                self.up(point_anticlockwise_vertical_up, starting_direction)
                self.down(point_anticlockwise_vertical_up, direction)
                link_up_left = self.get_link(point_anticlockwise_vertical_up, direction, lattice, dagger=True)
                link_up_left = np.ascontiguousarray(link_up_left)
                self.down(point_anticlockwise_vertical_up, starting_direction)
                link_up_left_down = self.get_link(point_anticlockwise_vertical_up, starting_direction, lattice, dagger=True)    #take link moving "down"
                link_up_left_down = np.ascontiguousarray(link_up_left_down)
                self.down(point_anticlockwise_vertical_up, starting_direction)                         #move "down"
                link_up_left_down_down = self.get_link(point_anticlockwise_vertical_up, starting_direction, lattice, dagger=True)    #take link moving "down"
                link_up_left_down_down = np.ascontiguousarray(link_up_left_down_down)
                link_up_left_down_down_right = self.get_link(point_anticlockwise_vertical_up, direction, lattice, dagger=False)
                link_up_left_down_down_right = np.ascontiguousarray(link_up_left_down_down_right)
                self.up(point_anticlockwise_vertical_up, direction)
                self.up(point_anticlockwise_vertical_up, starting_direction)
   
    
                link_right = self.get_link(point_clockwise_vertical_down, direction, lattice, dagger=False)                  #take link pointing "right"
                link_right = np.ascontiguousarray(link_right)
    
                #clockwise vertical down
                self.up(point_clockwise_vertical_down, direction)
                self.down(point_clockwise_vertical_down, starting_direction)                                    #move "right"
                link_right_down = self.get_link(point_clockwise_vertical_down, starting_direction, lattice, dagger=True)    #take link moving "down"
                link_right_down = np.ascontiguousarray(link_right_down)
                self.down(point_clockwise_vertical_down, starting_direction)                         #move "down"
                link_right_down_down = self.get_link(point_clockwise_vertical_down, starting_direction, lattice, dagger=True)             #take link moving "left"
                link_right_down_down = np.ascontiguousarray(link_right_down_down)
                self.down(point_clockwise_vertical_down, direction)
                link_right_down_down_left = self.get_link(point_clockwise_vertical_down, direction, lattice, dagger=True)
                link_right_down_down_left = np.ascontiguousarray(link_right_down_down_left)
                link_right_down_down_left_up = self.get_link(point_clockwise_vertical_down, starting_direction, lattice, dagger=False)
                link_right_down_down_left_up = np.ascontiguousarray(link_right_down_down_left_up)
                self.up(point_clockwise_vertical_down, starting_direction)
                self.up(point_clockwise_vertical_down, starting_direction)
    
                #clockwise horizonal
                self.up(point_clockwise_horizontal, direction)
                link_right_right = self.get_link(point_clockwise_horizontal, direction, lattice, dagger=False)                  #take link pointing "right"
                link_right_right = np.ascontiguousarray(link_right_right)
                self.up(point_clockwise_horizontal, direction)
                self.down(point_clockwise_horizontal, starting_direction)                                    #move "right"
                link_right_right_down = self.get_link(point_clockwise_horizontal, starting_direction, lattice, dagger=True)    #take link moving "down"
                link_right_right_down = np.ascontiguousarray(link_right_right_down)
                self.down(point_clockwise_horizontal, direction)                         #move "down"
                link_right_right_down_left = self.get_link(point_clockwise_horizontal, direction, lattice, dagger=True)             #take link moving "left"
                link_right_right_down_left = np.ascontiguousarray(link_right_right_down_left)
                self.down(point_clockwise_horizontal, direction)                         #move "down"
                link_right_right_down_left_left = self.get_link(point_clockwise_horizontal, direction, lattice, dagger=True)             #take link moving "left"
                link_right_right_down_left_left = np.ascontiguousarray(link_right_right_down_left_left)
                self.up(point_clockwise_horizontal, starting_direction)


                self.down(point_anticlockwise_vertical_down, direction)
                self.down(point_anticlockwise_horizontal, direction)
                link_left = self.get_link(point_anticlockwise_vertical_down, direction, lattice, dagger=True)
                link_left = np.ascontiguousarray(link_left)
    
                #anticlockwise vertical down
                self.down(point_anticlockwise_vertical_down, starting_direction)
                link_left_down = self.get_link(point_anticlockwise_vertical_down, starting_direction, lattice, dagger=True)
                link_left_down = np.ascontiguousarray(link_left_down)
                self.down(point_anticlockwise_vertical_down, starting_direction)
                link_left_down_down = self.get_link(point_anticlockwise_vertical_down, starting_direction, lattice, dagger=True)
                link_left_down_down = np.ascontiguousarray(link_left_down_down)
                link_left_down_down_right = self.get_link(point_anticlockwise_vertical_down, direction, lattice, dagger=False)
                link_left_down_down_right = np.ascontiguousarray(link_left_down_down_right)
                self.up(point_anticlockwise_vertical_down, direction)
                link_left_down_down_right_up = self.get_link(point_anticlockwise_vertical_down, starting_direction, lattice, dagger=False)
                link_left_down_down_right_up = np.ascontiguousarray(link_left_down_down_right_up)
                self.up(point_anticlockwise_vertical_down, starting_direction)
                self.up(point_anticlockwise_vertical_down, starting_direction)
    
                #anticlockwise horizontal
                self.down(point_anticlockwise_horizontal, direction)
                link_left_left = self.get_link(point_anticlockwise_horizontal, direction, lattice, dagger=True)
                link_left_left = np.ascontiguousarray(link_left_left)
                self.down(point_anticlockwise_horizontal, starting_direction)
                link_left_left_down = self.get_link(point_anticlockwise_horizontal, starting_direction, lattice, dagger=True)
                link_left_left_down = np.ascontiguousarray(link_left_left_down)
                link_left_left_down_right = self.get_link(point_anticlockwise_horizontal, direction, lattice, dagger=False)
                link_left_left_down_right = np.ascontiguousarray(link_left_left_down_right)
                self.up(point_anticlockwise_horizontal, direction)
                link_left_left_down_right_right = self.get_link(point_anticlockwise_horizontal, direction, lattice, dagger=False)
                link_left_left_down_right_right = np.ascontiguousarray(link_left_left_down_right_right)
                self.up(point_anticlockwise_horizontal, direction)
                self.up(point_anticlockwise_horizontal, starting_direction)

    
                clockwise_vertical_up += link_up @ link_up_right @ link_up_right_down @  link_up_right_down_down @ link_up_right_down_down_left
                clockwise_vertical_down += link_right @ link_right_down @ link_right_down_down @ link_right_down_down_left @ link_right_down_down_left_up
                anticlockwise_vertical_up += link_up @ link_up_left @ link_up_left_down @ link_up_left_down_down @ link_up_left_down_down_right
                anticlockwise_vertical_down += link_left @ link_left_down @ link_left_down_down @ link_left_down_down_right @ link_left_down_down_right_up
                clockwise_horizonal += link_right @ link_right_right @ link_right_right_down @ link_right_right_down_left @ link_right_right_down_left_left
                anticlockwise_horizontal += link_left @ link_left_left @ link_left_left_down @ link_left_left_down_right @ link_left_left_down_right_right
    
        gamma = clockwise_vertical_up + clockwise_vertical_down + anticlockwise_vertical_up + anticlockwise_vertical_down + clockwise_horizonal + anticlockwise_horizontal
        
        return gamma

    def lattice_update(self, lattice, Xs):
        
        """
        Performs a lattice update using the Metropolis algorithm, supporting both 
        improved and unimproved actions.

        Parameters:
            - lattice: The gauge field lattice.
            - Xs: A set of random SU(3) matrices used for updates.
        """

        n_hits = 10  # Number of Metropolis hits per link update

        for t in range(self.N):
            for x in range(self.N):
                for y in range(self.N):
                    for z in range(self.N):
                        for mu in range(self.dim):  # Iterate over lattice dimensions
                            coord = [t, x, y, z]
                            
                            # Compute fundamental plaquette term
                            gamma = self.plaquette(lattice, coord, mu)
                            
                            # Compute improved rectangle term if needed
                            if self.use_improved:
                                gamma_imp = self.plaquette_improved(lattice, coord, mu)

                            for _ in range(n_hits):  # Perform Metropolis hits
                                xi = np.random.randint(2, self.N_matrix * 2)
                                U_old = lattice[coord[0], coord[1], coord[2], coord[3], mu]
                                U_new = np.dot(Xs[xi], U_old)

                                if self.use_improved:
                                    dS = -self.beta_imp / 3 * (
                                        (5 / (3 * self.u0**4)) * np.real(np.trace(np.dot(U_new - U_old, gamma)))
                                        - (1 / (12 * self.u0**6)) * np.real(np.trace(np.dot(U_new - U_old, gamma_imp)))
                                        )
                                else:
                                    dS = -(self.beta / 3) * np.real(np.trace(np.dot(U_new - U_old, gamma)))

                                # Accept or reject the update
                                if dS < 0 or np.exp(-dS) > np.random.uniform(0, 1):
                                    lattice[coord[0], coord[1], coord[2], coord[3], mu] = U_new
    
    def thermalize_lattice(self, lattice, update_Ms, N_cor):
        """
        Thermalizes the lattice by performing N_cor update steps.
    
        Args:
            lattice (ndarray): The lattice to thermalize.
            update_Ms (ndarray): Update matrices for the lattice.
            N_cor (int): Number of thermalization steps.
        """
        for _ in tqdm(range(N_cor), desc="Thermalization"):
            self.lattice_update(lattice, update_Ms)
        print("\n Thermalization complete")
        
