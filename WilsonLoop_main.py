from WilsonLatticeUtils import WilsonLatticeUtils

# Parameters
N = 8
dim = 4
N_matrix = 100
eps = 0.24
beta = 5.5
beta_imp = 1.719
u0 = 0.797
N_cf = 5
N_cor = 50
use_improved = False

# Initialize utility
utils = WilsonLatticeUtils(N, dim, N_matrix, eps, beta, beta_imp, u0, use_improved)

# Main simulation for standard plaquette
lattice_standard = utils.create_lattice()
update_Ms = utils.generate_matrices_pool()

print("\n Starting thermalization")
utils.thermalize_lattice(lattice_standard, update_Ms, N_cor)

width = 1
heigth = 2
loop = utils.wilson_over_lattice(lattice_standard, update_Ms, width, heigth, N_cf, N_cor)