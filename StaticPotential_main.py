from StaticPotentialUtils import StaticPotentialUtils
from WilsonLatticeUtils import WilsonLatticeUtils

#Initial parameters
a = 0.25
N = 8
dim = 4
eps = 0.24
u0 = 0.797
beta = 5.5
beta_imp = 1.719
r_max = 5  # max r + 1
t_max = 5  # max t + 1
N_matrix = 100
N_cf = 10
N_cor = 50

# Create an instance of WilsonLatticeUtils
wilson_utils = WilsonLatticeUtils(N, dim, N_matrix, eps, beta, beta_imp, u0, use_improved = True)

# Use the instance to initialize StaticPotentialUtils
static_potential = StaticPotentialUtils(
    N = wilson_utils.N,
    dim = wilson_utils.dim,
    N_matrix = wilson_utils.N_matrix,
    eps = wilson_utils.eps,
    beta = wilson_utils.beta,
    beta_imp = wilson_utils.beta_imp,
    u0 = wilson_utils.u0,
    use_improved = wilson_utils.use_improved,
    a = a
)

lattice = wilson_utils.create_lattice() 
update_Ms = wilson_utils.generate_matrices_pool()

# Thermalization and Simulation
WL_avg, WL_avg_err, radius = static_potential.run_simulation(
    lattice = lattice,
    update_Ms = update_Ms,
    N_cf = N_cf,
    N_cor = N_cor,
    t_max = t_max,
    r_max = r_max,
)

potential, potential_err_log = static_potential.analyze_results(WL_avg, WL_avg_err, radius)

static_potential.plot_static_potential(radius, potential, potential_err_log, 'Vqcd_notimproved.pdf')