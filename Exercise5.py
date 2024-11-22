import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Lattice parameters
L = 50  # Lattice size (spatial points)
N_steps = 5000  # Number of Monte Carlo steps
beta = 1.0  # Coupling constant set to 1
max_time = 20  # Limit the time slice to the first 10 time steps (t=0 to t=9)

# Function to initialize a scalar field
def initialize_scalar_field(L):
    return np.random.uniform(-1, 1, size=(L,))

# Function to compute the action
def compute_action(field, beta):
    S = 0
    for x in range(L):
        S += (field[x] - field[(x + 1) % L]) ** 2
    return beta * S

# Metropolis update for the scalar field
def metropolis_update(field, beta):
    for _ in range(L):
        x = np.random.randint(0, L)
        delta_phi = np.random.uniform(-0.5, 0.5)  # Small random change
        new_field = field.copy()
        new_field[x] += delta_phi

        S_old = compute_action(field, beta)
        S_new = compute_action(new_field, beta)
        delta_S = S_new - S_old

        # Handle overflow in np.exp
        if delta_S > 700:
            prob = 0
        else:
            prob = np.exp(-delta_S)

        if np.random.rand() < prob:
            field[x] = new_field[x]
    return field

# Function to compute the two-point correlation function
def compute_correlation_function(field, beta, N_steps):
    correlators = np.zeros(L)
    field_configs = []

    # Monte Carlo simulation
    for step in range(N_steps):
        field = metropolis_update(field, beta)
        field_configs.append(field.copy())

    # Compute C(t) as an average over configurations
    field_configs = np.array(field_configs)
    for t in range(L):
        correlators[t] = np.mean(
            field_configs[:, 0] * field_configs[:, t]
        )  # Spatial average

    return correlators

# Define an exponential decay with a constant term for fitting
def exponential_with_offset(t, A, m, B):
    return A * np.exp(-m * t) + B

# Main simulation
field = initialize_scalar_field(L)
correlation_function = compute_correlation_function(field, beta, N_steps)

# Fit the mass gap and constant term
t_values = np.arange(L)
try:
    popt, _ = curve_fit(exponential_with_offset, t_values, correlation_function, p0=[1.0, 1.0, 0.0])
    mass_gap = popt[1]
    Z_2 = 1 / popt[0]  # Estimate the wavefunction renormalization constant (Z_2)
    print(f"Estimated mass gap: {mass_gap:.3f}")
    print(f"Estimated wavefunction renormalization constant (Z_2): {Z_2:.3f}")
except RuntimeError:
    print("Curve fitting failed.")

# Limit the time slice to the first 'max_time' time steps
t_values_limited = t_values[:max_time]
correlation_function_limited = correlation_function[:max_time]

# Plotting the correlation function (limited time slices)
plt.figure(figsize=(10, 6))
plt.plot(t_values_limited, correlation_function_limited, 'o-', label="Monte Carlo Data")
plt.title(f"Two-Point Correlation Function (First {max_time} Time Slices)")
plt.xlabel("t")
plt.ylabel("C(t)")
plt.legend()
plt.grid()
plt.show()

# Filter out zero or negative values for the log-scale plot
valid_indices = correlation_function_limited > 0
log_t_values = t_values_limited[valid_indices]
log_correlation_function = correlation_function_limited[valid_indices]

# Plot in log-scale for the limited time slices
plt.figure(figsize=(10, 6))
plt.semilogy(log_t_values, log_correlation_function, 'o-', label="Monte Carlo Data (Log-Scale)")
plt.title(f"Two-Point Correlation Function (Log Scale, First {max_time} Time Slices)")
plt.xlabel("t")
plt.ylabel("log(C(t))")
plt.legend()
plt.grid()
plt.show()
