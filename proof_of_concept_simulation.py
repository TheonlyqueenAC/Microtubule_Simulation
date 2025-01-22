import matplotlib.pyplot as plt
import numpy as np

from MicrotubuleSimulation.fibonacci_simulation_refactored import (
    generate_fibonacci_sequence,
    normalize_fibonacci_sequence,
    initialize_wave_function,
    evolve_wave_function
)

# Define constants
hbar = 1.0  # Reduced Planck's constant (in natural units for simplicity)
m = 1.0  # Mass of the particle (also in natural units)
L = 10.0  # Length of the domain
N = 100  # Number of grid points
dx = L / N  # Spatial step size
dt = 0.01  # Time step

# Define spatial grid points
x = np.linspace(0, L, N)

# Define potentials
v_constant = np.ones_like(x) * 0.5  # Uniform constant potential
v_quadratic = 0.1 * (x - L / 2) ** 2  # Quadratic potential

# Generate Fibonacci sequence and normalize it
fib_sequence = generate_fibonacci_sequence(N)
fib_ratios = normalize_fibonacci_sequence(fib_sequence, L)
v_fibonacci = fib_ratios

# Debugging: Validate potentials
print("Fibonacci Ratios:", fib_ratios)
print("v_fibonacci shape:", v_fibonacci.shape)
print("v_constant shape:", v_constant.shape)
print("v_quadratic shape:", v_quadratic.shape)

# Check grid consistency
assert len(x) == len(v_fibonacci), "Mismatch between spatial grid and Fibonacci potential"

# Initialize wave function
center = L / 2
width = 1.0
psi_global = initialize_wave_function(x, center, width)
# Generate Fibonacci sequence and normalize it
fib_sequence = generate_fibonacci_sequence(N)
fib_ratios = normalize_fibonacci_sequence(fib_sequence, L)



# Define potentials
v_fibonacci = fib_ratios  # Fibonacci potential
v_constant = np.ones_like(x) * 0.5  # Uniform constant potential
v_quadratic = 0.1 * (x - L / 2) ** 2  # Quadratic potential

# Debugging: Validate potentials
print("Fibonacci Ratios:", fib_ratios)
print("v_fibonacci shape:", v_fibonacci.shape)
print("v_constant shape:", v_constant.shape)
print("v_quadratic shape:", v_quadratic.shape)

# Check grid consistency
assert len(x) == len(v_fibonacci), "Mismatch between spatial grid and Fibonacci potential"

# Initialize wave function
center = L / 2
width = 1.0
psi_global = initialize_wave_function(x, center, width)


# Helper Functions
def track_variance(psi, potential, dx, time_steps):
    """Track variance for the given potential."""
    var_list = []
    psi_current = psi.copy()
    for _ in range(time_steps):
        psi_current = evolve_wave_function(psi_current, potential, dx, dt)
        com = np.sum(x * np.abs(psi_current)**2) * dx
        var = np.sum((x - com)**2 * np.abs(psi_current)**2) * dx
        var_list.append(var)
    return var_list


def compute_energies(psi_current, v_fibonacci, dx, hbar, m):
    """
    Compute kinetic and potential energy from a given wavefunction and potential.

    Parameters:
        psi_current (np.ndarray): Current wavefunction values (complex array).
        v_fibonacci (np.ndarray): Potential function values.
        dx (float): Spatial step size.
        hbar (float): Reduced Planck's constant.
        m (float): Mass of the particle.

    Returns:
        float: Kinetic energy.
        float: Potential energy.
    """
    # Compute the magnitude of the wavefunction
    magnitude = np.abs(psi_current)
    grad_magnitude = np.gradient(magnitude, dx)

    # Compute the kinetic energy
    squared_gradient = grad_magnitude ** 2
    kinetic_energy = 0.5 * hbar ** 2 / m * np.sum(squared_gradient) * dx

    # Compute the potential energy
    potential_energy = np.sum(v_fibonacci * magnitude ** 2) * dx

    # Return the computed energies
    return kinetic_energy, potential_energy

def plot_variance(time_steps, dt, var_lists, labels, title):

    """Plot variance comparison."""
    plt.figure(figsize=(12, 6))
    for var_list, label in zip(var_lists, labels):
        plt.plot(np.arange(time_steps) * dt, var_list, label=label)
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.title(title)
    plt.legend()
    plt.show()
