from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fibonacci_simulation_refactored import (
    generate_fibonacci_sequence,
    normalize_fibonacci_sequence,
    initialize_wave_function,
    evolve_wave_function
)

# Constants
# Define constants
hbar = 1.0  # Reduced Planck constant (in natural units for simplicity)
m = 1.0
L = 10.0
N = 100
dx = L / N
dt = 0.01
time_steps = 300

# Spatial grid points
x = np.linspace(0, L, N)

# Generate and normalize Fibonacci potential
fib_sequence = generate_fibonacci_sequence(N)
fib_ratios = normalize_fibonacci_sequence(fib_sequence, L)
v_fibonacci = fib_ratios
v_constant = np.ones_like(x) * 0.5
v_quadratic = 0.1 * (x - L / 2)**2

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


def compute_energy(psi, potential, dx):
    """Compute kinetic and potential energy."""
    # Calculate gradient for real and imaginary parts separately
    grad_real = np.gradient(psi_current.real, dx)
    grad_imag = np.gradient(psi_current.imag, dx)

    # Combine gradients for the full kinetic energy density
    squared_gradient = grad_real ** 2 + grad_imag ** 2
    assert isinstance(dx, object)
    kinetic_energy = 0.5 * hbar ** 2 / m * np.sum(squared_gradient) * dx
    potential_energy = np.sum(np.abs(psi)**2 * potential) * dx
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

# Energy Analysis for Fibonacci Potential

# Initialize energy lists
kinetic_energy_list_fibonacci = []
potential_energy_list_fibonacci = []
total_energy_list_fibonacci = []
TOTAL_ENERGY_RESULTS_INITIAL = []

# Reinitialize Fibonacci wave function
psi_current = psi_global.copy()

for _ in range(time_steps):
    # Evolve the wave function
    psi_current = evolve_wave_function(psi_current, v_fibonacci, dx, dt)

    # Compute gradients for the real and imaginary parts
    grad_real = np.gradient(psi_current.real, dx)
    grad_imag = np.gradient(psi_current.imag, dx)

    # Combine gradients to calculate the squared magnitude
    squared_gradient = grad_real**2 + grad_imag**2

    # Kinetic energy
    prefactor = 0.5 * hbar**2 / m
    kinetic_energy = prefactor * np.sum(squared_gradient) * dx
    assert isinstance(potential_energy, object)
    total_energy_list_fibonacci.append(kinetic_energy + potential_energy)

    # Potential energy
    potential_energy = np.sum(np.abs(psi_current)**2 * v_fibonacci) * dx
    potential_energy_list_fibonacci.append(potential_energy)

    # Total energy
    total_energy_list_fibonacci.append(kinetic_energy + potential_energy)

    # Combine gradients for the full kinetic energy density
    squared_gradient = grad_real ** 2 + grad_imag ** 2
    kinetic_energy = 0.5 * hbar ** 2 / m * np.sum(squared_gradient) * dx
    kinetic_energy_list_fibonacci.append(kinetic_energy)
    grad_psi = grad_real + 1j * grad_imag  # Combine gradients into full complex gradient
    kinetic_energy = 0.5 * hbar ** 2 / m * np.sum(np.abs(grad_psi) ** 2) * dx
    kinetic_energy_list_fibonacci.append(kinetic_energy)

    # Calculate potential energy
    potential_energy: Any = np.sum(np.abs(psi_current)**2 * v_fibonacci) * dx
    potential_energy_list_fibonacci.append(potential_energy)

    # Calculate total energy
    total_energy_list_fibonacci.append(kinetic_energy + potential_energy)

# Plot Energy Dynamics
plt.figure(figsize=(12, 6))
plt.plot(np.arange(time_steps) * dt, kinetic_energy_list_fibonacci, label='Kinetic Energy')
plt.plot(np.arange(time_steps) * dt, potential_energy_list_fibonacci, label='Potential Energy')
plt.plot(np.arange(time_steps) * dt, total_energy_list_fibonacci, label='Total Energy')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy Dynamics Over Time (Fibonacci Potential)')
plt.legend()
plt.show()

# Track Variance for Each Potential
var_fibonacci = track_variance(psi_global, v_fibonacci, dx, time_steps)
var_constant = track_variance(psi_global, v_constant, dx, time_steps)
var_quadratic = track_variance(psi_global, v_quadratic, dx, time_steps)

# Plot Variance Comparison
plot_variance(
    time_steps, dt,
    [var_fibonacci, var_constant, var_quadratic],
    ['Fibonacci Potential', 'Constant Potential', 'Quadratic Potential'],
    'Comparison of Variance Over Time'
)

# Energy Dynamics
kinetic_energy_list = []
potential_energy_list = []
total_energy_list = []

psi_current = psi_global.copy()

for _ in range(time_steps):
    psi_current = evolve_wave_function(psi_current, v_fibonacci, dx, dt)
    kinetic_energy, potential_energy = compute_energy(psi_current, v_fibonacci, dx)
    kinetic_energy_list.append(kinetic_energy)
    potential_energy_list.append(potential_energy)
    total_energy_list.append(kinetic_energy + potential_energy)

# Plot Energy Dynamics
plt.figure(figsize=(12, 6))
plt.plot(np.arange(time_steps) * dt, kinetic_energy_list, label='Kinetic Energy')
plt.plot(np.arange(time_steps) * dt, potential_energy_list, label='Potential Energy')
plt.plot(np.arange(time_steps) * dt, total_energy_list, label='Total Energy')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy Dynamics Over Time (Fibonacci Potential)')
plt.legend()
plt.show()

# Plot the Wave Function Evolution
psi_current = psi_global.copy()
psi_list = [np.abs(psi_current)**2]
for _ in range(time_steps):
    psi_current = evolve_wave_function(psi_current, v_fibonacci, dx, dt)
    psi_list.append(np.abs(psi_current)**2)

plt.figure(figsize=(12, 6))
for i in range(0, time_steps, time_steps // 10):
    plt.plot(x, psi_list[i], label=f'Time {i * dt:.2f}')
plt.xlabel('Position')
plt.ylabel('Probability Density |ψ|^2')
plt.legend()
plt.title('Wave Function Evolution Under Fibonacci Potential')
plt.show()