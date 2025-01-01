import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0  # Reduced Planck's constant (arbitrary units)
m = 1.0  # Effective mass of the wave packet (arbitrary units)
L = 10.0  # Length of the simulated microtubule (arbitrary units)
N = 100  # Number of points in space
dx = L / N  # Spatial step size
dt = 0.01  # Time step size
time_steps = 300  # Total number of time steps

# Create spatial grid
x = np.linspace(0, L, N)

# Initialize wavefunction: Gaussian packet
sigma = 1.0  # Width of the wave packet
x0 = L / 2  # Center of the wave packet
Psi = np.exp(-0.5 * ((x - x0) / sigma) ** 2)  # Gaussian
Psi = Psi / np.sqrt(np.sum(Psi ** 2))  # Normalize wavefunction

# No potential: No potential for this simple case
V = np.zeros_like(x)

# Store the wavefunction's probability density for visualization
Psi_list = [np.abs(Psi) ** 2]


# Function to evolve the wavefunction using finite differences
def evolve(Psi, V, dx, dt):
    # Compute the Laplacian using finite differences
    Laplacian = (np.roll(Psi, -1) - 2 * Psi + np.roll(Psi, 1)) / dx ** 2

    # Update the wavefunction using the Schrödinger equation
    Psi_new = Psi - (1j * hbar * dt / (2 * m)) * Laplacian + (1j * V * dt / hbar) * Psi

    return Psi_new


# Time evolution loop
for _ in range(time_steps):
    Psi = evolve(Psi, V, dx, dt)
    Psi_list.append(np.abs(Psi) ** 2)  # Append the probability density

# Visualization: Plot the time evolution of the wavefunction
plt.figure(figsize=(10, 6))
for i in range(0, time_steps, time_steps // 10):  # Plot 10 snapshots
    plt.plot(x, Psi_list[i], label=f'Time {i * dt:.2f}')
plt.xlabel('Position')
plt.ylabel('Probability Density |Psi|^2')
plt.title('Time Evolution of Quantum Coherence')
plt.legend()
plt.savefig('quantum_coherence_evolution.png')
plt.show()