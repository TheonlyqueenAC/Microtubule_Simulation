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
psi = np.exp(-0.5 * ((x - x0) / sigma) ** 2)  # Gaussian wave packet
psi /= np.sqrt(np.sum(np.abs(psi) ** 2))  # Normalize wavefunction

# No potential
V = np.zeros_like(x)

# Store the wavefunction's probability density for visualization
psi_list = [np.abs(psi) ** 2]

# Function to evolve the wavefunction using finite differences
def evolve(psi, V, dx, dt):
    Laplacian = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx ** 2
    psi_new = psi - (1j * hbar * dt / (2 * m)) * (Laplacian + V * psi)
    return psi_new

# Time evolution loop
for _ in range(time_steps):
    psi = evolve(psi, V, dx, dt)
    psi_list.append(np.abs(psi) ** 2)

# Visualization: Plot the time evolution of the wavefunction
plt.figure(figsize=(10, 6))
for i in range(0, time_steps, time_steps // 10):  # Plot 10 snapshots
    plt.plot(x, psi_list[i], label=f'Time {i * dt:.2f}')
plt.xlabel('Position')
plt.ylabel('Probability Density |ψ|^2')
plt.title('Time Evolution of Quantum Coherence')
plt.legend()
plt.savefig('quantum_coherence_evolution.png')  # Save the plot as a PNG file
plt.show()