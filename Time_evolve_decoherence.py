import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0  # Effective mass
L = 10.0  # Length of the domain
N = 100  # Number of spatial points
dx = L / N  # Spatial step size
dt = 0.01  # Time step size
time_steps = 300  # Total number of time steps
Gamma = 0.05  # Constant decoherence rate
x = np.linspace(0, L, N)

# Initialize wavefunction: Gaussian wave packet
def initialize_wave_function(grid, center, width):
    wave_packet = np.exp(-0.5 * ((grid - center) / width) ** 2)
    return wave_packet / np.sqrt(np.sum(np.abs(wave_packet) ** 2))  # Normalize

psi = initialize_wave_function(x, L / 2, L / 10)  # Initial wavefunction

# No potential
V = np.zeros_like(x)

# Store the wavefunction's probability density for visualization
psi_list = [np.abs(psi) ** 2]

# Function to evolve the wavefunction with decoherence
def evolve_with_decoherence(Psi, V, dx, dt, Gamma):
    # Compute Laplacian
    laplacian = (np.roll(Psi, -1) - 2 * Psi + np.roll(Psi, 1)) / dx**2
    # Time evolution with decoherence
    Psi_new = Psi - (1j * hbar * dt / (2 * m)) * laplacian + (1j * V * dt / hbar) * Psi - Gamma * Psi * dt
    # Normalize the wavefunction
    norm_factor = np.sqrt(np.sum(np.abs(Psi_new)**2) * dx)
    Psi_new /= norm_factor
    return Psi_new

# Time evolution loop
for _ in range(time_steps):
    psi = evolve_with_decoherence(psi, V, dx, dt, Gamma)
    psi_list.append(np.abs(psi) ** 2)  # Store probability density

# Visualization: Plot the time evolution of the wavefunction
plt.figure(figsize=(10, 6))
for i in range(0, time_steps, time_steps // 10):  # Plot 10 snapshots
    plt.plot(x, psi_list[i], label=f'Time {i * dt:.2f}')
plt.xlabel('Position')
plt.ylabel('Probability Density |ψ|^2')
plt.title('Time Evolution of Quantum Decoherence')
plt.legend()
plt.savefig('Time_evolution_quantum_decoherence.png')  # Save the plot as a PNG file
plt.show()
# Define spatial grid
