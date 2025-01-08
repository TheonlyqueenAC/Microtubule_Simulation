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

# Generate Fibonacci sequence
fib_sequence = [0, 1]
for _ in range(2, N):
    fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])

# Normalize Fibonacci sequence to fit spatial domain
fib_sequence = np.array(fib_sequence[:N])
fib_ratios = fib_sequence / np.max(fib_sequence) * L  # Scale to spatial length L
x = np.linspace(0, L, N)

# Initialize wave function: Gaussian packet
sigma = 1.0  # Width of the wave packet
x0 = L / 2  # Center of the wave packet at the middle of the grid

psi_global = np.exp(-0.5 * ((x - x0) / sigma) ** 2)  # Gaussian wave packet
psi_global = psi_global / np.sqrt(np.sum(np.abs(psi_global) ** 2))  # Normalize wave function

# No potential
v_global = np.zeros_like(x)

# Store the wave function's probability density for visualization
psi_list = [np.abs(psi_global) ** 2]

# Function to evolve the wave function using finite differences
def evolve_wave_function(psi_local, potential, dx_local, dt_local):
    laplacian = (np.roll(psi_local, -1) - 2 * psi_local + np.roll(psi_local, 1)) / dx_local ** 2
    psi_new = psi_local - (1j * hbar * dt_local / (2 * m)) * (laplacian + potential * psi_local)
    return psi_new

# Time evolution loop
psi_current = psi_global.copy()
v_current = v_global.copy()

for _ in range(time_steps):
    psi_current = evolve_wave_function(psi_current, v_current, dx, dt)
    psi_list.append(np.abs(psi_current) ** 2)

# Visualization: Plot the time evolution of the wave function
plt.figure(figsize=(10, 6))
for i in range(0, time_steps, time_steps // 10):  # Plot 10 snapshots
    plt.plot(x, psi_list[i], label=f'Time {i * dt:.2f}')
plt.xlabel('Position (Fibonacci-Scaled)')
plt.ylabel('Probability Density |ψ|^2')
plt.title('Quantum Coherence Evolution with Fibonacci Scaling')
plt.legend()
plt.savefig('fibonacci_coherence_evolution.png')
plt.show()