import numpy as np
from matplotlib import pyplot as plt

# Constants
# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0  # Effective mass
L = 10.0  # Length of the domain
N = 100  # Number of spatial points
dx = L / N  # Spatial step size
dt = 0.01  # Time step size
time_steps = 300  # Total number of time steps

def generate_fibonacci_sequence(size):
    """Generate a Fibonacci sequence up to the given size."""
    fib_sequence = [0, 1]
    for _ in range(2, size):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return np.array(fib_sequence[:size])

def normalize_fibonacci_sequence(fib_sequence, max_length):
    """Normalize Fibonacci sequence to fit within the spatial domain."""
    return fib_sequence / np.max(fib_sequence) * max_length

def initialize_wave_function(grid, center, width):
    """Create a Gaussian wave packet."""
    wave_packet = np.exp(-0.5 * ((grid - center) / width) ** 2)
    return wave_packet / np.sqrt(np.sum(np.abs(wave_packet) ** 2))

# No potential
x = np.linspace(0, L, N)  # Define the spatial grid
V = np.zeros_like(x)

# Initialize the wavefunction as a Gaussian wave packet
psi = initialize_wave_function(x, L / 2, L / 10)

# Store the wavefunction's probability density for visualization
psi_list = [np.abs(psi) ** 2]

def evolve_wave_function(psi_local, potential, dx_local, dt_local):
    """Evolve the wave function using finite differences."""
    # Compute the Laplacian for the finite difference method
    laplacian = (np.roll(psi_local, -1) - 2 * psi_local + np.roll(psi_local, 1)) / dx_local**2
    psi_new = psi_local - (1j * hbar * dt_local / (2 * m)) * (laplacian + potential * psi_local)

    # Debug print: Norm before normalization
    print(f"Norm before evolution: {np.sum(np.abs(psi_local)**2) * dx_local}")
    print(f"Norm after evolution (pre-normalization): {np.sum(np.abs(psi_new)**2) * dx_local}")

    # Normalize the wave function to conserve probability
    norm_factor = np.sqrt(np.sum(np.abs(psi_new)**2) * dx_local)
    psi_new /= norm_factor

    # Debug print: Norm after normalization
    print(f"Norm after normalization: {np.sum(np.abs(psi_new)**2) * dx_local}")

    return psi_new
# Time evolution loop
for _ in range(time_steps):
    psi = evolve_wave_function(psi, V, dx, dt)
    psi_list.append(np.abs(psi) ** 2)

# Visualization: Plot the time evolution of the wavefunction
plt.figure(figsize=(10, 6))
for i in range(0, time_steps, time_steps // 10):  # Plot 10 snapshots
    plt.plot(x, psi_list[i], label=f'Time {i * dt:.2f}')
plt.xlabel('Position')
plt.ylabel('Probability Density |ψ|^2')
plt.title('Time Evolution of Quantum Coherence')
plt.legend()
plt.savefig('Time_evolution_quantum_coherence.png')  # Save the plot as a PNG file
plt.show()