import numpy as np

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
