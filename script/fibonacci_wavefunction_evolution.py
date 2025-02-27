import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import os


# Create custom scientific colormaps
def create_scientific_cmap(name='quantum'):
    if name == 'quantum':
        # Blue to red colormap for probability density
        colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    elif name == 'coherence':
        # Green to purple for coherence measures
        colors = [(0, 0.1, 0), (0, 0.5, 0), (0, 1, 0), (0.5, 0, 0.5), (0.7, 0, 1), (1, 0, 1)]
    else:
        # Default
        colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]

    return LinearSegmentedColormap.from_list(name, colors, N=256)


# Calculate golden ratio
phi = (1 + np.sqrt(5)) / 2

# Parameters for the simulation
grid_size = 200
n_frames = 30
L = 10.0  # Domain size
dx = L / grid_size
dt = 0.05


# Generate Fibonacci sequence
def generate_fibonacci_sequence(n):
    """Generate Fibonacci sequence up to n terms."""
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return np.array(fib)


# Generate Fibonacci sequence
n_terms = 30
fib_seq = generate_fibonacci_sequence(n_terms)


# Safe ratio calculation to avoid divide by zero
def calculate_fibonacci_ratios(sequence):
    ratios = []
    for i in range(1, len(sequence) - 1):
        if sequence[i] != 0:
            ratios.append(sequence[i + 1] / sequence[i])
    return np.array(ratios)


# Calculate ratios of consecutive terms
fib_ratios = calculate_fibonacci_ratios(fib_seq)

# Create grid
x = np.linspace(0, L, grid_size)
y = np.linspace(0, L, grid_size)
X, Y = np.meshgrid(x, y)


# Create Fibonacci-scaled potential
def create_fibonacci_potential(X, Y, fib_seq, scale=0.5):
    # Normalize Fibonacci sequence
    norm_fib = fib_seq / np.max(fib_seq) * scale

    # Create potential that follows Fibonacci spacing
    V = np.zeros_like(X)
    for i, val in enumerate(norm_fib):
        if i < 5:  # Skip first few terms
            continue
        V += np.exp(-((X - val * L) ** 2 + (Y - val * L) ** 2) / (0.05 ** 2))

    return V


# Create regular (uniform) potential for comparison
def create_uniform_potential(X, Y, n_terms, scale=0.5):
    # Create uniform spacing
    uniform_seq = np.linspace(0, 1, n_terms) * scale

    # Create potential with uniform spacing
    V = np.zeros_like(X)
    for i, val in enumerate(uniform_seq):
        if i < 5:  # Skip first few terms
            continue
        V += np.exp(-((X - val * L) ** 2 + (Y - val * L) ** 2) / (0.05 ** 2))

    return V


# Simulate wavefunction evolution
def evolve_wavefunction(psi, V, dx, dt, steps=1):
    """Evolve wavefunction using split-step Fourier method"""
    hbar = 1.0
    m = 1.0

    # Momentum space grid
    kx = np.fft.fftfreq(psi.shape[0], dx) * 2 * np.pi
    ky = np.fft.fftfreq(psi.shape[1], dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_SQ = KX ** 2 + KY ** 2

    # Kinetic energy operator in momentum space
    T_k = -0.5 * hbar ** 2 / m * K_SQ

    # Time evolution
    for _ in range(steps):
        # Half-step in position space
        psi = psi * np.exp(-1j * V * dt / (2 * hbar))

        # Full step in momentum space
        psi_k = np.fft.fft2(psi)
        psi_k = psi_k * np.exp(-1j * T_k * dt / hbar)
        psi = np.fft.ifft2(psi_k)

        # Half-step in position space
        psi = psi * np.exp(-1j * V * dt / (2 * hbar))

        # Normalize
        psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx ** 2)

    return psi


# Create potentials
V_fibonacci = create_fibonacci_potential(X, Y, fib_seq)
V_uniform = create_uniform_potential(X, Y, n_terms)

# Initialize wavefunctions
psi_init = np.exp(-((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (0.2 ** 2)) * np.exp(1j * X)
psi_init = psi_init / np.sqrt(np.sum(np.abs(psi_init) ** 2) * (L / grid_size) ** 2)

psi_fib = psi_init.copy()
psi_uniform = psi_init.copy()

# Create colormaps
cmap_quantum = create_scientific_cmap('quantum')
cmap_coherence = create_scientific_cmap('coherence')

# Prepare arrays for variance tracking
variance_fib = []
variance_uniform = []

# Time evolution
for _ in range(n_frames):
    # Evolve systems
    psi_fib = evolve_wavefunction(psi_fib, V_fibonacci, dx, dt)
    psi_uniform = evolve_wavefunction(psi_uniform, V_uniform, dx, dt)

    # Calculate variances
    variance_fib.append(np.var(np.abs(psi_fib) ** 2))
    variance_uniform.append(np.var(np.abs(psi_uniform) ** 2))

# Plotting
plt.figure(figsize=(15, 10))

# Variance Comparison Plot
plt.subplot(2, 1, 1)
plt.plot(range(n_frames), variance_fib, 'b-', linewidth=2, label='Fibonacci-Scaled')
plt.plot(range(n_frames), variance_uniform, 'r-', linewidth=2, label='Uniform')
plt.xlabel('Time Steps')
plt.ylabel('Wavefunction Variance')
plt.title('Coherence Comparison: Wavefunction Dispersion')
plt.grid(True)
plt.legend()

# Variance Ratio and Improvement Percentage
plt.subplot(2, 1, 2)
variance_ratio = np.array(variance_uniform) / np.array(variance_fib)
improvement_percent = (np.array(variance_uniform) - np.array(variance_fib)) / np.array(variance_uniform) * 100

plt.plot(range(n_frames), variance_ratio, 'g-', linewidth=2, label='Variance Ratio (Uniform/Fibonacci)')
plt.plot(range(n_frames), improvement_percent, 'm--', linewidth=2, label='Improvement Percentage')
plt.xlabel('Time Steps')
plt.ylabel('Ratio / Percentage')
plt.title('Quantitative Improvement with Fibonacci Scaling')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('fibonacci_statistical_comparison.png', dpi=300)
plt.show()

# Fibonacci Ratios Plot
plt.figure(figsize=(10, 6))
plt.plot(range(2, len(fib_seq)), fib_ratios, 'o-', color='navy')
plt.axhline(y=phi, color='red', linestyle='--', alpha=0.7, label=f'Golden Ratio (φ ≈ {phi:.8f})')
plt.xlabel('n')
plt.ylabel('F(n+1) / F(n)')
plt.title('Convergence of Fibonacci Ratios to Golden Ratio')
plt.grid(True)
plt.legend()
plt.show()

print("Simulation and analysis complete.")