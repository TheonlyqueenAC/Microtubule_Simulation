import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import os
import json
import time
import shutil
from pathlib import Path

# Create output directories
output_dirs = [
    '/Users/acdmacmini/Desktop/microtutuble_simulation/datafiles',
    './datafiles'  # Local PyCharm directory
]

for directory in output_dirs:
    os.makedirs(directory, exist_ok=True)

# Create visualization directories
visualization_dir = '../figures/final_figures'
os.makedirs(visualization_dir, exist_ok=True)


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
n_frames = 60  # Increased frames to reach t=3.0
L = 10.0  # Domain size
dx = L / grid_size
dt = 0.05  # With 60 frames and dt=0.05, total time will be 3.0


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

# Create arrays to store all wavefunction data
all_psi_fib = []
all_psi_uniform = []

# Store the initial state
all_psi_fib.append(np.abs(psi_fib) ** 2)
all_psi_uniform.append(np.abs(psi_uniform) ** 2)

# Time evolution
for frame in range(n_frames):
    current_time = (frame + 1) * dt

    # Evolve systems
    psi_fib = evolve_wavefunction(psi_fib, V_fibonacci, dx, dt)
    psi_uniform = evolve_wavefunction(psi_uniform, V_uniform, dx, dt)

    # Store probability densities
    all_psi_fib.append(np.abs(psi_fib) ** 2)
    all_psi_uniform.append(np.abs(psi_uniform) ** 2)

    # Calculate variances
    variance_fib.append(np.var(np.abs(psi_fib) ** 2))
    variance_uniform.append(np.var(np.abs(psi_uniform) ** 2))

    # Generate snapshots at t=0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
    if abs(current_time % 0.5) < 1e-10 or frame == n_frames - 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Fibonacci wavefunction
        im1 = axes[0].imshow(np.abs(psi_fib) ** 2, extent=[0, L, 0, L], origin='lower', cmap=cmap_quantum)
        axes[0].set_title(f'Fibonacci-Scaled Wavefunction (t={current_time:.1f})')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0], label='Probability Density')

        # Plot Uniform wavefunction
        im2 = axes[1].imshow(np.abs(psi_uniform) ** 2, extent=[0, L, 0, L], origin='lower', cmap=cmap_quantum)
        axes[1].set_title(f'Uniform-Scaled Wavefunction (t={current_time:.1f})')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1], label='Probability Density')

        plt.tight_layout()
        snapshot_filename = f"{visualization_dir}/wavefunction_t_{current_time:.1f}.png"
        plt.savefig(snapshot_filename, dpi=300)
        print(f"Saved snapshot at t={current_time:.1f}")
        plt.close()

# Create animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Initial plots
im1 = ax1.imshow(all_psi_fib[0], extent=[0, L, 0, L], origin='lower', cmap=cmap_quantum, vmin=0,
                 vmax=np.max(all_psi_fib[0]) * 1.2)
ax1.set_title('Fibonacci-Scaled Wavefunction')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
plt.colorbar(im1, ax=ax1, label='Probability Density')

im2 = ax2.imshow(all_psi_uniform[0], extent=[0, L, 0, L], origin='lower', cmap=cmap_quantum, vmin=0,
                 vmax=np.max(all_psi_uniform[0]) * 1.2)
ax2.set_title('Uniform-Scaled Wavefunction')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
plt.colorbar(im2, ax=ax2, label='Probability Density')

time_text = fig.suptitle(f't = 0.0')


# Animation update function
def update(frame):
    im1.set_array(all_psi_fib[frame])
    im2.set_array(all_psi_uniform[frame])
    time_text.set_text(f't = {frame * dt:.2f}')
    return im1, im2, time_text


# Create animation
ani = animation.FuncAnimation(fig, update, frames=n_frames + 1, interval=100, blit=False)

# Save animation as MP4
writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
animation_filename = f"{visualization_dir}/wavefunction_evolution.mp4"
ani.save(animation_filename, writer=writer)
print(f"Saved animation to {animation_filename}")
plt.close()

# Plotting statistical analysis
plt.figure(figsize=(15, 10))

# Variance Comparison Plot
plt.subplot(2, 1, 1)
plt.plot(np.arange(n_frames) * dt, variance_fib, 'b-', linewidth=2, label='Fibonacci-Scaled')
plt.plot(np.arange(n_frames) * dt, variance_uniform, 'r-', linewidth=2, label='Uniform')
plt.xlabel('Time')
plt.ylabel('Wavefunction Variance')
plt.title('Coherence Comparison: Wavefunction Dispersion')
plt.grid(True)
plt.legend()

# Variance Ratio and Improvement Percentage
plt.subplot(2, 1, 2)
variance_ratio = np.array(variance_uniform) / np.array(variance_fib)
improvement_percent = (np.array(variance_uniform) - np.array(variance_fib)) / np.array(variance_uniform) * 100

plt.plot(np.arange(n_frames) * dt, variance_ratio, 'g-', linewidth=2, label='Variance Ratio (Uniform/Fibonacci)')
plt.plot(np.arange(n_frames) * dt, improvement_percent, 'm--', linewidth=2, label='Improvement Percentage')
plt.xlabel('Time')
plt.ylabel('Ratio / Percentage')
plt.title('Quantitative Improvement with Fibonacci Scaling')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(f'{visualization_dir}/fibonacci_statistical_comparison.png', dpi=300)
plt.close()

# Fibonacci Ratios Plot
plt.figure(figsize=(10, 6))
plt.plot(range(2, len(fib_seq)), fib_ratios, 'o-', color='navy')
plt.axhline(y=phi, color='red', linestyle='--', alpha=0.7, label=f'Golden Ratio (φ ≈ {phi:.8f})')
plt.xlabel('n')
plt.ylabel('F(n+1) / F(n)')
plt.title('Convergence of Fibonacci Ratios to Golden Ratio')
plt.grid(True)
plt.legend()
plt.savefig(f'{visualization_dir}/fibonacci_ratios.png', dpi=300)
plt.close()

# Save raw data to files
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Prepare data for export
data_to_save = {
    'parameters': {
        'grid_size': grid_size,
        'n_frames': n_frames,
        'domain_size': L,
        'dx': dx,
        'dt': dt,
        'n_terms': n_terms
    },
    'variance_fib': variance_fib,
    'variance_uniform': variance_uniform,
    'variance_ratio': variance_ratio.tolist(),
    'improvement_percent': improvement_percent.tolist(),
    'time_points': (np.arange(n_frames) * dt).tolist(),
    'fibonacci_sequence': fib_seq.tolist(),
    'fibonacci_ratios': fib_ratios.tolist()
}

# Save metadata and statistics to JSON
for directory in output_dirs:
    json_filename = os.path.join(directory, f'simulation_results_{timestamp}.json')
    with open(json_filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Saved JSON data to {json_filename}")

    # Save numpy arrays for probability densities
    # Note: We're saving only key timepoints to save space
    save_timepoints = [0, 10, 20, 30, 40, 50, 59]  # t=0, 0.5, 1.0, 1.5, 2.0, 2.5, ~3.0

    for i in save_timepoints:
        # Save Fibonacci wavefunction data
        fib_filename = os.path.join(directory, f'psi_fib_t{i * dt:.1f}_{timestamp}.npy')
        np.save(fib_filename, all_psi_fib[i])

        # Save Uniform wavefunction data
        uniform_filename = os.path.join(directory, f'psi_uniform_t{i * dt:.1f}_{timestamp}.npy')
        np.save(uniform_filename, all_psi_uniform[i])

    # Also copy visualization files to these directories
    for viz_file in os.listdir(visualization_dir):
        if viz_file.endswith('.png') or viz_file.endswith('.mp4'):
            src = os.path.join(visualization_dir, viz_file)
            dst = os.path.join(directory, viz_file)
            shutil.copy2(src, dst)

print("Simulation and analysis complete. All data saved.")