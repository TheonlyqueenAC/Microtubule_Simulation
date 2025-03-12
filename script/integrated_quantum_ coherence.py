import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0  # Effective mass
L = 10.0  # Axial length of microtubule
R_inner = 7.0  # Inner radius of microtubule
R_outer = 12.5  # Outer radius of microtubule
N_r = 100  # Number of radial grid points
N_z = 100  # Number of axial grid points
dr = (R_outer - R_inner) / N_r  # Radial step size
dz = L / N_z  # Axial step size
dt = 0.01  # Time step size
time_steps = 300  # Total time steps

# Cytokine parameters
V_0 = 5.0  # Peak cytokine potential
Gamma_0 = 0.05  # Baseline decoherence rate
alpha_c = 0.1  # Scaling factor for cytokine-induced decoherence

# Create spatial grids
r = np.linspace(R_inner, R_outer, N_r)
z = np.linspace(0, L, N_z)
R, Z = np.meshgrid(r, z)  # 2D grid for visualization

# Create custom scientific colormaps
density_cmap = LinearSegmentedColormap.from_list(
    'density', [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)], N=256)
cytokine_cmap = LinearSegmentedColormap.from_list(
    'cytokine', [(0, 0, 0), (0.5, 0, 0), (1, 0, 0), (1, 0.5, 0), (1, 1, 0)], N=256)


# Generate Fibonacci sequence for scaling
def fibonacci_sequence(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return np.array(fib)


# Apply Fibonacci scaling to the grid
fib_seq = fibonacci_sequence(N_z + 10)[-N_z:]  # Use last N_z elements
fib_scaling = fib_seq / np.max(fib_seq) * L  # Normalize to domain size

# Create Fibonacci-scaled grid
Z_fib = np.zeros_like(Z)
for i in range(N_z):
    Z_fib[i, :] = fib_scaling[i]

# Initialize wavefunctions and cytokine field
# Regular grid wavefunction
sigma_z = L / 10
z0 = L / 2
Psi_reg = np.exp(-0.5 * ((Z - z0) / sigma_z) ** 2) * (R_outer - R)
Psi_reg /= np.sqrt(np.sum(np.abs(Psi_reg) ** 2 * r[:, None]) * dr * dz)

# Fibonacci-scaled wavefunction
Psi_fib = np.exp(-0.5 * ((Z_fib - z0) / sigma_z) ** 2) * (R_outer - R)
Psi_fib /= np.sqrt(np.sum(np.abs(Psi_fib) ** 2 * r[:, None]) * dr * dz)

# Initial cytokine concentration (centered perturbation)
C = np.exp(-((Z - L / 2) ** 2) / (2 * (L / 10) ** 2)) * np.exp(-((R - R_outer) ** 2) / (2 * (R_outer - R_inner) ** 2))
C = np.clip(C, 0, 1)  # Normalize to range [0,1]


# Calculate event horizon boundary
def calculate_event_horizon(Gamma):
    """Calculate coherence-preserving boundary based on decoherence field."""
    r_h = 1 / (1 + np.mean(Gamma, axis=1) / 5)
    r_h_scaled = R_inner + (R_outer - R_inner) * r_h / np.max(r_h)
    return r_h_scaled

# Potential and decoherence setup
V_base = 5.0 * np.cos(2 * np.pi * Z / L)  # Base potential from tubulin
V_walls = np.zeros_like(R)
V_walls[R < R_inner] = 1e6  # Confinement at inner wall
V_walls[R > R_outer] = 1e6  # Confinement at outer wall
V_reg = V_base + V_walls
V_fib = V_base + V_walls

# Cytokine-dependent decoherence
Gamma_cytokine = Gamma_0 * (1 + alpha_c * C)


# Time evolution functions
def evolve_cytokines(C, dr, dz, dt, D_c=0.1, kappa_c=0.01):
    """Evolve cytokine concentration field."""
    laplacian_r = (np.roll(C, -1, axis=0) - 2 * C + np.roll(C, 1, axis=0)) / dr ** 2
    laplacian_z = (np.roll(C, -1, axis=1) - 2 * C + np.roll(C, 1, axis=1)) / dz ** 2
    return C + dt * (D_c * (laplacian_r + laplacian_z) - kappa_c * C)


def evolve_wavefunction(Psi, V, Gamma, dr, dz, dt):
    """Evolve wavefunction with potential and decoherence."""
    laplacian_r = (np.roll(Psi, -1, axis=0) - 2 * Psi + np.roll(Psi, 1, axis=0)) / dr ** 2
    laplacian_z = (np.roll(Psi, -1, axis=1) - 2 * Psi + np.roll(Psi, 1, axis=1)) / dz ** 2

    # Add radial term for cylindrical coordinates
    for i in range(N_r):
        if r[i] > 1e-10:  # Avoid division by zero
            laplacian_r[:, i] += (1 / r[i]) * (np.roll(Psi, -1, axis=0) - np.roll(Psi, 1, axis=0))[:, i] / (2 * dr)

    # Time evolution with split-step method
    Psi_half = Psi * np.exp(-0.5j * V * dt / hbar)  # Half-step in potential
    Psi_k = Psi_half - 1j * hbar * dt / (2 * m) * (laplacian_r + laplacian_z)  # Full step in kinetic
    Psi_new = Psi_k * np.exp(-0.5j * V * dt / hbar)  # Half-step in potential

    # Add decoherence term
    Psi_new = Psi_new * np.exp(-Gamma * dt)

    # Normalize
    norm = np.sqrt(np.sum(np.abs(Psi_new) ** 2 * r[:, None]) * dr * dz)
    return Psi_new / norm


# Set up data storage
frames_to_save = 50  # Number of frames to save (to manage memory)
save_interval = time_steps // frames_to_save

# Storage for visualization data
Psi_reg_list = []
Psi_fib_list = []
cytokine_list = []
event_horizon_list = []
variance_reg = []
variance_fib = []


# Calculate variance (measure of dispersion)
def calculate_variance(Psi, R, Z, dr, dz):
    """Calculate spatial variance of the wavefunction."""
    r_mean = np.sum(R * np.abs(Psi) ** 2 * r[:, None]) * dr * dz
    z_mean = np.sum(Z * np.abs(Psi) ** 2 * r[:, None]) * dr * dz
    var_r = np.sum((R - r_mean) ** 2 * np.abs(Psi) ** 2 * r[:, None]) * dr * dz
    var_z = np.sum((Z - z_mean) ** 2 * np.abs(Psi) ** 2 * r[:, None]) * dr * dz
    return var_r + var_z  # Total spatial variance

# Store initial state
Psi_reg_list.append(np.abs(Psi_reg) ** 2)
Psi_fib_list.append(np.abs(Psi_fib) ** 2)
cytokine_list.append(C.copy())
event_horizon_list.append(calculate_event_horizon(Gamma_cytokine))
variance_reg.append(calculate_variance(Psi_reg, R, Z, dr, dz))
variance_fib.append(calculate_variance(Psi_fib, R, Z_fib, dr, dz))

# Time evolution with progress tracking
print("Starting time evolution simulation...")
for step in range(1, time_steps + 1):
    # Update cytokine field
    C = evolve_cytokines(C, dr, dz, dt)
    C = np.clip(C, 0, 1)  # Keep concentration bounded

    # Update decoherence field
    Gamma_cytokine = Gamma_0 * (1 + alpha_c * C)

    # Update wavefunctions
    Psi_reg = evolve_wavefunction(Psi_reg, V_reg, Gamma_cytokine, dr, dz, dt)
    Psi_fib = evolve_wavefunction(Psi_fib, V_fib, Gamma_cytokine, dr, dz, dt)

    # Store results at specified intervals
    if step % save_interval == 0 or step == 1:
        Psi_reg_list.append(np.abs(Psi_reg) ** 2)
        Psi_fib_list.append(np.abs(Psi_fib) ** 2)
        cytokine_list.append(C.copy())
        event_horizon_list.append(calculate_event_horizon(Gamma_cytokine))
        variance_reg.append(calculate_variance(Psi_reg, R, Z, dr, dz))
        variance_fib.append(calculate_variance(Psi_fib, R, Z_fib, dr, dz))

        # Progress update
        print(f"Step {step}/{time_steps} ({step / time_steps * 100:.1f}%) completed")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, height_ratios=[2, 1])

# Top row: side-by-side visualizations
ax_reg = fig.add_subplot(gs[0, 0])
ax_fib = fig.add_subplot(gs[0, 1])
ax_cyt = fig.add_subplot(gs[0, 2])

# Bottom row: quantitative comparisons
ax_var = fig.add_subplot(gs[1, :2])  # Variance comparison
ax_prof = fig.add_subplot(gs[1, 2])  # Axial profile

# Initial plots
contour_reg = ax_reg.contourf(Z, R, Psi_reg_list[0], levels=50, cmap=density_cmap)
contour_fib = ax_fib.contourf(Z, R, Psi_fib_list[0], levels=50, cmap=density_cmap)
contour_cyt = ax_cyt.contourf(Z, R, cytokine_list[0], levels=50, cmap=cytokine_cmap)

# Add event horizon overlays
horizon_reg, = ax_reg.plot(z, event_horizon_list[0], 'r--', linewidth=2, label='Event Horizon')
horizon_fib, = ax_fib.plot(z, event_horizon_list[0], 'r--', linewidth=2, label='Event Horizon')

# Add colorbars
cbar_reg = plt.colorbar(contour_reg, ax=ax_reg)
cbar_fib = plt.colorbar(contour_fib, ax=ax_fib)
cbar_cyt = plt.colorbar(contour_cyt, ax=ax_cyt)

# Set titles and labels
ax_reg.set_title('Uniform Grid')
ax_fib.set_title('Fibonacci-Scaled Grid')
ax_cyt.set_title('Cytokine Perturbation')

for ax in [ax_reg, ax_fib, ax_cyt]:
    ax.set_xlabel('Axial Position (z)')
    ax.set_ylabel('Radial Position (r)')
    ax.legend()

# Set up variance comparison plot
var_reg_line, = ax_var.plot([0], variance_reg[:1], 'b-', linewidth=2, label='Uniform Grid')
var_fib_line, = ax_var.plot([0], variance_fib[:1], 'r-', linewidth=2, label='Fibonacci-Scaled')
ax_var.set_xlabel('Time Step')
ax_var.set_ylabel('Wavefunction Variance')
ax_var.set_title('Coherence Comparison (Lower Variance = Better Coherence)')
ax_var.legend()
ax_var.grid(True)

# Set up axial profile plot
z_vals = np.linspace(0, L, N_z)
prof_reg, = ax_prof.plot(z_vals, np.mean(Psi_reg_list[0], axis=0), 'b-', linewidth=2, label='Uniform Grid')
prof_fib, = ax_prof.plot(z_vals, np.mean(Psi_fib_list[0], axis=0), 'r-', linewidth=2, label='Fibonacci-Scaled')
ax_prof.set_xlabel('Axial Position (z)')
ax_prof.set_ylabel('Mean Probability Density')
ax_prof.set_title('Axial Probability Profile')
ax_prof.legend()
ax_prof.grid(True)


# Animation update function
def update(frame):
    # Clear main plots
    ax_reg.clear()
    ax_fib.clear()
    ax_cyt.clear()

    # Contour plots
    contour_reg = ax_reg.contourf(Z, R, Psi_reg_list[frame], levels=50, cmap=density_cmap)
    contour_fib = ax_fib.contourf(Z, R, Psi_fib_list[frame], levels=50, cmap=density_cmap)
    contour_cyt = ax_cyt.contourf(Z, R, cytokine_list[frame], levels=50, cmap=cytokine_cmap)

    # Event horizon overlays clearly at each step
    ax_reg.plot(z, event_horizon_list[frame], 'r--', linewidth=2, label='Event Horizon')
    ax_fib.plot(z, event_horizon_list[frame], 'r--', linewidth=2, label='Event Horizon')

    # Titles with time
    actual_time = frame * save_interval * dt
    ax_reg.set_title(f'Uniform Grid (t={actual_time:.2f})')
    ax_fib.set_title(f'Fibonacci-Scaled Grid (t={actual_time:.2f})')
    ax_cyt.set_title(f'Cytokine Perturbation (t={actual_time:.2f})')

    # Consistent labels and axis limits
    for ax in [ax_reg, ax_fib, ax_cyt]:
        ax.set_xlabel('Axial Position (z)')
        ax.set_ylabel('Radial Position (r)')
        ax.set_xlim(0, L)
        ax.set_ylim(R_inner, R_outer)
        ax.legend(loc='upper right')

    # Update variance plot
    var_reg_line.set_data(range(frame + 1), variance_reg[:frame + 1])
    var_fib_line.set_data(range(frame + 1), variance_fib[:frame + 1])
    ax_var.relim()
    ax_var.autoscale_view()
    ax_var.set_xlabel('Time Step')
    ax_var.set_ylabel('Wavefunction Variance')
    ax_var.grid(True)
    ax_var.legend()

    # Update axial profile plot
    prof_reg.set_ydata(np.mean(Psi_reg_list[frame], axis=0))
    prof_fib.set_ydata(np.mean(Psi_fib_list[frame], axis=0))
    ax_prof.relim()
    ax_prof.autoscale_view()
    ax_prof.set_xlabel('Axial Position (z)')
    ax_prof.set_ylabel('Mean Probability Density')
    ax_prof.grid(True)
    ax_prof.legend()

    return [contour_reg, contour_fib, contour_cyt, var_reg_line, var_fib_line, prof_reg, prof_fib]

# Create animation
ani = FuncAnimation(fig, update, frames=len(Psi_reg_list), interval=200, blit=False)

# Save animation
print("Saving animation...")
writer = FFMpegWriter(fps=10, metadata=dict(artist='AC Demidont'), bitrate=5000)
ani.save('integrated_quantum_evolution.mp4', writer=writer)
print("Animation saved successfully!")

# Save final frame as static image
plt.figure(figsize=(18, 10))
plt.subplot(2, 3, 1)
plt.contourf(Z, R, Psi_reg_list[-1], levels=50, cmap=density_cmap)
plt.plot(z, event_horizon_list[-1], 'r--', linewidth=2)
plt.title('Final Uniform Grid')
plt.xlabel('Axial Position (z)')
plt.ylabel('Radial Position (r)')
plt.colorbar()

plt.subplot(2, 3, 2)
plt.contourf(Z, R, Psi_fib_list[-1], levels=50, cmap=density_cmap)
plt.plot(z, event_horizon_list[-1], 'r--', linewidth=2)
plt.title('Final Fibonacci-Scaled Grid')
plt.xlabel('Axial Position (z)')
plt.ylabel('Radial Position (r)')
plt.colorbar()

plt.subplot(2, 3, 3)
plt.contourf(Z, R, cytokine_list[-1], levels=50, cmap=cytokine_cmap)
plt.title('Final Cytokine Distribution')
plt.xlabel('Axial Position (z)')
plt.ylabel('Radial Position (r)')
plt.colorbar()

plt.subplot(2, 3, 4)
plt.plot(range(len(variance_reg)), variance_reg, 'b-', label='Uniform Grid')
plt.plot(range(len(variance_fib)), variance_fib, 'r-', label='Fibonacci-Scaled')
plt.xlabel('Time Step')
plt.ylabel('Wavefunction Variance')
plt.title('Coherence Comparison')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 5)
improvement = (np.array(variance_reg) - np.array(variance_fib)) / np.array(variance_reg) * 100
plt.plot(range(len(improvement)), improvement, 'g-')
plt.xlabel('Time Step')
plt.ylabel('Improvement (%)')
plt.title('Fibonacci Coherence Improvement')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(z, np.mean(Psi_reg_list[-1], axis=0), 'b-', label='Uniform Grid')
plt.plot(z, np.mean(Psi_fib_list[-1], axis=0), 'r-', label='Fibonacci-Scaled')
plt.xlabel('Axial Position (z)')
plt.ylabel('Mean Probability Density')
plt.title('Final Axial Profile')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('integrated_quantum_comparison.png', dpi=300)
plt.show()