import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Constants - using more physically meaningful notation
hbar = 1.0545718e-34  # Reduced Planck's constant (J·s)
m_eff = 1.0e-27  # Effective mass (kg)
L = 8.0e-9  # Axial length of microtubule (8 nm)
R_inner = 7.0e-9  # Inner radius of microtubule (7 nm)
R_outer = 12.0e-9  # Outer radius of microtubule (12 nm)
N_r = 100  # Number of radial grid points
N_z = 100  # Number of axial grid points
dr = (R_outer - R_inner) / N_r  # Radial step size
dz = L / N_z  # Axial step size
dt = 1.0e-15  # Time step size (femtoseconds)
time_steps = 300  # Total time steps

# Create spatial grids
r = np.linspace(R_inner, R_outer, N_r)
z = np.linspace(0, L, N_z)
R, Z = np.meshgrid(r, z)  # 2D grid for visualization

# Custom colormap for better visualization
colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
cmap_name = 'quantum_density'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# Initialize wavefunction: Gaussian in z, uniform in r with physical parameters
sigma_z = L / 10  # Width of Gaussian (0.8 nm)
z0 = L / 2  # Center position (4 nm)
# Initial wavefunction in position basis
Psi = np.exp(-0.5 * ((Z - z0) / sigma_z) ** 2) * np.sin(np.pi * (R - R_inner) / (R_outer - R_inner))
# Normalize
Psi /= np.sqrt(np.sum(np.abs(Psi) ** 2 * R) * dr * dz)  # Correct normalization for cylindrical coordinates

# Potential function with realistic parameters
# Tubulin periodicity (alpha/beta dimers with 8nm spacing)
V_tubulin = 5.0e-21 * np.cos(2 * np.pi * Z / (8.0e-9))  # ~5 meV potential
# Confining walls (hard boundary conditions)
V_walls = np.zeros_like(R)
V_walls[R < R_inner + dr] = 1e-18  # Inner wall (effectively infinite)
V_walls[R > R_outer - dr] = 1e-18  # Outer wall (effectively infinite)
V = V_tubulin + V_walls


# Time evolution function with improved stability and physics
def evolve_wavefunction(Psi, V, dr, dz, dt):
    """Evolve wavefunction using finite difference method with cylindrical coordinates."""
    # Radial part of Laplacian (includes 1/r * d/dr term for cylindrical coords)
    r_mid = r[:-1] + dr / 2  # Midpoints for derivative calculation
    d_dr = np.zeros_like(Psi)
    d_dr[:, 1:] = (Psi[:, 1:] - Psi[:, :-1]) / dr  # Forward difference

    # Second derivative in r
    d2_dr2 = np.zeros_like(Psi)
    d2_dr2[:, 1:-1] = (Psi[:, 2:] - 2 * Psi[:, 1:-1] + Psi[:, :-2]) / dr ** 2

    # Incorporate cylindrical coordinate terms
    radial_term = d2_dr2.copy()
    for i in range(1, N_r - 1):
        radial_term[:, i] += d_dr[:, i] / r[i]

    # Axial part of Laplacian (simpler - Cartesian)
    axial_term = np.zeros_like(Psi)
    axial_term[1:-1, :] = (Psi[2:, :] - 2 * Psi[1:-1, :] + Psi[:-2, :]) / dz ** 2

    # Combined Laplacian
    laplacian = radial_term + axial_term

    # Time evolution (split-step method for better stability)
    Psi_half = Psi * np.exp(-0.5j * V * dt / hbar)  # Half-step in potential
    Psi_k = Psi_half + 0.5j * hbar * dt / m_eff * laplacian  # Full step in kinetic
    Psi_new = Psi_k * np.exp(-0.5j * V * dt / hbar)  # Half-step in potential

    # Normalize
    norm = np.sqrt(np.sum(np.abs(Psi_new) ** 2 * R) * dr * dz)
    return Psi_new / norm


# Time evolution loop with data collection
Psi_list = []
t_list = []
energy_list = []

# Store initial state
Psi_list.append(np.abs(Psi) ** 2)
t_list.append(0)


# Function to calculate energy
def calculate_energy(Psi, V, dr, dz):
    """Calculate expectation value of energy."""
    # Kinetic energy via finite difference laplacian
    laplacian = np.zeros_like(Psi)
    laplacian[1:-1, 1:-1] = (Psi[2:, 1:-1] + Psi[:-2, 1:-1] + Psi[1:-1, 2:] + Psi[1:-1, :-2] - 4 * Psi[1:-1, 1:-1]) / (
                dr ** 2)
    kinetic = -0.5 * hbar ** 2 / m_eff * np.sum(np.conj(Psi) * laplacian * R) * dr * dz

    # Potential energy
    potential = np.sum(np.abs(Psi) ** 2 * V * R) * dr * dz

    return (kinetic + potential).real


# Calculate initial energy
energy_list.append(calculate_energy(Psi, V, dr, dz))

# Evolution with progress tracking
print("Starting time evolution simulation...")
for step in range(1, time_steps + 1):
    Psi = evolve_wavefunction(Psi, V, dr, dz, dt)

    # Store results (only store every 10th step to save memory)
    if step % 10 == 0:
        Psi_list.append(np.abs(Psi) ** 2)
        t_list.append(step * dt)
        energy_list.append(calculate_energy(Psi, V, dr, dz))

        # Progress update
        if step % 50 == 0:
            print(f"Completed step {step}/{time_steps} ({step / time_steps * 100:.1f}%)")

# Convert lists to arrays for easier handling
Psi_array = np.array(Psi_list)
t_array = np.array(t_list)
energy_array = np.array(energy_list)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.4])

# Main animation panel
ax_main = plt.subplot(gs[0, :])
ax_energy = plt.subplot(gs[1, 0])
ax_projection = plt.subplot(gs[1, 1])

# Initial plots
contour = ax_main.contourf(Z, R, Psi_list[0], levels=50, cmap=cm)
cbar = plt.colorbar(contour, ax=ax_main, label=r'Probability Density $|\Psi|^2$')
ax_main.set_xlabel('Axial Position (nm)')
ax_main.set_ylabel('Radial Position (nm)')
ax_main.set_title('Wavefunction Evolution in Cylindrical Microtubule', fontsize=14)

# Energy plot initialization
energy_line, = ax_energy.plot(t_array[:1] * 1e15, energy_array[:1] * 1e21, 'r-', linewidth=2)
ax_energy.set_xlabel('Time (fs)')
ax_energy.set_ylabel('Energy (zJ)')
ax_energy.set_title('System Energy')
ax_energy.grid(True)

# Projection plot initialization
z_projection, = ax_projection.plot(z, np.sum(Psi_list[0], axis=1), 'b-', linewidth=2)
ax_projection.set_xlabel('Axial Position (nm)')
ax_projection.set_ylabel('Axial Probability')
ax_projection.set_title('Axial Projection')
ax_projection.grid(True)


# Animation update function
def update(frame):
    # Update main contour plot
    ax_main.clear()
    contour = ax_main.contourf(Z * 1e9, R * 1e9, Psi_list[frame], levels=50, cmap=cm)
    ax_main.set_xlabel('Axial Position (nm)')
    ax_main.set_ylabel('Radial Position (nm)')
    ax_main.set_title(f'Wavefunction Evolution at t = {t_list[frame] * 1e15:.2f} fs', fontsize=14)

    # Update energy plot
    energy_line.set_data(t_array[:frame + 1] * 1e15, energy_array[:frame + 1] * 1e21)
    ax_energy.relim()
    ax_energy.autoscale_view()

    # Update projection plot
    z_projection.set_data(z * 1e9, np.sum(Psi_list[frame], axis=1))
    ax_projection.relim()
    ax_projection.autoscale_view()

    return contour, energy_line, z_projection


# Create animation
ani = FuncAnimation(fig, update, frames=len(Psi_list), interval=100)

# Save animation
print("Saving animation...")
writer = FFMpegWriter(fps=15, metadata=dict(artist='AC Demidont'), bitrate=5000)
ani.save('enhanced_cylindrical_evolution.mp4', writer=writer)
print("Animation saved successfully.")

# Show final frame
plt.figure(figsize=(10, 8))
plt.contourf(Z * 1e9, R * 1e9, Psi_list[-1], levels=50, cmap=cm)
plt.colorbar(label=r'Probability Density $|\Psi|^2$')
plt.xlabel('Axial Position (nm)')
plt.ylabel('Radial Position (nm)')
plt.title('Final Wavefunction Distribution')
plt.savefig('final_wavefunction.png', dpi=300)
plt.show()