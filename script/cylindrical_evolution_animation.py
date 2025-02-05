
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0  # Effective mass
L = 10.0  # Axial length of microtubule
R_inner = 7.0  # Inner radius of microtubule (arbitrary units)
R_outer = 12.5  # Outer radius of microtubule (arbitrary units)
N_r = 100  # Number of radial grid points
N_z = 100  # Number of axial grid points
dr = (R_outer - R_inner) / N_r  # Radial step size
dz = L / N_z  # Axial step size
dt = 0.01  # Time step size
time_steps = 300  # Total time steps
r = np.linspace(R_inner, R_outer, N_r)
sigma_r = (R_outer - R_inner) / 10  # Standard deviation for radial decoherence
gamma = 0.05 * (1 + np.exp(-((r - R_outer) / sigma_r) ** 2))  # Example radial decoherence
Psi = np.zeros_like(r)  # Initialize Psi before any operation
Psi -= gamma * Psi * dt

# Create spatial grids
r = np.linspace(R_inner, R_outer, N_r)
z = np.linspace(0, L, N_z)
R, Z = np.meshgrid(r, z)  # 2D grid for visualization

# Initialize wavefunction: Gaussian in z, uniform in r
sigma_z = L / 10
z0 = L / 2
Psi = np.exp(-0.5 * ((Z - z0) / sigma_z) ** 2) * (R_outer - R)  # Constrain radially
Psi /= np.sqrt(np.sum(np.abs(Psi) ** 2))  # Normalize

# Potential: Tubulin periodicity in z, confining walls in r
V_tubulin = 5.0 * np.cos(2 * np.pi * Z / L)
V_walls = np.zeros_like(R)
V_walls[R < R_inner] = 1e6  # Confinement at inner wall
V_walls[R > R_outer] = 1e6  # Confinement at outer wall
V = V_tubulin + V_walls


# Time evolution function
def evolve_cylindrical(Psi, V, dr, dz, dt):
    laplacian_r = (np.roll(Psi, -1, axis=0) - 2 * Psi + np.roll(Psi, 1, axis=0)) / dr ** 2
    laplacian_z = (np.roll(Psi, -1, axis=1) - 2 * Psi + np.roll(Psi, 1, axis=1)) / dz ** 2
    laplacian_r /= (R ** 2 + 1e-6)  # Avoid division by zero
    Psi_new = Psi - (1j * hbar * dt / (2 * m)) * (laplacian_r + laplacian_z + V * Psi)
    return Psi_new


# Time evolution loop
Psi_list = [np.abs(Psi) ** 2]
for _ in range(time_steps):
    Psi = evolve_cylindrical(Psi, V, dr, dz, dt)
    Psi_list.append(np.abs(Psi) ** 2)

# Normalize the wavefunction at each step
norm_factor = np.sum(np.abs(Psi) ** 2 * r[:, None]) * dr * dz  # Cylindrical volume element
Psi /= np.sqrt(norm_factor)

fig, ax = plt.subplots(figsize=(10, 6))
cbar = None  # Declare global colorbar variable


# Update function for animation
def update(frame):
    ax.clear()
    psi_frame = Psi_list[frame]  # Get the probability density at this time step
    contour = ax.contourf(z, r, psi_frame, levels=50, cmap='viridis')  # Save contour plot
    if frame == 0:
        # Add the colorbar only once (for the first frame)
        global cbar
        cbar = fig.colorbar(contour, ax=ax, label='|Ψ|^2')  # Associate colorbar with contour plot
    ax.set_xlabel('Axial Position (z)')
    ax.set_ylabel('Radial Position (r)')
    ax.set_title(f'Time Evolution | Time = {frame * dt:.2f}')


# Animation setup
FRAME_INTERVAL_MS = 100  # Frame interval in milliseconds
psi_data = Psi_list  # Renamed for clarity
animation = FuncAnimation(fig=fig, func=update, frames=len(psi_data), interval=FRAME_INTERVAL_MS)

# Save the animation using FFmpeg writer
print("Starting to save animation...")
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
animation.save('cylindrical_evolution.mp4', writer=writer)
print("Animation saved successfully.")

# Visualization: Probability density at final time step
plt.figure(figsize=(10, 6))
plt.contourf(Z, R, Psi_list[-1], levels=50, cmap='viridis')
plt.xlabel('Axial Position (z)')
plt.ylabel('Radial Position (r)')
plt.title('Probability Density |Ψ|^2 in Cylindrical Geometry')
plt.colorbar(label='|Ψ|^2')
plt.show()