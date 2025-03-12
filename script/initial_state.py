import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec

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

# Cytokine parameters
V_0 = 5.0  # Peak cytokine potential
Gamma_0 = 0.05  # Baseline decoherence rate
alpha_c = 0.1  # Scaling factor for cytokine-induced decoherence

# Create spatial grids
r = np.linspace(R_inner, R_outer, N_r)
z = np.linspace(0, L, N_z)
R, Z = np.meshgrid(r, z)  # 2D grid for visualization


# Fibonacci scaling function
def generate_fibonacci_sequence(n):
    """Generate Fibonacci sequence up to n terms."""
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return np.array(fib)


# Apply Fibonacci scaling to spatial coordinates
fib_sequence = generate_fibonacci_sequence(N_z)
fib_scaling = fib_sequence[-N_z:] / np.max(fib_sequence[-N_z:]) * L
Z_fib = np.zeros_like(Z)
for i in range(N_z):
    Z_fib[i, :] = fib_scaling[i]

# Initialize wavefunctions: regular and Fibonacci-scaled
sigma_z = L / 10
z0 = L / 2
# Regular wavefunction
Psi_reg = np.exp(-0.5 * ((Z - z0) / sigma_z) ** 2) * (R_outer - R)
Psi_reg /= np.sqrt(np.sum(np.abs(Psi_reg) ** 2) * dr * dz)

# Fibonacci-scaled wavefunction
Psi_fib = np.exp(-0.5 * ((Z_fib - z0) / sigma_z) ** 2) * (R_outer - R)
Psi_fib /= np.sqrt(np.sum(np.abs(Psi_fib) ** 2) * dr * dz)

# Initial cytokine concentration (centered perturbation)
C = np.exp(-((Z - L / 2) ** 2) / (2 * (L / 10) ** 2)) * np.exp(-((R - R_outer) ** 2) / (2 * (R_outer - R_inner) ** 2))
C = np.clip(C, 0, 1)  # Normalize to range [0,1]

# Potentials with and without cytokine influence
V_base = 5.0 * np.cos(2 * np.pi * Z / L)
V_walls = np.zeros_like(R)
V_walls[R < R_inner] = 1e6  # Confinement at inner wall
V_walls[R > R_outer] = 1e6  # Confinement at outer wall
V_reg = V_base + V_walls
V_fib = V_base + V_walls

# Cytokine-dependent decoherence
Gamma_cytokine = Gamma_0 * (1 + alpha_c * C)


# Event horizon calculation
def calculate_event_horizon(Gamma):
    r_h = 1 / (1 + np.mean(Gamma, axis=1) / 5)
    r_h_scaled = R_inner + (R_outer - R_inner) * r_h / np.max(r_h)
    return r_h_scaled


# Time evolution function with decoherence
def evolve_wavefunction(Psi, V, Gamma, dr, dz, dt):
    laplacian_r = (np.roll(Psi, -1, axis=0) - 2 * Psi + np.roll(Psi, 1, axis=0)) / dr ** 2
    laplacian_z = (np.roll(Psi, -1, axis=1) - 2 * Psi + np.roll(Psi, 1, axis=1)) / dz ** 2
    Psi_new = Psi - (1j * hbar * dt / (2 * m)) * (laplacian_r + laplacian_z + V * Psi) - Gamma * Psi * dt
    norm_factor = np.sqrt(np.sum(np.abs(Psi_new) ** 2 * r[:, None]) * dr * dz)
    return Psi_new / norm_factor


# Store probability densities and event horizons
Psi_reg_list = [np.abs(Psi_reg) ** 2]
Psi_fib_list = [np.abs(Psi_fib) ** 2]
event_horizon_list = [calculate_event_horizon(Gamma_cytokine)]
cytokine_list = [C.copy()]

# Time evolution loop
for t in range(time_steps):
    # Update cytokine field (simple diffusion model)
    laplacian_r_C = (np.roll(C, -1, axis=0) - 2 * C + np.roll(C, 1, axis=0)) / dr ** 2
    laplacian_z_C = (np.roll(C, -1, axis=1) - 2 * C + np.roll(C, 1, axis=1)) / dz ** 2
    C = C + dt * (0.1 * (laplacian_r_C + laplacian_z_C) - 0.01 * C)
    C = np.clip(C, 0, 1)  # Keep within bounds

    # Update cytokine-dependent decoherence
    Gamma_cytokine = Gamma_0 * (1 + alpha_c * C)

    # Update regular wavefunction
    Psi_reg = evolve_wavefunction(Psi_reg, V_reg, Gamma_cytokine, dr, dz, dt)

    # Update Fibonacci-scaled wavefunction
    Psi_fib = evolve_wavefunction(Psi_fib, V_fib, Gamma_cytokine, dr, dz, dt)

    # Store results
    Psi_reg_list.append(np.abs(Psi_reg) ** 2)
    Psi_fib_list.append(np.abs(Psi_fib) ** 2)
    event_horizon_list.append(calculate_event_horizon(Gamma_cytokine))
    cytokine_list.append(C.copy())

# Create animation
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])  # Regular wavefunction
ax2 = fig.add_subplot(gs[0, 1])  # Fibonacci wavefunction
ax3 = fig.add_subplot(gs[0, 2])  # Cytokine field
ax4 = fig.add_subplot(gs[1, :])  # Coherence comparison over time

# Initialize contour plots
contour1 = ax1.contourf(Z, R, Psi_reg_list[0], levels=50, cmap='viridis')
contour2 = ax2.contourf(Z, R, Psi_fib_list[0], levels=50, cmap='viridis')
contour3 = ax3.contourf(Z, R, cytokine_list[0], levels=50, cmap='plasma')
line1, = ax4.plot(z, np.mean(Psi_reg_list[0], axis=0), 'b-', label='Regular')
line2, = ax4.plot(z, np.mean(Psi_fib_list[0], axis=0), 'r-', label='Fibonacci')

# Add colorbars
cbar1 = fig.colorbar(contour1, ax=ax1)
cbar2 = fig.colorbar(contour2, ax=ax2)
cbar3 = fig.colorbar(contour3, ax=ax3)

# Set titles
ax1.set_title('Regular Grid')
ax2.set_title('Fibonacci-Scaled Grid')
ax3.set_title('Cytokine Perturbation')
ax4.set_title('Coherence Comparison (Axial Average)')

# Set labels
ax1.set_xlabel('Axial Position (z)')
ax1.set_ylabel('Radial Position (r)')
ax2.set_xlabel('Axial Position (z)')
ax2.set_ylabel('Radial Position (r)')
ax3.set_xlabel('Axial Position (z)')
ax3.set_ylabel('Radial Position (r)')
ax4.set_xlabel('Axial Position (z)')
ax4.set_ylabel('Mean Probability Density')
ax4.legend()


# Update function for animation
def update(frame):
    # Update contour plots
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Plot regular wavefunction with event horizon
    cont1 = ax1.contourf(Z, R, Psi_reg_list[frame], levels=50, cmap='viridis')
    ax1.plot(z, event_horizon_list[frame], 'r--', linewidth=2, label='Event Horizon')
    ax1.set_title(f'Regular Grid (t={frame * dt:.2f})')
    ax1.set_xlabel('Axial Position (z)')
    ax1.set_ylabel('Radial Position (r)')

    # Plot Fibonacci wavefunction with event horizon
    cont2 = ax2.contourf(Z, R, Psi_fib_list[frame], levels=50, cmap='viridis')
    ax2.plot(z, event_horizon_list[frame], 'r--', linewidth=2, label='Event Horizon')
    ax2.set_title(f'Fibonacci-Scaled Grid (t={frame * dt:.2f})')
    ax2.set_xlabel('Axial Position (z)')
    ax2.set_ylabel('Radial Position (r)')

    # Plot cytokine field
    cont3 = ax3.contourf(Z, R, cytokine_list[frame], levels=50, cmap='plasma')
    ax3.set_title(f'Cytokine Field (t={frame * dt:.2f})')
    ax3.set_xlabel('Axial Position (z)')
    ax3.set_ylabel('Radial Position (r)')

    # Update line plots for coherence comparison
    line1.set_ydata(np.mean(Psi_reg_list[frame], axis=0))
    line2.set_ydata(np.mean(Psi_fib_list[frame], axis=0))

    # Set axis limits
    ax1.set_xlim(0, L)
    ax1.set_ylim(R_inner, R_outer)
    ax2.set_xlim(0, L)
    ax2.set_ylim(R_inner, R_outer)
    ax3.set_xlim(0, L)
    ax3.set_ylim(R_inner, R_outer)

    return cont1, cont2, cont3, line1, line2


# Create animation
ani = FuncAnimation(fig, update, frames=len(Psi_reg_list), interval=100, blit=False)

# Save animation
writer = FFMpegWriter(fps=15, metadata=dict(artist='AC Demidont'), bitrate=3000)
ani.save('quantum_coherence_evolution.mp4', writer=writer)
print("Animation saved successfully.")

plt.tight_layout()
plt.show()