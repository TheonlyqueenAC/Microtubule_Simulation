import numpy as np
import matplotlib.pyplot as plt

# Define constants from literature
D_c = 0.1  # Cytokine diffusion coefficient
kappa_c = 0.05  # Cytokine degradation rate
alpha = 0.5  # Strengthened cytokine impact on coherence loss

# 2D Spatial and Temporal Parameters
Lx, Ly = 10, 10  # Spatial domain size
Nx, Ny = 40, 40  # Reduced spatial resolution for faster execution
dx, dy = Lx / Nx, Ly / Ny  # Spatial step size
T_max = 30  # Shorter simulation time
Nt = 200  # Reduced time steps for speed
dt = min(0.005, T_max / Nt)  # Small time step for stability

# Initialize coherence field in 2D
coherence_field = np.ones((Nx, Ny), dtype=complex)  # Initial coherence field

# Stochastic noise parameters for cytokine fluctuations
sigma = 0.05  # Increased noise strength to enhance cytokine bursts

# Define HIV Infection Phases Based on Literature
def cytokine_fluctuation(t, phase):
    if phase == "acute":
        return 1.0 * np.exp(-t / 5)  # Rapid cytokine spike, then decay
    elif phase == "ART-controlled":
        return 0.3 + 0.1 * np.sin(0.05 * t)  # Low persistent cytokine oscillations
    elif phase == "uncontrolled":
        return 0.8 + 0.3 * np.sin(0.2 * t) + sigma * np.random.randn()  # High variability and large fluctuations

# Function for explicit finite-difference cytokine diffusion
def cytokine_diffusion_step(C, t, phase):
    C_new = np.copy(C)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            laplacian = (C[i+1, j] + C[i-1, j] + C[i, j+1] + C[i, j-1] - 4*C[i, j]) / dx**2
            noise = sigma * np.random.randn() * cytokine_fluctuation(t, phase)  # Apply literature-based cytokine model
            C_new[i, j] = C[i, j] + dt * (D_c * laplacian - kappa_c * C[i, j] + noise)
    return np.clip(C_new, 0, 1)  # Prevent extreme values

# Simulate cytokine diffusion for all three phases
phases = ["acute", "ART-controlled", "uncontrolled"]
C_t_dict = {phase: np.zeros((Nx, Ny, Nt)) for phase in phases}

for phase in phases:
    C_t_dict[phase][:, :, 0] = np.zeros((Nx, Ny))  # Initial state
    C_t_dict[phase][Nx//3:Nx//2, Ny//3:Ny//2, 0] = 0.5  # Localized cytokine initiation
    for t in range(1, Nt):
        C_t_dict[phase][:, :, t] = cytokine_diffusion_step(C_t_dict[phase][:, :, t-1], t, phase)

# Define coherence evolution equation in 2D with stronger cytokine effect
def coherence_evolution(coherence_field, C_t, dt, alpha):
    coherence_new = np.copy(coherence_field)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            laplacian = (coherence_field[i+1, j] + coherence_field[i-1, j] +
                         coherence_field[i, j+1] + coherence_field[i, j-1] - 4*coherence_field[i, j]) / max(dx**2, 1e-6)
            alpha_C = np.clip(alpha * C_t[i, j] * (1 + 0.2 * np.random.randn()), -10, 10)  # Randomized cytokine effect
            coherence_new[i, j] += -1j * dt * (laplacian - alpha_C) * coherence_field[i, j] * 0.99  # Stability damping
            coherence_new[i, j] *= np.exp(-0.02 * C_t[i, j])  # Apply exponential coherence decay where cytokine is high
            coherence_new[i, j] = np.clip(coherence_new[i, j], -1e6, 1e6)  # Prevent overflow
    return coherence_new

# Simulate coherence evolution for all three phases
coherence_t_dict = {phase: np.zeros((Nx, Ny, Nt), dtype=complex) for phase in phases}

for phase in phases:
    coherence_t_dict[phase][:, :, 0] = np.ones((Nx, Ny), dtype=complex)  # Initial coherence field
    for t in range(1, Nt):
        coherence_t_dict[phase][:, :, t] = coherence_evolution(coherence_t_dict[phase][:, :, t-1], C_t_dict[phase][:, :, t], dt, alpha)

# Visualization of multi-panel results
fig, ax = plt.subplots(3, 2, figsize=(12, 12))

for i, phase in enumerate(phases):
    # Plot final cytokine concentration field
    ax[i, 0].imshow(C_t_dict[phase][:, :, -1], cmap='plasma', extent=[0, Lx, 0, Ly])
    ax[i, 0].set_title(f"Cytokine Concentration ({phase.capitalize()})")
    ax[i, 0].set_xlabel("X Position")
    ax[i, 0].set_ylabel("Y Position")

    # Plot final coherence magnitude
    ax[i, 1].imshow(np.abs(coherence_t_dict[phase][:, :, -1]), cmap='viridis', extent=[0, Lx, 0, Ly])
    ax[i, 1].set_title(f"Quantum Coherence ({phase.capitalize()})")
    ax[i, 1].set_xlabel("X Position")
    ax[i, 1].set_ylabel("Y Position")

plt.tight_layout()
plt.show()