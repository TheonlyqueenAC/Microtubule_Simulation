import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define constants from NIH study
D_c = 0.1  # Cytokine diffusion coefficient
kappa_c = 0.05  # Cytokine degradation rate
alpha = 0.2  # Influence of cytokine concentration on coherence loss
viral_toxicity_factor = 0.005  # Viral load impact on structural damage

# Spatial and temporal parameters
L = 10  # Spatial domain length
Nx = 100  # Number of spatial points
dx = L / Nx  # Spatial step size
T_max = 50  # Maximum simulation time
Nt = 500  # Number of time steps
dt = min(0.01, T_max / Nt)  # Ensure dt is small enough

# Initialize cytokine concentration and coherence field
cytokine_conc = np.zeros(Nx)
coherence_field = np.ones(Nx, dtype=complex)  # Initial coherence field

# Define initial cytokine perturbation (adjusted for biological accuracy)
cytokine_conc[Nx // 3:Nx // 2] = 0.5  # Reduced from 1.0 to 0.5 for sustained, realistic activation


# Function for cytokine diffusion and degradation
def cytokine_evolution(_, cytokine_conc):
    dCdt = np.zeros_like(cytokine_conc)
    for i in range(1, Nx - 1):
        dCdt[i] = D_c * (cytokine_conc[i + 1] - 2 * cytokine_conc[i] + cytokine_conc[i - 1]) / max(dx ** 2,
                                                                                                   1e-6) - kappa_c * \
                  cytokine_conc[i]
    return dCdt


# Solve cytokine diffusion equation
time_points = np.linspace(0, T_max, Nt)
cytokine_sol = solve_ivp(cytokine_evolution, [0, T_max], cytokine_conc, t_eval=time_points)

# Extract cytokine concentration over time
C_t = cytokine_sol.y


# Define coherence evolution equation (Schrödinger-like with cytokine perturbation)
def coherence_evolution(coherence_field, C_t, dt, alpha, t):
    coherence_new = np.copy(coherence_field)

    # Define HAND disease progression phases
    early_phase = t < 0.3 * Nt  # Early-stage HIV
    chronic_phase = 0.3 * Nt <= t < 0.8 * Nt  # ART-controlled HIV
    late_phase = t >= 0.8 * Nt  # End-stage HAND/AIDS

    for i in range(1, Nx - 1):
        laplacian = (coherence_field[i + 1] - 2 * coherence_field[i] + coherence_field[i - 1]) / max(dx ** 2,
                                                                                                     1e-6)  # Stability fix
        alpha_C = np.clip(alpha * C_t[i], -10, 10)  # Prevent extreme values

        # Reduce cytokine effect in late-stage HAND due to immune exhaustion
        if late_phase:
            alpha_C *= 0.2  # T-cell depletion means weaker cytokine response

        # Apply viral load effect in late-stage HAND (skyrocketing viral toxicity)
        viral_toxicity = viral_toxicity_factor * (t / Nt) * np.exp(-0.05 * (i - Nx // 2) ** 2) if late_phase else 0.0

        # Introduce stochastic fluctuations in early HAND
        if early_phase:
            alpha_C *= 1 + 0.05 * np.random.randn()

        coherence_new[i] += -1j * dt * (laplacian - alpha_C) * coherence_field[i] * 0.98  # Reduce stability damping

        # Apply cumulative structural degradation from viral load
        coherence_new[i] *= np.exp(-viral_toxicity * dt * (t / Nt))  # Progressive breakdown in late-stage HAND

    # Hard limit to prevent runaway values
    coherence_new = np.clip(coherence_new, -1e6, 1e6)
    return coherence_new


# Simulate coherence evolution over time
coherence_t = np.zeros((Nt, Nx), dtype=complex)
coherence_t[0, :] = coherence_field  # Initial coherence field

for t in range(1, Nt):
    coherence_t[t, :] = coherence_evolution(coherence_t[t - 1, :], C_t[:, t], dt, alpha, t)

# Generate final high-resolution figures
fig, ax = plt.subplots(2, 1, figsize=(10, 8), dpi=300)

# Plot cytokine concentration over time
ax[0].imshow(C_t, aspect='auto', cmap='inferno', extent=[0, T_max, 0, L])
ax[0].set_title("Cytokine Concentration Over Time", fontsize=12)
ax[0].set_xlabel("Time", fontsize=10)
ax[0].set_ylabel("Spatial Position", fontsize=10)

# Plot coherence magnitude evolution
coherence_plot = np.log1p(np.abs(coherence_t))  # Normalize for stability
ax[1].imshow(coherence_plot, aspect='auto', cmap='magma', extent=[0, T_max, 0, L])
ax[1].set_title("Quantum Coherence Evolution Under Cytokine and Viral Perturbation", fontsize=12)
ax[1].set_xlabel("Time", fontsize=10)
ax[1].set_ylabel("Spatial Position", fontsize=10)

plt.tight_layout()

# Save final figures
fig.savefig("HIV_cytokine_coherence_evolution_final.png", dpi=300)

plt.show()