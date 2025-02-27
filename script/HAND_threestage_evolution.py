import numpy as np
import matplotlib.pyplot as plt
import time

# ===== CONFIGURATION =====
# Set ultra-light test mode
ULTRA_LIGHT = False

# ===== CONSTANTS =====
# Simplified constants (no physical units to avoid tiny timesteps)
hbar = 1.0  # Normalized Planck constant
m = 1.0  # Normalized mass
k_B = 1.0  # Normalized Boltzmann constant
T = 1.0  # Normalized temperature
phi = (1 + np.sqrt(5)) / 2  # Golden ratio

# ===== SIMULATION PARAMETERS =====
# Minimal grid size for test run
if ULTRA_LIGHT:
    N_r = 5  # Radial grid points
    N_z = 10  # Axial grid points
    N_t = 5  # Number of time steps
else:
    N_r = 20  # Radial grid points
    N_z = 40  # Axial grid points
    N_t = 20  # Number of time steps

# Normalized dimensions
L = 1.0
R_inner = 0.7
R_outer = 1.0

# Spatial discretization
dr = (R_outer - R_inner) / N_r
dz = L / N_z

# Create spatial grids
r = np.linspace(R_inner, R_outer, N_r)
z = np.linspace(0, L, N_z)
R, Z = np.meshgrid(r, z)

# Simplified time step (avoid extremely small values)
dt = 0.1
print(f"Grid size: {N_r}x{N_z}, Time steps: {N_t}")
print(f"Time step: {dt}")


# ===== MINIMAL SIMULATION FUNCTIONS =====

def initialize_gaussian_wavefunction():
    """Create a simple Gaussian wavefunction."""
    psi = np.zeros((N_z, N_r), dtype=complex)

    # Center of the Gaussian
    z0, r0 = L / 2, (R_inner + R_outer) / 2

    # Width of the Gaussian
    sigma_z, sigma_r = L / 5, (R_outer - R_inner) / 5

    # Create the Gaussian with Fibonacci modulation
    for i in range(N_z):
        for j in range(N_r):
            # Basic Gaussian
            gaussian = np.exp(-0.5 * ((z[i] - z0) / sigma_z) ** 2 - 0.5 * ((r[j] - r0) / sigma_r) ** 2)
            # Fibonacci modulation
            mod = np.sin(phi * np.pi * (r[j] - R_inner) / (R_outer - R_inner))
            psi[i, j] = gaussian * mod

    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi) ** 2))
    return psi / norm


def evolve_wavefunction(psi, cytokine_field=None):
    """Simplified wavefunction evolution."""
    if cytokine_field is None:
        cytokine_field = np.zeros((N_z, N_r))

    # Create basic potential and decoherence
    V = np.cos(2 * np.pi * Z / L) + cytokine_field  # Simple periodic potential plus cytokine effect
    Gamma = 0.01 * (1 + 0.5 * cytokine_field)  # Simple decoherence model

    # Very simplified evolution algorithm (just to demonstrate)
    laplacian = np.zeros_like(psi, dtype=complex)
    for i in range(1, N_z - 1):
        for j in range(1, N_r - 1):
            # Approximate Laplacian
            laplacian[i, j] = (psi[i + 1, j] + psi[i - 1, j] + psi[i, j + 1] + psi[i, j - 1] - 4 * psi[i, j]) / (
                        dr * dz)

    # Simple update (not physically accurate, but numerically stable)
    psi_new = psi - 1j * dt * (laplacian - V * psi) - dt * Gamma * psi

    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi_new) ** 2))
    return psi_new / norm


def initialize_cytokines(phase="acute"):
    """Initialize a simple cytokine field based on HIV phase."""
    # Start with zeros
    cytokines = np.zeros((N_z, N_r))

    if phase == "acute":
        # Localized high concentration near outer boundary
        for i in range(N_z):
            for j in range(N_r):
                if r[j] > 0.9 * R_outer:
                    cytokines[i, j] = 0.8 * np.exp(-((z[i] - L / 2) / (L / 4)) ** 2)
    elif phase == "chronic":
        # More distributed pattern
        for i in range(N_z):
            for j in range(N_r):
                cytokines[i, j] = 0.5 * np.exp(
                    -((r[j] - R_outer) / (R_outer - R_inner)) ** 2) + 0.2 * np.random.random()
    elif phase == "ART-controlled":
        # Low-level pattern
        for i in range(N_z):
            for j in range(N_r):
                cytokines[i, j] = 0.3 * np.exp(-((r[j] - R_outer) / (R_outer - R_inner)) ** 2)

    return cytokines


def calculate_event_horizon(psi, cytokine_field):
    """Calculate a simplified event horizon."""
    # Just use a threshold on the wavefunction probability
    probability = np.abs(psi) ** 2

    # Find where probability drops below threshold
    r_h = np.zeros(N_z)
    threshold = 0.1 * np.max(probability)

    for i in range(N_z):
        found = False
        for j in range(N_r - 1, -1, -1):  # Search from outer to inner
            if probability[i, j] > threshold and not found:
                r_h[i] = r[j]
                found = True

        # Default if not found
        if not found:
            r_h[i] = r[0]

    return r_h


def run_minimal_simulation(phase):
    """Run a minimal simulation for one HIV phase."""
    print(f"Running minimal simulation for {phase} phase...")
    start_time = time.time()

    # Initialize
    cytokines = initialize_cytokines(phase)
    psi = initialize_gaussian_wavefunction()

    # Minimal evolution
    for step in range(N_t):
        print(f"  Step {step + 1}/{N_t}...", end="\r")
        psi = evolve_wavefunction(psi, cytokines)

    # Compute final event horizon
    r_h = calculate_event_horizon(psi, cytokines)

    print(f"\nCompleted in {time.time() - start_time:.2f} seconds")

    # Return results
    return {
        "phase": phase,
        "final_cytokine": cytokines,
        "final_psi": np.abs(psi) ** 2,
        "final_horizon": r_h
    }


def visualize_minimal_results(results):
    """Create a simple visualization of results."""
    # Create figure
    fig, axes = plt.subplots(len(results), 3, figsize=(12, 4 * len(results)))

    # Ensure axes is a 2D array even with one phase
    if len(results) == 1:
        axes = np.array([axes])

    # Plot each phase
    for i, phase in enumerate(results.keys()):
        result = results[phase]

        # Cytokine concentration
        c1 = axes[i, 0].contourf(Z, R, result["final_cytokine"], levels=10, cmap='plasma')
        axes[i, 0].set_title(f"Cytokines - {phase}")
        axes[i, 0].set_xlabel("z")
        axes[i, 0].set_ylabel("r")
        plt.colorbar(c1, ax=axes[i, 0])

        # Wavefunction probability
        c2 = axes[i, 1].contourf(Z, R, result["final_psi"], levels=10, cmap='viridis')
        axes[i, 1].set_title(f"Quantum Coherence - {phase}")
        axes[i, 1].set_xlabel("z")
        axes[i, 1].set_ylabel("r")
        plt.colorbar(c2, ax=axes[i, 1])

        # Coherence with event horizon
        c3 = axes[i, 2].contourf(Z, R, result["final_psi"], levels=10, cmap='viridis', alpha=0.7)
        axes[i, 2].plot(z, result["final_horizon"], 'r--', linewidth=2, label="Event Horizon")
        axes[i, 2].set_title(f"With Event Horizon - {phase}")
        axes[i, 2].set_xlabel("z")
        axes[i, 2].set_ylabel("r")
        axes[i, 2].legend()

    plt.tight_layout()
    plt.savefig("minimal_quantum_coherence.png", dpi=150)
    plt.show()


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Use only one phase in ultra-light mode
    if ULTRA_LIGHT:
        phases = ["acute"]
    else:
        phases = ["acute", "ART-controlled", "chronic"]

    # Run simulations
    results = {}
    for phase in phases:
        results[phase] = run_minimal_simulation(phase)

    # Visualize
    visualize_minimal_results(results)

    print("\nSimulation complete. Results saved to minimal_quantum_coherence.png")