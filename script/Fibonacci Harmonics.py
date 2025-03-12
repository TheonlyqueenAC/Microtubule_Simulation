
import numpy as np
import matplotlib.pyplot as plt

# Constants
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
hbar = 1.0545718e-34  # Reduced Planck constant (Joule seconds)
m = 9.10938356e-31  # Electron mass (kg)


def fibonacci_scaling_wavefunction(time_steps, space_points, initial_wavefunction):
    """
    Simulates wavefunction evolution with Fibonacci harmonics.

    Parameters:
        time_steps (int): Number of time steps to evolve the wavefunction.
        space_points (int): Number of spatial points.
        initial_wavefunction (ndarray): Initial wavefunction as a 1D array.

    Returns:
        ndarray: Final wavefunction after evolution.
    """
    # Spatial grid
    x = np.linspace(-10, 10, space_points)
    dx = x[1] - x[0]

    # Time step based on scaled dynamics
    dt = 0.01 * (dx ** 2 * m) / (phi * hbar)  # Reduced time step for stability

    # Adjusted Potential Energy
    V = 0.5 * m * (phi * x / 5) ** 2  # Reduce scaling for stability

    # Boundary Absorbing Factor
    absorbing_factor = np.exp(-0.01 * (np.abs(x) - 9) ** 2)  # Absorb boundaries

    # Hamiltonian operator components
    kinetic = -0.5 * hbar ** 2 / m / dx ** 2
    H = np.diag(kinetic * np.ones(space_points - 1), -1) + \
        np.diag(kinetic * np.ones(space_points - 1), 1) + \
        np.diag(V)

    # Initialize the wavefunction
    psi = initial_wavefunction.copy()

    # Time evolution with normalization and boundary absorption
    for _ in range(time_steps):
        psi = (np.eye(space_points) - 1j * H * dt / hbar) @ psi  # Matrix multiplication for time evolution
        psi *= absorbing_factor  # Apply absorbing boundaries
        psi /= np.linalg.norm(psi)  # Normalize wavefunction

    return psi, x


# Initial wavefunction (Gaussian packet)
def gaussian_packet(x, x0=0, k0=5, sigma=1):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) * np.exp(1j * k0 * x)


# Simulation parameters
space_points = 500
time_steps = 1000
x = np.linspace(-10, 10, space_points)
initial_psi = gaussian_packet(x)

# Run simulation
final_psi, x = fibonacci_scaling_wavefunction(time_steps, space_points, initial_psi)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(x, np.abs(initial_psi) ** 2, label="Initial |Psi|^2")
plt.plot(x, np.abs(final_psi) ** 2, label="Final |Psi|^2", linestyle='--')
plt.title("Wavefunction Evolution with Fibonacci Harmonics")
plt.xlabel("Position")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

