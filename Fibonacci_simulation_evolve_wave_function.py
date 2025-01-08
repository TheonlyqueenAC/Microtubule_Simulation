# Constant definitions
hbar = 1.0545718e-34  # Reduced Planck's constant (J·s)
m = 9.10938356e-31  # Mass of the particle (kg)

import numpy as np


def calculate_laplacian(wave_function: np.ndarray, spatial_step: float) -> np.ndarray:
    """Calculate the Laplacian of the wave function using finite differences."""
    return (np.roll(wave_function, -1) - 2 * wave_function + np.roll(wave_function, 1)) / spatial_step ** 2


def normalize_wave_function(wave_function: np.ndarray, spatial_step: float) -> np.ndarray:
    """Normalize the wave function to conserve total probability."""
    norm_factor = np.sqrt(np.sum(np.abs(wave_function) ** 2) * spatial_step)
    return wave_function / norm_factor


def evolve_wave_function(
        wave_function: np.ndarray, potential: np.ndarray, spatial_step: float, time_step: float
) -> np.ndarray:
    """
    Evolve the wave function using finite differences.

    Args:
        wave_function: The local wave function array as a NumPy array.
        potential: The potential energy array as a NumPy array.
        spatial_step: The spatial discretization step size.
        time_step: The time step for evolution.

    Returns:
        The evolved wave function as a normalized NumPy array.
    """
    laplacian = calculate_laplacian(wave_function, spatial_step)
    # Update the wave function using the Schrödinger equation
    wave_function_new = wave_function - (1j * hbar * time_step / (2 * m)) * (
            laplacian + potential * wave_function
    )
    # Normalize the wave function
    return normalize_wave_function(wave_function_new, spatial_step)

