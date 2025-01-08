import numpy as np
from fibonacci_simulation_refactored import (
    generate_fibonacci_sequence,
    normalize_fibonacci_sequence,
    initialize_wave_function,
    evolve_wave_function
)

# Define constants and grid
L = 10.0
N = 100
dx = L / N
x = np.linspace(0, L, N)

# Generate and normalize Fibonacci sequence
fib_sequence = generate_fibonacci_sequence(N)
fib_ratios = normalize_fibonacci_sequence(fib_sequence, L)

# Initialize wave function
sigma = 1.0
x0 = L / 2
psi_global = initialize_wave_function(x, x0, sigma)

# Define zero potential
v_global = np.zeros_like(x)

# Time evolution
psi_list = [np.abs(psi_global) ** 2]
psi_current = psi_global.copy()
time_steps = 300
dt = 0.01

for _ in range(time_steps):
    psi_current = evolve_wave_function(psi_current, v_global, dx, dt)
    psi_list.append(np.abs(psi_current) ** 2)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i in range(0, time_steps, time_steps // 10):
    plt.plot(x, psi_list[i], label=f'Time {i * dt:.2f}')
plt.xlabel('Position')
plt.ylabel('Probability Density |ψ|^2')
plt.legend()
plt.show()