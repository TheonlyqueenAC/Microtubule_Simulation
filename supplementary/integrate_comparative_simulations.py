import numpy as np
import matplotlib.pyplot as plt

def fibonacci_sequence(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

def simulate_wavefunction(grid_size, fibonacci_scaling=True):
    x = np.linspace(0, 10, grid_size)
    if fibonacci_scaling:
        x *= np.array(fibonacci_sequence(grid_size)[:grid_size]) / max(fibonacci_sequence(grid_size))
    wavefunction = np.exp(-x**2) * np.sin(2 * np.pi * x)
    return x, wavefunction

x_fib, wf_fib = simulate_wavefunction(100, fibonacci_scaling=True)
x_non_fib, wf_non_fib = simulate_wavefunction(100, fibonacci_scaling=False)

plt.plot(x_fib, wf_fib, label="Fibonacci Scaling")
plt.plot(x_non_fib, wf_non_fib, label="Non-Fibonacci Scaling")
plt.legend()
plt.title("Quantum Coherence Simulation")
plt.show()