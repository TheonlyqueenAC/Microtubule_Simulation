import unittest
import numpy as np
from MicrotubuleSimulation.fibonacci_simulation_refactored import (
    generate_fibonacci_sequence,
    normalize_fibonacci_sequence,
    initialize_wave_function,
    evolve_wave_function
)

class TestFibonacciSimulation(unittest.TestCase):
    def test_generate_fibonacci_sequence(self):
        # Test basic Fibonacci sequence
        expected = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
        result = generate_fibonacci_sequence(10)
        np.testing.assert_array_equal(result, expected)

    def test_normalize_fibonacci_sequence(self):
        # Test normalization
        fib_sequence = np.array([0, 1, 1, 2, 3, 5])
        max_length = 10
        result = normalize_fibonacci_sequence(fib_sequence, max_length)
        self.assertTrue(np.max(result) <= max_length)
        self.assertAlmostEqual(np.max(result), max_length, delta=1e-6)

    def test_initialize_wave_function(self):
        # Test wave function initialization
        grid = np.linspace(0, 10, 100)
        center = 5
        width = 1.0
        wave_function = initialize_wave_function(grid, center, width)
        self.assertAlmostEqual(np.sum(np.abs(wave_function) ** 2), 1, delta=1e-6)

    def test_evolve_wave_function(self):
        # Test wave function evolution
        grid = np.linspace(0, 10, 100)
        center = 5
        width = 1.0
        potential = np.zeros_like(grid)
        dx = grid[1] - grid[0]
        dt = 0.01

        psi = initialize_wave_function(grid, center, width)
        evolved_psi = evolve_wave_function(psi, potential, dx, dt)

        # Test shape and basic conservation
        self.assertEqual(len(psi), len(evolved_psi))
        prob_density = np.sum(np.abs(evolved_psi) ** 2) * dx
        self.assertAlmostEqual(prob_density, 1, delta=1e-6)

if __name__ == '__main__':
    unittest.main()