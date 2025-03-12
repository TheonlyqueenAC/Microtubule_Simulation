import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
import json
from datetime import datetime
import time


class MicrotubuleQuantumSimulator:
    """
    A comprehensive simulator for quantum coherence in microtubular structures,
    with support for Fibonacci scaling and cytokine-induced decoherence.
    """

    def __init__(self, config=None):
        """Initialize the simulator with configuration parameters."""
        # Default configuration
        self.config = {
            # Physical constants
            'hbar': 1.0,  # Reduced Planck's constant
            'm': 1.0,  # Effective mass

            # Geometry parameters
            'L': 10.0,  # Axial length of microtubule
            'R_inner': 7.0,  # Inner radius
            'R_outer': 12.5,  # Outer radius
            'N_r': 100,  # Number of radial grid points
            'N_z': 100,  # Number of axial grid points

            # Simulation parameters
            'dt': 0.01,  # Time step
            'time_steps': 300,  # Total simulation steps
            'frames_to_save': 50,  # Number of frames to save

            # Cytokine parameters
            'V_0': 5.0,  # Peak cytokine potential
            'Gamma_0': 0.05,  # Baseline decoherence rate
            'alpha_c': 0.1,  # Scaling factor for cytokine-induced decoherence
            'D_c': 0.1,  # Cytokine diffusion coefficient
            'kappa_c': 0.01,  # Cytokine degradation rate

            # HIV phase parameters (cytokine levels)
            'hiv_phase': 'none',  # 'none', 'acute', 'art_controlled', 'chronic'

            # Output directories
            'output_dir': 'results',
            'figures_dir': 'figures',
            'data_dir': 'datafiles',

            # Visualization
            'cmap_quantum': 'quantum',
            'cmap_cytokine': 'cytokine'
        }

        # Update with provided configuration
        if config:
            self.config.update(config)

        # Initialize directories
        self._setup_directories()

        # Setup grids and parameters
        self._setup_grids()
        self._setup_parameters()

        # Initialize data storage
        self._initialize_storage()

        # Generate Fibonacci sequence and scaling
        self._setup_fibonacci_scaling()

        # Initialize scientific colormaps
        self._setup_colormaps()

        # Initialize wavefunctions and cytokine field
        self._initialize_quantum_states()

    def _setup_directories(self):
        """Set up output directories for data and figures."""
        # Create base directories
        base_dirs = [
            self.config['output_dir'],
            self.config['figures_dir'],
            self.config['data_dir']
        ]

        # Create PyCharm project directories
        for directory in base_dirs:
            os.makedirs(directory, exist_ok=True)

        # Create local directories on desktop
        desktop_dirs = [
            os.path.expanduser("~/Desktop/microtubule_simulation/figures"),
            os.path.expanduser("~/Desktop/microtubule_simulation/datafiles")
        ]

        for directory in desktop_dirs:
            os.makedirs(directory, exist_ok=True)

    def _setup_grids(self):
        """Set up spatial grids for the simulation."""
        # Extract parameters
        L = self.config['L']
        R_inner = self.config['R_inner']
        R_outer = self.config['R_outer']
        N_r = self.config['N_r']
        N_z = self.config['N_z']

        # Calculate grid spacings
        self.dr = (R_outer - R_inner) / N_r
        self.dz = L / N_z

        # Create grids
        self.r = np.linspace(R_inner, R_outer, N_r)
        self.z = np.linspace(0, L, N_z)
        self.R, self.Z = np.meshgrid(self.r, self.z)

    def _setup_parameters(self):
        """Set up simulation parameters based on configuration."""
        # Extract common parameters
        self.dt = self.config['dt']
        self.time_steps = self.config['time_steps']
        self.frames_to_save = self.config['frames_to_save']
        self.save_interval = self.time_steps // self.frames_to_save

        # HIV phase parameters - set cytokine intensity based on phase
        if self.config['hiv_phase'] == 'acute':
            self.cytokine_intensity = 0.7  # Moderately high
            self.cytokine_variability = 0.3
        elif self.config['hiv_phase'] == 'art_controlled':
            self.cytokine_intensity = 0.4  # Moderate
            self.cytokine_variability = 0.5
        elif self.config['hiv_phase'] == 'chronic':
            self.cytokine_intensity = 0.9  # Very high
            self.cytokine_variability = 0.2
        else:  # 'none' or any other value
            self.cytokine_intensity = 0.1  # Low baseline
            self.cytokine_variability = 0.1

    def _initialize_storage(self):
        """Initialize data storage arrays for simulation results."""
        self.Psi_reg_list = []
        self.Psi_fib_list = []
        self.cytokine_list = []
        self.event_horizon_list = []
        self.variance_reg = []
        self.variance_fib = []
        self.coherence_measure_reg = []
        self.coherence_measure_fib = []
        self.simulation_timestamps = []

    def _setup_fibonacci_scaling(self):
        """Generate Fibonacci sequence and apply scaling to the grid."""
        # Generate sequence
        fib_seq = self._generate_fibonacci_sequence(self.config['N_z'] + 10)
        fib_seq = fib_seq[-self.config['N_z']:]  # Use last N_z elements

        # Calculate golden ratio
        self.phi = (1 + np.sqrt(5)) / 2

        # Calculate Fibonacci ratios
        self.fib_ratios = self._calculate_fibonacci_ratios(fib_seq)

        # Normalize to domain size
        self.fib_scaling = fib_seq / np.max(fib_seq) * self.config['L']

        # Create Fibonacci-scaled grid
        self.Z_fib = np.zeros_like(self.Z)
        for i in range(self.config['N_z']):
            self.Z_fib[i, :] = self.fib_scaling[i]

    def _setup_colormaps(self):
        """Initialize scientific colormaps for visualization."""
        # Quantum density colormap (blue to red)
        self.cmap_quantum = LinearSegmentedColormap.from_list(
            'quantum',
            [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
            N=256
        )

        # Cytokine concentration colormap (black/red to yellow)
        self.cmap_cytokine = LinearSegmentedColormap.from_list(
            'cytokine',
            [(0, 0, 0), (0.5, 0, 0), (1, 0, 0), (1, 0.5, 0), (1, 1, 0)],
            N=256
        )

        # Coherence measure colormap (green to purple)
        self.cmap_coherence = LinearSegmentedColormap.from_list(
            'coherence',
            [(0, 0.1, 0), (0, 0.5, 0), (0, 1, 0), (0.5, 0, 0.5), (0.7, 0, 1), (1, 0, 1)],
            N=256
        )

    def _initialize_quantum_states(self):
        """Initialize wavefunction and cytokine field."""
        # Extract parameters
        L = self.config['L']
        R_outer = self.config['R_outer']

        # Initialize Gaussian wavepacket
        sigma_z = L / 10
        z0 = L / 2

        # Regular grid wavefunction
        self.Psi_reg = np.exp(-0.5 * ((self.Z - z0) / sigma_z) ** 2) * (R_outer - self.R)
        self.Psi_reg /= np.sqrt(np.sum(np.abs(self.Psi_reg) ** 2 * self.r[:, None]) * self.dr * self.dz)

        # Fibonacci-scaled wavefunction
        self.Psi_fib = np.exp(-0.5 * ((self.Z_fib - z0) / sigma_z) ** 2) * (R_outer - self.R)
        self.Psi_fib /= np.sqrt(np.sum(np.abs(self.Psi_fib) ** 2 * self.r[:, None]) * self.dr * self.dz)

        # Initialize cytokine field based on HIV phase
        intensity = self.cytokine_intensity
        variability = self.cytokine_variability

        if self.config['hiv_phase'] == 'chronic':
            # For chronic, high concentration throughout with peaks
            base = intensity * np.ones_like(self.Z) * 0.5
            peaks = intensity * np.exp(-((self.Z - L / 3) ** 2) / (2 * (L / 6) ** 2)) + \
                    intensity * np.exp(-((self.Z - 2 * L / 3) ** 2) / (2 * (L / 6) ** 2))
            self.C = base + (1 - variability) * peaks
        elif self.config['hiv_phase'] == 'acute':
            # For acute, focused high concentration in upper area
            self.C = intensity * np.exp(-((self.Z - L / 3) ** 2) / (2 * (L / 4) ** 2)) * \
                     np.exp(-((self.R - self.config['R_outer']) ** 2) / (
                                 2 * (self.config['R_outer'] - self.config['R_inner']) ** 2))
        elif self.config['hiv_phase'] == 'art_controlled':
            # For ART-controlled, scattered smaller concentrations
            base = intensity * 0.3 * np.ones_like(self.Z)
            spots = intensity * variability * (
                    np.exp(-((self.Z - L / 5) ** 2 + (
                                self.R - (self.config['R_inner'] + self.config['R_outer']) / 2) ** 2) / (
                                       2 * (L / 10) ** 2)) +
                    np.exp(-((self.Z - 3 * L / 5) ** 2 + (
                                self.R - (self.config['R_inner'] + self.config['R_outer']) / 2) ** 2) / (
                                       2 * (L / 10) ** 2)) +
                    np.exp(-((self.Z - 4 * L / 5) ** 2 + (
                                self.R - (self.config['R_inner'] + self.config['R_outer']) / 2) ** 2) / (
                                       2 * (L / 10) ** 2))
            )
            self.C = base + spots
        else:  # 'none' or default
            # Default minimal concentration
            self.C = intensity * np.exp(-((self.Z - L / 2) ** 2) / (2 * (L / 8) ** 2)) * \
                     np.exp(-((self.R - self.config['R_outer']) ** 2) / \
                            (2 * (self.config['R_outer'] - self.config['R_inner']) ** 2))

        # Ensure concentration is bounded
        self.C = np.clip(self.C, 0, 1)

        # Setup potential and decoherence
        self._setup_potential()
        self._setup_decoherence()

        # Store initial state
        self._store_current_state(0)

    def _setup_potential(self):
        """Setup the potential fields for the simulation."""
        # Base potential from tubulin structure
        V_base = 5.0 * np.cos(2 * np.pi * self.Z / self.config['L'])

        # Confinement walls
        V_walls = np.zeros_like(self.R)
        V_walls[self.R < self.config['R_inner']] = 1e6  # Inner wall
        V_walls[self.R > self.config['R_outer']] = 1e6  # Outer wall

        # Set regular and Fibonacci potentials
        self.V_reg = V_base + V_walls
        self.V_fib = V_base + V_walls

        # Add cytokine-induced potential
        self.V_cytokine = self.config['V_0'] * self.C

    def _setup_decoherence(self):
        """Setup decoherence fields."""
        # Cytokine-induced decoherence
        self.Gamma_cytokine = self.config['Gamma_0'] * (1 + self.config['alpha_c'] * self.C)

    def _store_current_state(self, step_num):
        """Store the current state of the simulation for later analysis."""
        # Calculate probability densities
        prob_reg = np.abs(self.Psi_reg) ** 2
        prob_fib = np.abs(self.Psi_fib) ** 2

        # Store wavefunctions, cytokine field, and event horizon
        self.Psi_reg_list.append(prob_reg)
        self.Psi_fib_list.append(prob_fib)
        self.cytokine_list.append(self.C.copy())
        self.event_horizon_list.append(self._calculate_event_horizon(self.Gamma_cytokine))

        # Store variance and coherence metrics
        self.variance_reg.append(self._calculate_variance(self.Psi_reg, self.R, self.Z, self.dr, self.dz))
        self.variance_fib.append(self._calculate_variance(self.Psi_fib, self.R, self.Z_fib, self.dr, self.dz))

        # Calculate coherence measure (inverse of decoherence rate weighted by probability)
        coherence_reg = np.sum(prob_reg * np.exp(-self.Gamma_cytokine)) / np.sum(prob_reg)
        coherence_fib = np.sum(prob_fib * np.exp(-self.Gamma_cytokine)) / np.sum(prob_fib)

        self.coherence_measure_reg.append(coherence_reg)
        self.coherence_measure_fib.append(coherence_fib)

        # Store simulation timestamp
        self.simulation_timestamps.append(step_num * self.dt)

    def _generate_fibonacci_sequence(self, n):
        """Generate Fibonacci sequence up to n terms."""
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i - 1] + fib[i - 2])
        return np.array(fib)

    def _calculate_fibonacci_ratios(self, sequence):
        """Calculate ratios of consecutive Fibonacci numbers."""
        ratios = []
        for i in range(1, len(sequence) - 1):
            if sequence[i] != 0:
                ratios.append(sequence[i + 1] / sequence[i])
        return np.array(ratios)

    def _calculate_event_horizon(self, Gamma):
        """Calculate coherence-preserving boundary based on decoherence field."""
        # Calculate r_h as 1/(1 + mean decoherence rate/scaling factor)
        k = 5.0  # Empirical scaling factor
        r_h = 1 / (1 + np.mean(Gamma, axis=1) / k)

        # Scale to physical dimensions
        r_h_scaled = self.config['R_inner'] + (self.config['R_outer'] - self.config['R_inner']) * r_h / np.max(r_h)

        return r_h_scaled

    def _calculate_variance(self, Psi, R, Z, dr, dz):
        """Calculate spatial variance of the wavefunction."""
        # Calculate probability density
        prob = np.abs(Psi) ** 2

        # Calculate mean positions
        r_mean = np.sum(R * prob * self.r[:, None]) * dr * dz
        z_mean = np.sum(Z * prob * self.r[:, None]) * dr * dz

        # Calculate variances
        var_r = np.sum((R - r_mean) ** 2 * prob * self.r[:, None]) * dr * dz
        var_z = np.sum((Z - z_mean) ** 2 * prob * self.r[:, None]) * dr * dz

        # Return total spatial variance
        return var_r + var_z

    def _evolve_cytokines(self, C, dr, dz, dt):
        """Evolve cytokine concentration field."""
        D_c = self.config['D_c']  # Diffusion coefficient
        kappa_c = self.config['kappa_c']  # Degradation rate

        # Calculate Laplacian using finite differences
        laplacian_r = (np.roll(C, -1, axis=0) - 2 * C + np.roll(C, 1, axis=0)) / dr ** 2
        laplacian_z = (np.roll(C, -1, axis=1) - 2 * C + np.roll(C, 1, axis=1)) / dz ** 2

        # Update concentration field
        C_new = C + dt * (D_c * (laplacian_r + laplacian_z) - kappa_c * C)

        # Ensure concentration stays bounded
        return np.clip(C_new, 0, 1)

    def _evolve_wavefunction(self, Psi, V, Gamma, dr, dz, dt):
        """Evolve wavefunction with potential and decoherence."""
        hbar = self.config['hbar']
        m = self.config['m']

        # Calculate Laplacian using finite differences
        laplacian_r = (np.roll(Psi, -1, axis=0) - 2 * Psi + np.roll(Psi, 1, axis=0)) / dr ** 2
        laplacian_z = (np.roll(Psi, -1, axis=1) - 2 * Psi + np.roll(Psi, 1, axis=1)) / dz ** 2

        # Add radial term for cylindrical coordinates
        for i in range(len(self.r)):
            if self.r[i] > 1e-10:  # Avoid division by zero
                laplacian_r[:, i] += (1 / self.r[i]) * (np.roll(Psi, -1, axis=0) - np.roll(Psi, 1, axis=0))[:, i] / (
                            2 * dr)

        # Time evolution with split-step method
        Psi_half = Psi * np.exp(-0.5j * V * dt / hbar)  # Half-step in potential
        Psi_k = Psi_half - 1j * hbar * dt / (2 * m) * (laplacian_r + laplacian_z)  # Full step in kinetic
        Psi_new = Psi_k * np.exp(-0.5j * V * dt / hbar)  # Half-step in potential

        # Add decoherence term
        Psi_new = Psi_new * np.exp(-Gamma * dt)

        # Normalize
        norm = np.sqrt(np.sum(np.abs(Psi_new) ** 2 * self.r[:, None]) * dr * dz)

        return Psi_new / norm

    def run_simulation(self):
        """Run the full quantum simulation for the specified time steps."""
        print(f"Starting simulation with {self.time_steps} steps...")
        start_time = time.time()

        # Time evolution loop
        for step in range(1, self.time_steps + 1):
            # Update cytokine field
            self.C = self._evolve_cytokines(self.C, self.dr, self.dz, self.dt)

            # Update decoherence field
            self.Gamma_cytokine = self.config['Gamma_0'] * (1 + self.config['alpha_c'] * self.C)

            # Evolution with total potential (base + cytokine)
            total_V_reg = self.V_reg + self.V_cytokine
            total_V_fib = self.V_fib + self.V_cytokine

            # Update wavefunctions
            self.Psi_reg = self._evolve_wavefunction(self.Psi_reg, total_V_reg, self.Gamma_cytokine, self.dr, self.dz,
                                                     self.dt)
            self.Psi_fib = self._evolve_wavefunction(self.Psi_fib, total_V_fib, self.Gamma_cytokine, self.dr, self.dz,
                                                     self.dt)

            # Store results at specified intervals
            if step % self.save_interval == 0 or step == 1:
                self._store_current_state(step)
                progress = step / self.time_steps * 100
                elapsed = time.time() - start_time
                estimated_total = elapsed * self.time_steps / step
                remaining = estimated_total - elapsed

                print(f"Step {step}/{self.time_steps} ({progress:.1f}%) - "
                      f"Time elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")

        print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
        return {
            "timestamps": self.simulation_timestamps,
            "variance_reg": self.variance_reg,
            "variance_fib": self.variance_fib,
            "coherence_reg": self.coherence_measure_reg,
            "coherence_fib": self.coherence_measure_fib
        }

    def save_data(self):
        """Save simulation data to files."""
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hiv_phase = self.config['hiv_phase']
        base_filename = f"microtubule_simulation_{hiv_phase}_{timestamp}"

        # Prepare summary data
        summary_data = {
            "config": self.config,
            "timestamps": self.simulation_timestamps,
            "variance_reg": self.variance_reg,
            "variance_fib": self.variance_fib,
            "coherence_reg": self.coherence_measure_reg,
            "coherence_fib": self.coherence_measure_fib,
            "event_horizon_final": self.event_horizon_list[-1].tolist()
        }

        # Save to PyCharm project directory
        with open(os.path.join(self.config['data_dir'], f"{base_filename}_summary.json"), 'w') as f:
            json.dump(summary_data, f, indent=2)

        # Save to desktop directory
        desktop_path = os.path.expanduser(f"~/Desktop/microtubule_simulation/datafiles/{base_filename}_summary.json")
        with open(desktop_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        # Save numpy arrays for deeper analysis
        np_data = {
            "Psi_reg_final": np.abs(self.Psi_reg) ** 2,
            "Psi_fib_final": np.abs(self.Psi_fib) ** 2,
            "cytokine_final": self.C,
            "event_horizon_final": self.event_horizon_list[-1]
        }

        np.savez(os.path.join(self.config['data_dir'], f"{base_filename}_arrays.npz"), **np_data)
        np.savez(os.path.expanduser(f"~/Desktop/microtubule_simulation/datafiles/{base_filename}_arrays.npz"),
                 **np_data)

        print(f"Data saved to {self.config['data_dir']} and ~/Desktop/microtubule_simulation/datafiles/")
        return base_filename

    def create_animation(self, filename_base=None):
        """Create and save animation of the simulation."""
        if filename_base is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hiv_phase = self.config['hiv_phase']
            filename_base = f"microtubule_animation_{hiv_phase}_{timestamp}"

        # Create figure with subplots
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, height_ratios=[2, 1])

        # Top row: side-by-side visualizations
        ax_reg = fig.add_subplot(gs[0, 0])
        ax_fib = fig.add_subplot(gs[0, 1])
        ax_cyt = fig.add_subplot(gs[0, 2])

        # Bottom row: quantitative comparisons
        ax_var = fig.add_subplot(gs[1, :2])  # Variance comparison
        ax_prof = fig.add_subplot(gs[1, 2])  # Axial profile

        # Initial plots
        contour_reg = ax_reg.contourf(self.Z, self.R, self.Psi_reg_list[0], levels=50, cmap=self.cmap_quantum)
        contour_fib = ax_fib.contourf(self.Z, self.R, self.Psi_fib_list[0], levels=50, cmap=self.cmap_quantum)
        contour_cyt = ax_cyt.contourf(self.Z, self.R, self.cytokine_list[0], levels=50, cmap=self.cmap_cytokine)

        # Add event horizon overlays
        horizon_reg, = ax_reg.plot(self.z, self.event_horizon_list[0], 'r--', linewidth=2, label='Event Horizon')
        horizon_fib, = ax_fib.plot(self.z, self.event_horizon_list[0], 'r--', linewidth=2, label='Event Horizon')

        # Add colorbars
        cbar_reg = plt.colorbar(contour_reg, ax=ax_reg)
        cbar_fib = plt.colorbar(contour_fib, ax=ax_fib)
        cbar_cyt = plt.colorbar(contour_cyt, ax=ax_cyt)

        # Set titles and labels
        ax_reg.set_title('Standard Grid')
        ax_fib.set_title('Fibonacci-Scaled Grid')
        ax_cyt.set_title('Cytokine Perturbation')

        for ax in [ax_reg, ax_fib, ax_cyt]:
            ax.set_xlabel('Axial Position (z)')
            ax.set_ylabel('Radial Position (r)')
            ax.legend()

        # Set up variance comparison plot
        var_reg_line, = ax_var.plot([0], self.variance_reg[:1], 'b-', linewidth=2, label='Standard Grid')
        var_fib_line, = ax_var.plot([0], self.variance_fib[:1], 'r-', linewidth=2, label='Fibonacci-Scaled')
        ax_var.set_xlabel('Time Step')
        ax_var.set_ylabel('Wavefunction Variance')
        ax_var.set_title('Coherence Comparison (Lower Variance = Better Coherence)')
        ax_var.legend()
        ax_var.grid(True)

        # Set up axial profile plot
        z_vals = np.linspace(0, self.config['L'], self.config['N_z'])
        prof_reg, = ax_prof.plot(z_vals, np.mean(self.Psi_reg_list[0], axis=0), 'b-', linewidth=2,
                                 label='Standard Grid')
        prof_fib, = ax_prof.plot(z_vals, np.mean(self.Psi_fib_list[0], axis=0), 'r-', linewidth=2,
                                 label='Fibonacci-Scaled')
        ax_prof.set_xlabel('Axial Position (z)')
        ax_prof.set_ylabel('Mean Probability Density')
        ax_prof.set_title('Axial Probability Profile')
        ax_prof.legend()
        ax_prof.grid(True)

        # Animation update function
        def update(frame):
            # Clear main plots
            for ax in [ax_reg, ax_fib, ax_cyt]:
                ax.clear()

            # Update contour plots
            contour_reg = ax_reg.contourf(self.Z, self.R, self.Psi_reg_list[frame], levels=50, cmap=self.cmap_quantum)
            contour_fib = ax_fib.contourf(self.Z, self.R, self.Psi_fib_list[frame], levels=50, cmap=self.cmap_quantum)
            contour_cyt = ax_cyt.contourf(self.Z, self.R, self.cytokine_list[frame], levels=50, cmap=self.cmap_cytokine)

            # Add event horizon overlays
            ax_reg.plot(self.z, self.event_horizon_list[frame], 'r--', linewidth=2, label='Event Horizon')
            ax_fib.plot(self.z, self.event_horizon_list[frame], 'r--', linewidth=2, label='Event Horizon')

            # Set titles with time indication
            actual_time = self.simulation_timestamps[frame]
            ax_reg.set_title(f'Standard Grid (t={actual_time:.2f})')
            ax_fib.set_title(f'Fibonacci-Scaled Grid (t={actual_time:.2f})')
            ax_cyt.set_title(f'Cytokine Perturbation (t={actual_time:.2f})')

            # Set labels
            for ax in [ax_reg, ax_fib, ax_cyt]:
                ax.set_xlabel('Axial Position (z)')
                ax.set_ylabel('Radial Position (r)')
                ax.legend()

            # Update variance plot
            var_reg_line.set_data(range(frame + 1), self.variance_reg[:frame + 1])
            var_fib_line.set_data(range(frame + 1), self.variance_fib[:frame + 1])
            ax_var.relim()
            ax_var.autoscale_view()

            # Update profile plot
            prof_reg.set_ydata(np.mean(self.Psi_reg_list[frame], axis=0))
            prof_fib.set_ydata(np.mean(self.Psi_fib_list[frame], axis=0))
            ax_prof.relim()
            ax_prof.autoscale_view()

            return [contour_reg, contour_fib, contour_cyt, var_reg_line, var_fib_line, prof_reg, prof_fib]

        # Create animation
        print("Creating animation...")
        ani = FuncAnimation(fig, update, frames=len(self.Psi_reg_list), interval=200, blit=False)

        # Save animation to both directories
        writer = FFMpegWriter(fps=10, metadata=dict(artist='AC Demidont'), bitrate=5000)

        # Project directory path
        project_path = os.path.join(self.config['figures_dir'], f"{filename_base}.mp4")
        ani.save(project_path, writer=writer)

        # Desktop path
        desktop_path = os.path.expanduser(f"~/Desktop/microtubule_simulation/figures/{filename_base}.mp4")
        ani.save(desktop_path, writer=writer)

        plt.close(fig)
        print(f"Animation saved to {project_path} and {desktop_path}")

        return project_path

    def generate_visualizations(self, filename_base=None):
        """Generate static visualizations of simulation results."""
        if filename_base is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hiv_phase = self.config['hiv_phase']
            filename_base = f"microtubule_visualization_{hiv_phase}_{timestamp}"

        # Create comprehensive figure comparing regular and Fibonacci grids
        fig = plt.figure(figsize=(18, 15))

        # 1. Final State Comparison
        plt.subplot(3, 3, 1)
        plt.contourf(self.Z, self.R, self.Psi_reg_list[-1], levels=50, cmap=self.cmap_quantum)
        plt.plot(self.z, self.event_horizon_list[-1], 'r--', linewidth=2)
        plt.title('Final Standard Grid')
        plt.xlabel('Axial Position (z)')
        plt.ylabel('Radial Position (r)')
        plt.colorbar()

        plt.subplot(3, 3, 2)
        plt.contourf(self.Z, self.R, self.Psi_fib_list[-1], levels=50, cmap=self.cmap_quantum)
        plt.plot(self.z, self.event_horizon_list[-1], 'r--', linewidth=2)
        plt.title('Final Fibonacci-Scaled Grid')
        plt.xlabel('Axial Position (z)')
        plt.ylabel('Radial Position (r)')
        plt.colorbar()

        plt.subplot(3, 3, 3)
        plt.contourf(self.Z, self.R, self.cytokine_list[-1], levels=50, cmap=self.cmap_cytokine)
        plt.title(f'Final Cytokine Distribution ({self.config["hiv_phase"]} phase)')
        plt.xlabel('Axial Position (z)')
        plt.ylabel('Radial Position (r)')
        plt.colorbar()

        # 2. Variance Comparison
        plt.subplot(3, 3, 4)
        plt.plot(self.simulation_timestamps, self.variance_reg, 'b-', linewidth=2, label='Standard Grid')
        plt.plot(self.simulation_timestamps, self.variance_fib, 'r-', linewidth=2, label='Fibonacci-Scaled')
        plt.xlabel('Time')
        plt.ylabel('Wavefunction Variance')
        plt.title('Coherence Comparison (Lower = Better)')
        plt.legend()
        plt.grid(True)

        # 3. Coherence Measure Comparison
        plt.subplot(3, 3, 5)
        plt.plot(self.simulation_timestamps, self.coherence_measure_reg, 'b-', linewidth=2, label='Standard Grid')
        plt.plot(self.simulation_timestamps, self.coherence_measure_fib, 'r-', linewidth=2, label='Fibonacci-Scaled')
        plt.xlabel('Time')
        plt.ylabel('Coherence Measure')
        plt.title('Coherence Preservation (Higher = Better)')
        plt.legend()
        plt.grid(True)

        # 4. Improvement Percentage
        plt.subplot(3, 3, 6)
        improvement = (np.array(self.variance_reg) - np.array(self.variance_fib)) / np.array(self.variance_reg) * 100
        plt.plot(self.simulation_timestamps, improvement, 'g-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Improvement (%)')
        plt.title('Fibonacci Coherence Improvement')
        mean_improvement = np.mean(improvement)
        plt.text(0.5, 0.9, f'Mean: {mean_improvement:.2f}%', transform=plt.gca().transAxes,
                 horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.7))
        plt.grid(True)

        # 5. Axial Profile Comparison
        plt.subplot(3, 3, 7)
        plt.plot(self.z, np.mean(self.Psi_reg_list[-1], axis=0), 'b-', linewidth=2, label='Standard Grid')
        plt.plot(self.z, np.mean(self.Psi_fib_list[-1], axis=0), 'r-', linewidth=2, label='Fibonacci-Scaled')
        plt.xlabel('Axial Position (z)')
        plt.ylabel('Mean Probability Density')
        plt.title('Final Axial Profile')
        plt.legend()
        plt.grid(True)

        # 6. Fibonacci Ratio Convergence
        plt.subplot(3, 3, 8)
        n_terms = len(self.fib_ratios) + 2
        fib_seq = self._generate_fibonacci_sequence(n_terms)
        fib_ratios = self._calculate_fibonacci_ratios(fib_seq)

        plt.plot(range(2, len(fib_seq)), fib_ratios, 'o-', color='navy')
        plt.axhline(y=self.phi, color='red', linestyle='--', alpha=0.7, label=f'Golden Ratio (φ ≈ {self.phi:.8f})')
        plt.xlabel('n')
        plt.ylabel('F(n+1) / F(n)')
        plt.title('Fibonacci Ratios to Golden Ratio')
        plt.grid(True)
        plt.legend()

        # 7. Coherence Lifetimes vs Golden Ratio
        plt.subplot(3, 3, 9)
        # Calculate coherence lifetimes (time to drop below 50% initial)
        initial_coherence_reg = self.coherence_measure_reg[0]
        initial_coherence_fib = self.coherence_measure_fib[0]

        coherence_threshold = 0.5

        # Find when coherence drops below threshold
        lifetime_reg = self.simulation_timestamps[-1]  # Default to max time
        lifetime_fib = self.simulation_timestamps[-1]

        for i, (c_reg, c_fib) in enumerate(zip(self.coherence_measure_reg, self.coherence_measure_fib)):
            if c_reg < coherence_threshold * initial_coherence_reg and lifetime_reg == self.simulation_timestamps[-1]:
                lifetime_reg = self.simulation_timestamps[i]
            if c_fib < coherence_threshold * initial_coherence_fib and lifetime_fib == self.simulation_timestamps[-1]:
                lifetime_fib = self.simulation_timestamps[i]

        # Plot lifetimes
        lifetime_ratio = lifetime_fib / lifetime_reg if lifetime_reg > 0 else 0
        lifetime_labels = ['Standard Grid', 'Fibonacci-Scaled']
        lifetime_values = [lifetime_reg, lifetime_fib]

        plt.bar(lifetime_labels, lifetime_values, color=['blue', 'red'])
        plt.axhline(y=lifetime_reg * self.phi, color='green', linestyle='--',
                    label=f'Standard × Golden Ratio')
        plt.ylabel('Coherence Lifetime')
        plt.title(f'Coherence Lifetimes\nRatio: {lifetime_ratio:.2f} vs Phi: {self.phi:.2f}')
        plt.grid(True, axis='y')
        plt.legend()

        # Save figure
        plt.tight_layout()

        # Project directory path
        project_path = os.path.join(self.config['figures_dir'], f"{filename_base}.png")
        plt.savefig(project_path, dpi=300)

        # Desktop path
        desktop_path = os.path.expanduser(f"~/Desktop/microtubule_simulation/figures/{filename_base}.png")
        plt.savefig(desktop_path, dpi=300)

        plt.close(fig)

        print(f"Visualizations saved to {project_path} and {desktop_path}")
        return project_path


# Main script for running simulations
def run_baseline_simulation():
    """Run baseline simulation without cytokine exposure."""
    config = {
        'hiv_phase': 'none',
        'time_steps': 300,
        'frames_to_save': 50,
        'alpha_c': 0.01  # Very low cytokine influence
    }

    simulator = MicrotubuleQuantumSimulator(config)
    results = simulator.run_simulation()
    base_filename = simulator.save_data()
    simulator.create_animation(base_filename)
    simulator.generate_visualizations(base_filename)

    return results


def run_hiv_phase_simulation(phase):
    """Run simulation for a specific HIV phase."""
    if phase not in ['acute', 'art_controlled', 'chronic']:
        raise ValueError(f"Invalid HIV phase: {phase}")

    config = {
        'hiv_phase': phase,
        'time_steps': 300,
        'frames_to_save': 50,
    }

    # Adjust cytokine parameters based on phase
    if phase == 'acute':
        config.update({
            'alpha_c': 0.3,  # Moderate cytokine influence
            'V_0': 3.0  # Moderate cytokine potential
        })
    elif phase == 'art_controlled':
        config.update({
            'alpha_c': 0.2,  # Lower cytokine influence
            'V_0': 2.0  # Lower cytokine potential
        })
    elif phase == 'chronic':
        config.update({
            'alpha_c': 0.5,  # High cytokine influence
            'V_0': 5.0  # High cytokine potential
        })

    simulator = MicrotubuleQuantumSimulator(config)
    results = simulator.run_simulation()
    base_filename = simulator.save_data()
    simulator.create_animation(base_filename)
    simulator.generate_visualizations(base_filename)

    return results


def run_fibonacci_comparison_simulations():
    """Run a series of simulations to compare Fibonacci vs regular grid performance."""
    results = {}

    # Baseline (no cytokines)
    print("\n==== Running Baseline Simulation ====")
    results['baseline'] = run_baseline_simulation()

    # HIV phases
    for phase in ['acute', 'art_controlled', 'chronic']:
        print(f"\n==== Running {phase.upper()} HIV Phase Simulation ====")
        results[phase] = run_hiv_phase_simulation(phase)

    return results


def generate_comparative_visualization(results):
    """Generate visualization comparing results across all simulations."""
    # Create figure for comparison
    fig = plt.figure(figsize=(15, 12))

    # 1. Variance comparison across all phases
    plt.subplot(2, 2, 1)
    phases = list(results.keys())
    phase_labels = {
        'baseline': 'Baseline (No Cytokines)',
        'acute': 'Acute HIV',
        'art_controlled': 'ART-Controlled HIV',
        'chronic': 'Chronic HIV'
    }

    for phase in phases:
        plt.plot(results[phase]['timestamps'], results[phase]['variance_reg'],
                 '--', linewidth=1.5, alpha=0.7, label=f"{phase_labels[phase]} - Standard")
        plt.plot(results[phase]['timestamps'], results[phase]['variance_fib'],
                 '-', linewidth=2, alpha=0.9, label=f"{phase_labels[phase]} - Fibonacci")

    plt.xlabel('Time')
    plt.ylabel('Wavefunction Variance')
    plt.title('Coherence Comparison Across HIV Phases')
    plt.legend()
    plt.grid(True)

    # 2. Coherence comparison
    plt.subplot(2, 2, 2)

    for phase in phases:
        plt.plot(results[phase]['timestamps'], results[phase]['coherence_reg'],
                 '--', linewidth=1.5, alpha=0.7, label=f"{phase_labels[phase]} - Standard")
        plt.plot(results[phase]['timestamps'], results[phase]['coherence_fib'],
                 '-', linewidth=2, alpha=0.9, label=f"{phase_labels[phase]} - Fibonacci")

    plt.xlabel('Time')
    plt.ylabel('Coherence Measure')
    plt.title('Coherence Preservation Across HIV Phases')
    plt.legend()
    plt.grid(True)

    # 3. Improvement percentage comparison
    plt.subplot(2, 2, 3)

    for phase in phases:
        improvement = (np.array(results[phase]['variance_reg']) - np.array(results[phase]['variance_fib'])) / np.array(
            results[phase]['variance_reg']) * 100
        plt.plot(results[phase]['timestamps'], improvement, '-', linewidth=2, label=phase_labels[phase])

        # Calculate mean improvement for this phase
        mean_imp = np.mean(improvement)
        y_pos = 10 + phases.index(phase) * 10  # Stagger text vertically
        plt.text(results[phase]['timestamps'][-1] * 0.6, y_pos,
                 f"{phase_labels[phase]}: {mean_imp:.2f}%",
                 bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel('Time')
    plt.ylabel('Fibonacci Improvement (%)')
    plt.title('Coherence Improvement Across HIV Phases')
    plt.legend()
    plt.grid(True)

    # 4. Coherence half-life comparison
    plt.subplot(2, 2, 4)

    # Calculate coherence half-lives
    half_lives_reg = []
    half_lives_fib = []
    phase_names = []

    for phase in phases:
        # Get initial coherence values
        initial_coherence_reg = results[phase]['coherence_reg'][0]
        initial_coherence_fib = results[phase]['coherence_fib'][0]

        # Find when coherence drops below 50%
        threshold = 0.5

        # Standard grid half-life
        half_life_reg = results[phase]['timestamps'][-1]  # Default to max time
        for i, c in enumerate(results[phase]['coherence_reg']):
            if c < threshold * initial_coherence_reg:
                half_life_reg = results[phase]['timestamps'][i]
                break

        # Fibonacci grid half-life
        half_life_fib = results[phase]['timestamps'][-1]  # Default to max time
        for i, c in enumerate(results[phase]['coherence_fib']):
            if c < threshold * initial_coherence_fib:
                half_life_fib = results[phase]['timestamps'][i]
                break

        half_lives_reg.append(half_life_reg)
        half_lives_fib.append(half_life_fib)
        phase_names.append(phase_labels[phase])

    # Create grouped bar chart
    x = np.arange(len(phase_names))
    width = 0.35

    plt.bar(x - width / 2, half_lives_reg, width, label='Standard Grid')
    plt.bar(x + width / 2, half_lives_fib, width, label='Fibonacci-Scaled')

    plt.xlabel('HIV Phase')
    plt.ylabel('Coherence Half-Life')
    plt.title('Coherence Persistence Comparison')
    plt.xticks(x, phase_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y')

    # Highlight the ratio
    for i, (hr, hf) in enumerate(zip(half_lives_reg, half_lives_fib)):
        ratio = hf / hr if hr > 0 else 0
        plt.text(i, max(hr, hf) + 0.05, f"Ratio: {ratio:.2f}x",
                 ha='center', va='bottom', rotation=0)

    # Save figure
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Project directory path
    project_path = os.path.join("../figures/figures", f"comparative_results_{timestamp}.png")
    plt.savefig(project_path, dpi=300)

    # Desktop path
    desktop_path = os.path.expanduser(f"~/Desktop/microtubule_simulation/figures/comparative_results_{timestamp}.png")
    plt.savefig(desktop_path, dpi=300)

    plt.close(fig)
    print(f"Comparative visualization saved to {project_path} and {desktop_path}")


if __name__ == "__main__":
    # Make sure directories exist
    os.makedirs("../figures/figures", exist_ok=True)
    os.makedirs("../supplementary/datafiles_supplementary", exist_ok=True)
    os.makedirs(os.path.expanduser("~/Desktop/microtubule_simulation/figures"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/Desktop/microtubule_simulation/datafiles"), exist_ok=True)

    print("Starting Microtubule Quantum Coherence Simulations")
    all_results = run_fibonacci_comparison_simulations()
    generate_comparative_visualization(all_results)
    print("All simulations and visualizations completed successfully!")