import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
import sys
import json
from datetime import datetime
import time

# Import the simulator class from the main package
# If the main package is not in the path, add your local path here
sys.path.append(os.path.abspath('.'))
from microtubule_quantum_coherence_full_simulation import MicrotubuleQuantumSimulator


def run_hiv_phase_detailed_simulation(phase):
    """
    Run a detailed simulation for a specific HIV phase with precise time point outputs.

    Args:
        phase (str): HIV phase - 'acute', 'art_controlled', or 'chronic'
    """
    # Configure the simulation based on the phase
    if phase not in ['acute', 'art_controlled', 'chronic']:
        raise ValueError(f"Invalid HIV phase: {phase}")

    # Base configuration with specific time steps/frames
    config = {
        'hiv_phase': phase,
        'time_steps': 400,  # Longer simulation for more detail
        'frames_to_save': 400,  # Save all frames for detailed analysis
        'dt': 0.0075,  # Smaller time step for precision

        # Output directories with phase-specific names
        'output_dir': f'results_{phase}',
        'figures_dir': f'figures_{phase}',
        'data_dir': f'datafiles_{phase}'
    }

    # Specific configurations for different HIV phases
    if phase == 'acute':
        config.update({
            'alpha_c': 0.3,  # Moderate cytokine influence
            'V_0': 3.0,  # Moderate cytokine potential
            'cytokine_intensity': 0.7,  # Concentrated but not overwhelming
            'cytokine_variability': 0.3  # Some variation
        })
    elif phase == 'art_controlled':
        config.update({
            'alpha_c': 0.2,  # Lower cytokine influence
            'V_0': 2.0,  # Lower cytokine potential
            'cytokine_intensity': 0.4,  # Reduced by treatment
            'cytokine_variability': 0.5  # More variation due to treatment
        })
    elif phase == 'chronic':
        config.update({
            'alpha_c': 0.5,  # High cytokine influence
            'V_0': 5.0,  # High cytokine potential
            'cytokine_intensity': 0.9,  # Very high concentration
            'cytokine_variability': 0.2  # More uniform distribution
        })

    print(f"\n=== Starting detailed {phase.upper()} HIV phase simulation ===")
    start_time = time.time()

    # Initialize and run the simulation
    simulator = MicrotubuleQuantumSimulator(config)
    simulator.run_simulation()

    # Save all data
    base_filename = simulator.save_data()

    # Generate visualizations for specific time points (t=0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    target_timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    create_time_point_snapshots(simulator, target_timestamps, phase)

    # Create animation
    animation_path = simulator.create_animation(f"{base_filename}_animation")

    # Generate comprehensive visualization
    viz_path = simulator.generate_visualizations(f"{base_filename}_analysis")

    elapsed_time = time.time() - start_time
    print(f"=== {phase.upper()} HIV phase simulation completed in {elapsed_time:.2f} seconds ===")
    print(f"Results saved to:")
    print(f"  - {animation_path}")
    print(f"  - {viz_path}")

    return simulator


def create_time_point_snapshots(simulator, target_times, phase):
    """
    Create individual snapshots of the simulation at specific time points.

    Args:
        simulator: The MicrotubuleQuantumSimulator instance
        target_times: List of time points to capture
        phase: HIV phase name for file naming
    """
    # Find the closest frames to the target times
    frame_indices = []
    for target_time in target_times:
        # Find closest time
        closest_idx = np.argmin(np.abs(np.array(simulator.simulation_timestamps) - target_time))
        frame_indices.append(closest_idx)

    for idx, time_point in zip(frame_indices, target_times):
        actual_time = simulator.simulation_timestamps[idx]

        # Create figure for this time point
        fig = plt.figure(figsize=(15, 8))

        # Standard grid wavefunction
        plt.subplot(2, 3, 1)
        plt.contourf(simulator.Z, simulator.R, simulator.Psi_reg_list[idx],
                     levels=50, cmap=simulator.cmap_quantum)
        plt.plot(simulator.z, simulator.event_horizon_list[idx], 'r--', linewidth=2)
        plt.title(f'Standard Grid (t={actual_time:.2f})')
        plt.xlabel('Axial Position (z)')
        plt.ylabel('Radial Position (r)')
        plt.colorbar()

        # Fibonacci grid wavefunction
        plt.subplot(2, 3, 2)
        plt.contourf(simulator.Z, simulator.R, simulator.Psi_fib_list[idx],
                     levels=50, cmap=simulator.cmap_quantum)
        plt.plot(simulator.z, simulator.event_horizon_list[idx], 'r--', linewidth=2)
        plt.title(f'Fibonacci-Scaled Grid (t={actual_time:.2f})')
        plt.xlabel('Axial Position (z)')
        plt.ylabel('Radial Position (r)')
        plt.colorbar()

        # Cytokine field
        plt.subplot(2, 3, 3)
        plt.contourf(simulator.Z, simulator.R, simulator.cytokine_list[idx],
                     levels=50, cmap=simulator.cmap_cytokine)
        plt.title(f'Cytokine Field (t={actual_time:.2f})')
        plt.xlabel('Axial Position (z)')
        plt.ylabel('Radial Position (r)')
        plt.colorbar()

        # Variance trajectory up to this point
        plt.subplot(2, 3, 4)
        plt.plot(simulator.simulation_timestamps[:idx + 1], simulator.variance_reg[:idx + 1], 'b-',
                 linewidth=2, label='Standard Grid')
        plt.plot(simulator.simulation_timestamps[:idx + 1], simulator.variance_fib[:idx + 1], 'r-',
                 linewidth=2, label='Fibonacci-Scaled')
        plt.axvline(x=actual_time, color='k', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Wavefunction Variance')
        plt.title('Coherence Comparison')
        plt.legend()
        plt.grid(True)

        # Axial profile comparison
        plt.subplot(2, 3, 5)
        plt.plot(simulator.z, np.mean(simulator.Psi_reg_list[idx], axis=0), 'b-',
                 linewidth=2, label='Standard Grid')
        plt.plot(simulator.z, np.mean(simulator.Psi_fib_list[idx], axis=0), 'r-',
                 linewidth=2, label='Fibonacci-Scaled')
        plt.xlabel('Axial Position (z)')
        plt.ylabel('Mean Probability Density')
        plt.title('Axial Profile')
        plt.legend()
        plt.grid(True)

        # Coherence measures
        plt.subplot(2, 3, 6)
        # Calculate local coherence measure (ratio to initial value)
        coherence_ratio_reg = simulator.coherence_measure_reg[idx] / simulator.coherence_measure_reg[0]
        coherence_ratio_fib = simulator.coherence_measure_fib[idx] / simulator.coherence_measure_fib[0]

        plt.bar(['Standard Grid', 'Fibonacci-Scaled'],
                [coherence_ratio_reg, coherence_ratio_fib],
                color=['blue', 'red'])
        plt.axhline(y=coherence_ratio_reg * simulator.phi, color='green', linestyle='--',
                    label=f'Standard × Golden Ratio')
        plt.ylabel('Coherence Ratio (to initial)')
        plt.title(
            f'Coherence Preservation\nRatio: {coherence_ratio_fib / coherence_ratio_reg:.2f} vs Phi: {simulator.phi:.2f}')
        plt.grid(True, axis='y')
        plt.legend()

        plt.tight_layout()

        # Save to directories
        timestamp = datetime.now().strftime("%Y%m%d")
        snapshot_filename = f"{phase}_t{time_point:.1f}_{timestamp}"

        # Project directory
        project_path = os.path.join(simulator.config['figures_dir'], f"{snapshot_filename}.png")
        plt.savefig(project_path, dpi=300)

        # Desktop directory
        desktop_path = os.path.expanduser(f"~/Desktop/microtubule_simulation/figures/{snapshot_filename}.png")
        plt.savefig(desktop_path, dpi=300)

        plt.close(fig)
        print(f"Snapshot at t={actual_time:.2f} saved to {project_path} and {desktop_path}")


def run_all_phases():
    """Run detailed simulations for all HIV phases."""
    results = {}

    for phase in ['acute', 'art_controlled', 'chronic']:
        print(f"\n==== Running Detailed {phase.upper()} HIV Phase Simulation ====")
        simulator = run_hiv_phase_detailed_simulation(phase)
        results[phase] = {
            "timestamps": simulator.simulation_timestamps,
            "variance_reg": simulator.variance_reg,
            "variance_fib": simulator.variance_fib,
            "coherence_reg": simulator.coherence_measure_reg,
            "coherence_fib": simulator.coherence_measure_fib
        }

    # Generate comparative visualization
    generate_comparative_visualization(results)

    print("\n==== All HIV phase simulations completed successfully! ====")
    return results


def generate_comparative_visualization(results):
    """
    Generate visualization comparing results across all HIV phases.

    Args:
        results: Dictionary containing simulation results for each phase
    """
    # Create figure for comparison
    fig = plt.figure(figsize=(20, 15))

    # Phase labels for plot legends
    phase_labels = {
        'acute': 'Acute HIV',
        'art_controlled': 'ART-Controlled HIV',
        'chronic': 'Chronic HIV'
    }

    # Color maps for consistent visualization
    phase_colors = {
        'acute': {'reg': 'tab:orange', 'fib': 'tab:red'},
        'art_controlled': {'reg': 'tab:blue', 'fib': 'tab:cyan'},
        'chronic': {'reg': 'tab:purple', 'fib': 'tab:brown'}
    }

    # 1. Wavefunction variance comparison across phases
    plt.subplot(3, 2, 1)
    for phase, color_dict in phase_colors.items():
        if phase in results:
            plt.plot(results[phase]['timestamps'], results[phase]['variance_reg'],
                     linestyle='--', color=color_dict['reg'], linewidth=1.5,
                     label=f"{phase_labels[phase]} - Standard")
            plt.plot(results[phase]['timestamps'], results[phase]['variance_fib'],
                     linestyle='-', color=color_dict['fib'], linewidth=2,
                     label=f"{phase_labels[phase]} - Fibonacci")

    plt.xlabel('Time')
    plt.ylabel('Wavefunction Variance')
    plt.title('Coherence Comparison Across HIV Phases')
    plt.legend()
    plt.grid(True)

    # 2. Coherence measure comparison across phases
    plt.subplot(3, 2, 2)
    for phase, color_dict in phase_colors.items():
        if phase in results:
            plt.plot(results[phase]['timestamps'], results[phase]['coherence_reg'],
                     linestyle='--', color=color_dict['reg'], linewidth=1.5,
                     label=f"{phase_labels[phase]} - Standard")
            plt.plot(results[phase]['timestamps'], results[phase]['coherence_fib'],
                     linestyle='-', color=color_dict['fib'], linewidth=2,
                     label=f"{phase_labels[phase]} - Fibonacci")

    plt.xlabel('Time')
    plt.ylabel('Coherence Measure')
    plt.title('Coherence Preservation Across HIV Phases')
    plt.legend()
    plt.grid(True)

    # 3. Normalized variance comparison (to better compare shapes)
    plt.subplot(3, 2, 3)
    for phase, color_dict in phase_colors.items():
        if phase in results:
            # Normalize by initial variance
            norm_var_reg = np.array(results[phase]['variance_reg']) / results[phase]['variance_reg'][0]
            norm_var_fib = np.array(results[phase]['variance_fib']) / results[phase]['variance_fib'][0]

            plt.plot(results[phase]['timestamps'], norm_var_reg,
                     linestyle='--', color=color_dict['reg'], linewidth=1.5,
                     label=f"{phase_labels[phase]} - Standard")
            plt.plot(results[phase]['timestamps'], norm_var_fib,
                     linestyle='-', color=color_dict['fib'], linewidth=2,
                     label=f"{phase_labels[phase]} - Fibonacci")

    plt.xlabel('Time')
    plt.ylabel('Normalized Variance (to initial)')
    plt.title('Normalized Coherence Degradation')
    plt.legend()
    plt.grid(True)

    # 4. Improvement percentage across phases
    plt.subplot(3, 2, 4)
    for phase, color_dict in phase_colors.items():
        if phase in results:
            improvement = (np.array(results[phase]['variance_reg']) - np.array(
                results[phase]['variance_fib'])) / np.array(results[phase]['variance_reg']) * 100
            plt.plot(results[phase]['timestamps'], improvement,
                     color=color_dict['fib'], linewidth=2,
                     label=f"{phase_labels[phase]}")

            # Calculate mean improvement
            mean_imp = np.mean(improvement)
            plt.text(results[phase]['timestamps'][-1] * 0.7,
                     mean_imp,
                     f"{mean_imp:.1f}%",
                     color=color_dict['fib'],
                     bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel('Time')
    plt.ylabel('Fibonacci Improvement (%)')
    plt.title('Coherence Improvement Percentage Across Phases')
    plt.legend()
    plt.grid(True)

    # 5. Coherence lifetimes comparison
    plt.subplot(3, 2, 5)

    # Calculate coherence lifetimes (time to reach 50% of initial)
    lifetimes_reg = []
    lifetimes_fib = []
    phase_names = []

    for phase in results:
        # Get initial coherence values
        initial_coherence_reg = results[phase]['coherence_reg'][0]
        initial_coherence_fib = results[phase]['coherence_fib'][0]

        # Find when coherence drops below 50%
        threshold = 0.5

        # Standard grid lifetime
        lifetime_reg = results[phase]['timestamps'][-1]  # Default to max time
        for i, c in enumerate(results[phase]['coherence_reg']):
            if c < threshold * initial_coherence_reg:
                lifetime_reg = results[phase]['timestamps'][i]
                break

        # Fibonacci grid lifetime
        lifetime_fib = results[phase]['timestamps'][-1]  # Default to max time
        for i, c in enumerate(results[phase]['coherence_fib']):
            if c < threshold * initial_coherence_fib:
                lifetime_fib = results[phase]['timestamps'][i]
                break

        lifetimes_reg.append(lifetime_reg)
        lifetimes_fib.append(lifetime_fib)
        phase_names.append(phase_labels[phase])

    # Create grouped bar chart
    x = np.arange(len(phase_names))
    width = 0.35

    plt.bar(x - width / 2, lifetimes_reg, width, label='Standard Grid', color='gray', alpha=0.7)
    plt.bar(x + width / 2, lifetimes_fib, width, label='Fibonacci-Scaled', color='gold', alpha=0.9)

    plt.xlabel('HIV Phase')
    plt.ylabel('Coherence Lifetime')
    plt.title('Coherence Persistence Comparison')
    plt.xticks(x, phase_names)
    plt.legend()
    plt.grid(True, axis='y')

    # Add improvement ratios
    for i, (reg, fib) in enumerate(zip(lifetimes_reg, lifetimes_fib)):
        ratio = fib / reg if reg > 0 else 0
        plt.text(i, max(reg, fib) + 0.05, f"Ratio: {ratio:.2f}x",
                 ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

    # 6. Golden ratio comparison
    plt.subplot(3, 2, 6)

    # Calculate lifetime ratios
    lifetime_ratios = []
    for reg, fib in zip(lifetimes_reg, lifetimes_fib):
        ratio = fib / reg if reg > 0 else 0
        lifetime_ratios.append(ratio)

    # Calculate golden ratio for reference
    phi = (1 + np.sqrt(5)) / 2

    # Create bar chart comparing to golden ratio
    plt.bar(x, lifetime_ratios, width=0.4, label='Observed Ratio', color='green', alpha=0.7)
    plt.axhline(y=phi, color='red', linestyle='--', label=f'Golden Ratio (φ ≈ {phi:.3f})')

    plt.xlabel('HIV Phase')
    plt.ylabel('Coherence Lifetime Ratio (Fib/Std)')
    plt.title('Coherence Ratio vs Golden Ratio')
    plt.xticks(x, phase_names)
    plt.legend()
    plt.grid(True, axis='y')

    # Add ratio values
    for i, ratio in enumerate(lifetime_ratios):
        deviation = ((ratio - phi) / phi) * 100
        plt.text(i, ratio + 0.05, f"{ratio:.2f}\n({deviation:+.1f}% vs φ)",
                 ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

    # Save the figure
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Project path
    project_path = os.path.join("../figures/figures", f"hiv_phase_comparison_{timestamp}.png")
    plt.savefig(project_path, dpi=300)

    # Desktop path
    desktop_path = os.path.expanduser(f"~/Desktop/microtubule_simulation/figures/hiv_phase_comparison_{timestamp}.png")
    plt.savefig(desktop_path, dpi=300)

    plt.close(fig)
    print(f"Comparative visualization saved to {project_path} and {desktop_path}")


if __name__ == "__main__":
    # Make sure directories exist
    os.makedirs("../figures/figures", exist_ok=True)
    os.makedirs("../supplementary/datafiles_supplementary", exist_ok=True)
    os.makedirs(os.path.expanduser("~/Desktop/microtubule_simulation/figures"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/Desktop/microtubule_simulation/datafiles"), exist_ok=True)

    # Create phase-specific directories
    for phase in ['acute', 'art_controlled', 'chronic']:
        os.makedirs(f"figures_{phase}", exist_ok=True)
        os.makedirs(f"datafiles_{phase}", exist_ok=True)

    print("Starting HIV Phase-Specific Microtubule Quantum Coherence Simulations")
    results = run_all_phases()