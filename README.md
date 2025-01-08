
Microtubule Simulation

Project Overview

Microtubule Simulation is a Python-based computational model that explores quantum coherence and decoherence within microtubules, inspired by concepts from quantum mechanics and cellular biology. This repository introduces simulations integrating Fibonacci scaling to visualize quantum wave packet evolution and its potential connection to event horizons in biological systems.

The project aims to deepen our understanding of how quantum effects may influence cellular processes and their broader implications for consciousness and coherence.

Features
	•	Gaussian Wave Packet Simulation: Simulates the time evolution of a Gaussian wave packet in a one-dimensional spatial domain.
	•	Fibonacci Scaling: Incorporates the Fibonacci sequence to scale spatial grids, adding a novel layer of biological and physical relevance to the model.
	•	Visualization: Generates real-time visualizations of quantum coherence evolution, showcasing wave packet dispersion and probability density.
	•	Open-Source: Fully reproducible with modular, extensible code for further research and experimentation.

Installation

To run this project, ensure you have Python installed (>= 3.9 recommended) and the required dependencies. Follow these steps:
	1.	Clone the repository:

git clone https://github.com/TheonlyqueenAC/Microtubule_Simulation.git


	2.	Navigate to the project directory:

cd Microtubule_Simulation


	3.	Create a virtual environment (optional but recommended):

python3 -m venv .venv
source .venv/bin/activate   # On Windows, use `.venv\Scripts\activate`


	4.	Install the dependencies:

pip install -r requirements.txt

Usage

Running the Simulations

This repository includes two primary Python scripts:
	1.	microtubule_simulation.py: A general simulation of wave packet evolution in microtubules.
	2.	fibonacci_simulation.py: An advanced version introducing Fibonacci scaling for spatial grid refinement.

To run either script:

python fibonacci_simulation.py

Outputs
	•	Figures: The simulation outputs visualizations saved as PNG files (e.g., fibonacci_coherence_evolution.png).
	•	Data: Intermediate and final results are saved for further analysis.

Files and Structure

├── CONTRIBUTING.md         # Guidelines for contributing to the project
├── README.md               # Project description and usage instructions
├── LICENSE                 # Licensing information
├── microtubule_simulation.py # Core simulation script
├── fibonacci_simulation.py  # Simulation with Fibonacci scaling
├── output/                 # Contains simulation outputs (PDFs, logs, etc.)
├── quantum_coherence_evolution.png # Visualization of wave packet evolution

Background and Inspiration

This project draws from research linking quantum mechanics to biological processes, particularly:
	1.	Hameroff & Penrose (1996): Proposed the Orch-OR theory connecting quantum mechanics to consciousness via microtubules.
	2.	Recent Advances: The role of Fibonacci patterns in biological systems and their potential influence on coherence.

Our simulation explores the hypothesis that microtubules may exhibit event-horizon-like behavior, analogous to astrophysical phenomena, and investigates the influence of Fibonacci scaling on quantum coherence.

Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, open an issue to discuss your ideas.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Inspired by interdisciplinary research spanning quantum physics, cellular biology, and computational modeling. Special thanks to the open-source community for tools enabling this exploration.

References
	1.	Hameroff S, Penrose R. Orchestrated reduction of quantum coherence in brain microtubules: A model for consciousness. Philosophical Transactions of the Royal Society A. 1998;356(1743):1869-96.
	2.	Nanopoulos DV, Mavromatos NE. Quantum coherence in microtubules and implications for consciousness. arXiv preprint.
	3.	Tegmark M. The importance of decoherence in brain processes. Physical Review E. 2000;61(4):4194-206.

Feel free to integrate this into your repository. Let me know if you’d like further refinements!


