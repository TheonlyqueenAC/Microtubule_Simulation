import numpy as np
import matplotlib.pyplot as plt
import csv

# Simulation parameters
time = np.linspace(0, 5, 500)  # Time from 0 to 5 seconds
coherence = np.sin(2 * np.pi * time)  # Example sine wave function

# Save the data to a CSV file
with open("../MicrotubuleSimulation/simulation_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Coherence"])
    for t, c in zip(time, coherence):
        writer.writerow([t, c])

# Plot the data
plt.plot(time, coherence, label="Coherence Over Time")
plt.axhline(y=0.5, color="red", linestyle="--", label="Coherence Threshold")
plt.title("Quantum Coherence Simulation")
plt.xlabel("Time")
plt.ylabel("Coherence")
plt.legend()
plt.show()