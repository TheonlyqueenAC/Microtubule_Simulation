import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters for the spiral
a = 0  # Initial radius
b = 0.1  # Controls spacing between loops
theta_max = 10 * np.pi  # Maximum angle (10 full loops)
theta_step = 0.01  # Step size for theta

# Generate theta values
theta = np.arange(0, theta_max, theta_step)

# Generate the spiral (polar coordinates to Cartesian)
r = a + b * theta  # Radial distance
x = r * np.cos(theta)  # X-coordinates
y = r * np.sin(theta)  # Y-coordinates

# Generate coherence values (e.g., sinusoidal coherence)
k = 2  # Frequency of coherence oscillation
coherence = np.sin(k * theta)

# Plot the spiral with coherence values
plt.figure(figsize=(8, 8))
plt.plot(x, y, label='Spiral Path', color='blue')
plt.scatter(x, y, c=coherence, cmap='coolwarm', label='Coherence')
plt.colorbar(label='Coherence')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Spiral Path with Coherence')
plt.legend()
plt.axis('equal')  # Ensure the spiral looks circular
# Export data
time = theta  # Use theta as the time variable
data = {"Time": time, "Coherence": coherence}
df = pd.DataFrame(data)
df.to_csv("ArchSpir_Coherence_Simulation_Data.csv", index=False)
print("Data exported to ArchSpir_Coherence_Simulation_Data.csv")
plt.show()