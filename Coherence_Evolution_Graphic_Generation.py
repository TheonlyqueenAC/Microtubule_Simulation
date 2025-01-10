import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# File paths for the individual plots
plots = [
    "fibonacci_coherence_evolution.png",
    "quantum_coherence_evolution.png",
    "time_evolution_quantum_coherence.png"
]

# Titles for each subplot
titles = [
    "Fibonacci Coherence Evolution",
    "Quantum Coherence Evolution",
    "Time Evolution of Coherence"
]

# Create a combined plot with subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

for i, ax in enumerate(axs):
    try:
        # Load the image
        img = mpimg.imread(plots[i])
        # Display the image in the subplot
        ax.imshow(img)
        ax.set_title(titles[i])
        ax.axis("off")  # Turn off axes for cleaner display
    except FileNotFoundError:
        print(f"Error: File {plots[i]} not found. Skipping.")

# Adjust layout and save the combined visualization
plt.tight_layout()
output_file = "combined_coherence_evolution.png"
plt.savefig(output_file)
print(f"Combined visualization saved as {output_file}")
plt.show()
