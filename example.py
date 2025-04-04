import numpy as np
import matplotlib.pyplot as plt

# Sample dataset (10 values, sorted)
data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# Calculate quartiles
Q1 = np.percentile(data, 25)  # First quartile (25th percentile)
Q2 = np.percentile(data, 50)  # Median (50th percentile)
Q3 = np.percentile(data, 75)  # Third quartile (75th percentile)

# Print quartile values
print(f"Q1 (25th percentile): {Q1}")
print(f"Q2 (Median, 50th percentile): {Q2}")
print(f"Q3 (75th percentile): {Q3}")

# Create a box plot to visualize quartiles using only Matplotlib
plt.figure(figsize=(8, 5))
plt.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor="skyblue"))
plt.title("Quartiles Visualization (Box Plot)")
plt.xlabel("Values")
plt.grid(True)
plt.savefig("ok.png")
plt.show()
