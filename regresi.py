import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Data
hours_studied = np.array([7, 4, 8, 5, 7, 3, 7, 8, 5, 4, 8, 8, 3, 6, 5, 2, 8, 6, 2, 5, 1, 6, 9, 1, 3, 7, 4, 9, 3, 5, 3, 7, 5, 9, 7, 2, 4, 9, 2, 9, 5, 2, 4, 7, 8, 3, 1, 4, 2, 8, 4, 2, 6, 6, 4, 6, 2, 2, 4, 8, 7, 9, 8, 5, 2, 5, 8, 9, 9, 1, 9, 7, 9, 8, 1, 8, 8, 3, 1, 8, 3, 3, 1, 5, 7, 9, 7, 9, 8, 2, 1, 7, 7, 8, 5, 3, 8, 6, 3])
sample_papers_practiced = np.array([1, 2, 2, 2, 5, 6, 6, 6, 2, 0, 5, 2, 2, 2, 8, 3, 4, 2, 9, 0, 3, 0, 6, 6, 3, 4, 9, 6, 5, 3, 3, 1, 9, 1, 3, 4, 3, 2, 1, 3, 4, 0, 0, 5, 4, 3, 0, 6, 3, 2, 6, 1, 9, 0, 3, 7, 5, 8, 0, 0, 4, 5, 7, 2, 5, 2, 7, 3, 1, 8, 9, 7, 5, 2, 1, 3, 3, 0, 8, 9, 7, 6, 5, 2, 3, 1, 6, 1, 9, 0, 5, 3, 7, 0, 8, 5, 1, 7, 4])
performance_index = np.array([91.0, 65.0, 45.0, 36.0, 66.0, 61.0, 63.0, 42.0, 61.0, 69.0, 84.0, 73.0, 27.0, 33.0, 68.0, 43.0, 67.0, 70.0, 30.0, 63.0, 71.0, 85.0, 73.0, 57.0, 35.0, 49.0, 66.0, 83.0, 74.0, 74.0, 39.0, 36.0, 58.0, 47.0, 60.0, 74.0, 42.0, 68.0, 32.0, 64.0, 45.0, 39.0, 58.0, 36.0, 71.0, 54.0, 17.0, 54.0, 58.0, 53.0, 27.0, 65.0, 75.0, 52.0, 78.0, 91.0, 33.0, 47.0, 78.0, 38.0, 70.0, 98.0, 87.0, 49.0, 41.0, 71.0, 54.0, 42.0, 91.0, 61.0, 74.0, 54.0, 81.0, 52.0, 65.0, 36.0, 61.0, 35.0, 15.0, 88.0, 45.0, 49.0, 33.0, 60.0, 71.0, 81.0, 67.0, 95.0, 58.0, 29.0, 21.0, 38.0, 60.0, 76.0, 69.0, 30.0, 57.0, 81.0, 36.0])

# Gabungkan fitur dalam satu array
X = np.column_stack((hours_studied, sample_papers_practiced))

# Model regresi linear
linear_model = LinearRegression()
linear_model.fit(X, performance_index)
linear_predictions = linear_model.predict(X)

# Hitung galat RMS
linear_rms_error = np.sqrt(mean_squared_error(performance_index, linear_predictions))
print(f'Linear RMS Error: {linear_rms_error}')

# Fungsi pangkat sederhana
def power_law(X, a, b, c):
    return a * np.power(X[:, 0], b) * np.power(X[:, 1], c)

# Awal tebakan parameter
initial_guess = [1, 1, 1]
params, _ = curve_fit(power_law, X, performance_index, p0=initial_guess)
power_predictions = power_law(X, *params)

# Hitung galat RMS
power_rms_error = np.sqrt(mean_squared_error(performance_index, power_predictions))
print(f'Power Law RMS Error: {power_rms_error}')

# Plot hasil regresi linear
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(performance_index, linear_predictions, color='blue', label='Prediksi Linear')
plt.plot([performance_index.min(), performance_index.max()], [performance_index.min(), performance_index.max()], 'k--', lw=2)
plt.xlabel('Nilai Aktual')
plt.ylabel('Nilai Prediksi')
plt.title('Regresi Linear')
plt.legend()

# Plot hasil regresi pangkat sederhana
plt.subplot(1, 2, 2)
plt.scatter(performance_index, power_predictions, color='red', label='Prediksi Pangkat')
plt.plot([performance_index.min(), performance_index.max()], [performance_index.min(), performance_index.max()], 'k--', lw=2)
plt.xlabel('Nilai Aktual')
plt.ylabel('Nilai Prediksi')
plt.title('Regresi Pangkat Sederhana')
plt.legend()

plt.tight_layout()
plt.show()
