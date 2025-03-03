import matplotlib.pyplot as plt
import numpy as np

# Cargar los ángulos
angulos_brazos = np.loadtxt("output/angulos_brazos.csv", delimiter=",")
angulos_piernas = np.loadtxt("output/angulos_piernas.csv", delimiter=",")

# Graficar los ángulos
plt.figure(figsize=(12, 6))
plt.plot(angulos_brazos, label="Ángulo de brazo")
plt.plot(angulos_piernas, label="Ángulo de pierna")
plt.xlabel("Fotograma")
plt.ylabel("Ángulo (°)")
plt.title("Análisis de ángulos de brazada y patada")
plt.legend()
plt.grid()
plt.savefig("output/grafico_angulos.png")
plt.show()