import matplotlib.pyplot as plt
import numpy as np

def plot_angles(angles_arm_path, angles_leg_path, output_image="output/grafico_angulos.png"):
    # Cargar los ángulos desde archivos CSV
    angulos_brazos = np.loadtxt(angles_arm_path, delimiter=",")
    angulos_piernas = np.loadtxt(angles_leg_path, delimiter=",")
    
    plt.figure(figsize=(12, 6))
    plt.plot(angulos_brazos, label="Ángulo de brazo")
    plt.plot(angulos_piernas, label="Ángulo de pierna")
    plt.xlabel("Fotograma")
    plt.ylabel("Ángulo (°)")
    plt.title("Análisis de ángulos de brazada y patada")
    plt.legend()
    plt.grid()
    plt.savefig(output_image)
    plt.show()

if __name__ == "__main__":
    plot_angles("output/angulos_brazos.csv", "output/angulos_piernas.csv")
