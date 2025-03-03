import numpy as np
import mediapipe as mp  

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose

# Función para calcular el ángulo entre tres puntos
def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Vectores BA y BC
    ba = a - b
    bc = c - b

    # Cálculo del ángulo
    coseno_angulo = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = np.arccos(coseno_angulo)
    return np.degrees(angulo)

# Cargar los puntos clave
keypoints_data = np.load("output/keypoints_data.npy")

# Listas para almacenar los ángulos
angulos_brazos = []
angulos_piernas = []

for keypoints in keypoints_data:
    # Ejemplo: Ángulo del codo (hombro, codo, muñeca)
    hombro_izq = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    codo_izq = keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    muneca_izq = keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value]
    angulo_brazo = calcular_angulo(hombro_izq, codo_izq, muneca_izq)
    angulos_brazos.append(angulo_brazo)

    # Ejemplo: Ángulo de la rodilla (cadera, rodilla, tobillo)
    cadera_izq = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
    rodilla_izq = keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value]
    tobillo_izq = keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    angulo_pierna = calcular_angulo(cadera_izq, rodilla_izq, tobillo_izq)
    angulos_piernas.append(angulo_pierna)

# Guardar los ángulos en archivos CSV
np.savetxt("output/angulos_brazos.csv", angulos_brazos, delimiter=",")
np.savetxt("output/angulos_piernas.csv", angulos_piernas, delimiter=",")

# Detección de errores (ejemplo: ángulo de brazo fuera de rango)
errores = []
for i, angulo in enumerate(angulos_brazos):
    if angulo < 90 or angulo > 160:  # Rango ideal de ángulo de brazo
        errores.append(f"Error en fotograma {i}: Ángulo de brazo fuera de rango ({angulo}°)")

# Guardar errores detectados
with open("output/errores_detectados.txt", "w") as f:
    f.write("\n".join(errores))