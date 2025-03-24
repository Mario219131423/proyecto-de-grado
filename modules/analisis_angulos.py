import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    coseno_angulo = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = np.arccos(coseno_angulo)
    return np.degrees(angulo)

def analizar_angulos(keypoints_data_file="output/keypoints_data.npy"):
    keypoints_data = np.load(keypoints_data_file)
    angulos_brazos = []
    angulos_piernas = []
    
    for keypoints in keypoints_data:
        hombro_izq = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        codo_izq = keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        muneca_izq = keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value]
        angulo_brazo = calcular_angulo(hombro_izq, codo_izq, muneca_izq)
        angulos_brazos.append(angulo_brazo)

        cadera_izq = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
        rodilla_izq = keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value]
        tobillo_izq = keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        angulo_pierna = calcular_angulo(cadera_izq, rodilla_izq, tobillo_izq)
        angulos_piernas.append(angulo_pierna)
        
    np.savetxt("output/angulos_brazos.csv", angulos_brazos, delimiter=",")
    np.savetxt("output/angulos_piernas.csv", angulos_piernas, delimiter=",")
    
    errores = []
    for i, angulo in enumerate(angulos_brazos):
        if angulo < 90 or angulo > 160:
            errores.append(f"Error en fotograma {i}: Ángulo de brazo fuera de rango ({angulo}°)")
    
    with open("output/errores_detectados.txt", "w") as f:
        f.write("\n".join(errores))

if __name__ == "__main__":
    analizar_angulos()
