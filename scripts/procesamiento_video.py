import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Cargar el video de prueba
video_path = "data/video_nadador.mp4"
cap = cv2.VideoCapture(video_path)

# Lista para almacenar los puntos clave
keypoints_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    results = pose.process(image)

    # Guardar los puntos clave si se detectan
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append((landmark.x, landmark.y, landmark.z))
        keypoints_data.append(keypoints)

    # Mostrar el video con anotaciones
    annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp.solutions.drawing_utils.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("Video procesado", annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Guardar los puntos clave en un archivo
np.save("output/keypoints_data.npy", np.array(keypoints_data))