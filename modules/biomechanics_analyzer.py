import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import csv
from datetime import datetime

# Rango ideal de ángulos para cada articulación (ajusta según necesites)
ANGLE_RANGES = {
    "CODO_IZQ": (80, 150),
    "CODO_DER": (80, 150),
    "HOMBRO_IZQ": (30, 160),
    "HOMBRO_DER": (30, 160),
    "CADERA_IZQ": (80, 140),
    "CADERA_DER": (80, 140),
    "RODILLA_IZQ": (70, 160),
    "RODILLA_DER": (70, 160),
}

class BiomechanicsAnalyzer:
    def __init__(self, show_skeleton=True):
        self.show_skeleton = show_skeleton
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

        # Configuración para grabar video
        self.output_video = None
        self.recording = False
        self.output_path = None

        # Colores para visualización
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }

        # Articulaciones principales para análisis
        self.main_joints = {
            "HOMBRO_IZQ": [self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            "CODO_IZQ": [self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            "MUNECA_IZQ": [self.mp_pose.PoseLandmark.LEFT_WRIST.value],
            "CADERA_IZQ": [self.mp_pose.PoseLandmark.LEFT_HIP.value],
            "RODILLA_IZQ": [self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            "TOBILLO_IZQ": [self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
            "HOMBRO_DER": [self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            "CODO_DER": [self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            "MUNECA_DER": [self.mp_pose.PoseLandmark.RIGHT_WRIST.value],
            "CADERA_DER": [self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            "RODILLA_DER": [self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
            "TOBILLO_DER": [self.mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        }

        # Articulaciones para calcular ángulos
        self.angle_joints = {
            "CODO_IZQ": [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
                self.mp_pose.PoseLandmark.LEFT_WRIST.value
            ],
            "HOMBRO_IZQ": [
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_ELBOW.value
            ],
            "CADERA_IZQ": [
                self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
            ],
            "RODILLA_IZQ": [
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                self.mp_pose.PoseLandmark.LEFT_ANKLE.value
            ],
            "CODO_DER": [
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                self.mp_pose.PoseLandmark.RIGHT_WRIST.value
            ],
            "HOMBRO_DER": [
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW.value
            ],
            "CADERA_DER": [
                self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            ],
            "RODILLA_DER": [
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
            ],
        }

        # Historial de posiciones y velocidades
        self.position_history = {}
        self.velocity_history = {}
        self.last_frame_time = time.time()

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def calculate_velocity(self, current_pos, prev_pos, time_delta):
        if prev_pos is None or time_delta == 0:
            return None
        dx = (current_pos[0] - prev_pos[0]) / time_delta
        dy = (current_pos[1] - prev_pos[1]) / time_delta
        velocity = math.sqrt(dx*dx + dy*dy)
        return velocity

    def start_recording(self, frame_width, frame_height, fps=30):
        if not self.recording:
            output_dir = "biomechanics_recordings"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = os.path.join(output_dir, f"biomechanics_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_video = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
            self.recording = True
            return True
        return False

    def stop_recording(self):
        if self.recording and self.output_video is not None:
            self.output_video.release()
            self.recording = False
            result_path = self.output_path
            self.output_path = None
            self.output_video = None
            return result_path
        return None

    def process_frame(self, frame, time_delta=None, return_angles=False):
        """Procesa un frame y retorna (imagen, diccionario_de_angulos) si return_angles es True."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if time_delta is None:
            current_time = time.time()
            time_delta = current_time - self.last_frame_time
            self.last_frame_time = current_time

        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        angles_dict = {}
        h, w, _ = image.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Dibujar esqueleto y datos solo si show_skeleton es True
            if self.show_skeleton:
                self.mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=self.colors['white'], thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=self.colors['blue'], thickness=2)
                )

            # Procesar articulaciones principales (guardamos coordenadas)
            for joint_name, joint_index in self.main_joints.items():
                idx = joint_index[0]
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                current_pos = (x, y)
                prev_pos = self.position_history.get(joint_name)
                velocity = self.calculate_velocity(current_pos, prev_pos, time_delta)
                self.position_history[joint_name] = current_pos
                if velocity is not None:
                    self.velocity_history[joint_name] = velocity

                if self.show_skeleton:
                    # Dibujar círculo y nombre (solo si se muestra el esqueleto)
                    cv2.circle(image, (x, y), 10, self.colors['red'], -1)
                    cv2.putText(image, joint_name, (x-10, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1, cv2.LINE_AA)
                    if joint_name in self.velocity_history:
                        vel_text = f"{self.velocity_history[joint_name]:.1f} px/s"
                        cv2.putText(image, vel_text, (x-10, y+20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['green'], 1, cv2.LINE_AA)

            # Calcular y mostrar ángulos
            for angle_name, points in self.angle_joints.items():
                if all(landmarks[i].visibility > 0.5 for i in points):
                    angle_val = self.calculate_angle(landmarks[points[0]], landmarks[points[1]], landmarks[points[2]])
                    angles_dict[angle_name] = angle_val
                    if self.show_skeleton:
                        x_c = int(landmarks[points[1]].x * w)
                        y_c = int(landmarks[points[1]].y * h)
                        rango = ANGLE_RANGES.get(angle_name, (0, 180))
                        if angle_val < rango[0] or angle_val > rango[1]:
                            text_color = self.colors['red']
                            text_str = f"{angle_val:.1f}° FUERA DE RANGO"
                        else:
                            text_color = self.colors['yellow']
                            text_str = f"{angle_val:.1f}°"
                        cv2.putText(image, text_str, (x_c+10, y_c+10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

        cv2.putText(image, "ANALISIS BIOMECANICO", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['yellow'], 2, cv2.LINE_AA)
        if self.recording:
            cv2.putText(image, "REC", (w-70, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['red'], 2, cv2.LINE_AA)
            cv2.circle(image, (w-30, 20), 10, self.colors['red'], -1)
            if self.output_video is not None:
                self.output_video.write(image)

        return (image, angles_dict) if return_angles else image

    def close(self):
        if self.recording:
            self.stop_recording()
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()


# =====================================================
#   FUNCIONES GLOBALES PARA PROCESAR VIDEO CON BOTONES
# =====================================================

paused = False   # Estado global de pausa
buttons = {}     # Diccionario global con la posición de botones
analyzer = None  # Referencia global al analizador

def mouse_callback(event, x, y, flags, param):
    """Callback para detectar clic en botones."""
    global paused, analyzer, buttons
    if event == cv2.EVENT_LBUTTONDOWN:
        for btn_name, (x1, y1, x2, y2) in buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if btn_name == "PAUSE":
                    # Cambia el estado de pausa
                    paused = not paused
                elif btn_name == "SKELETON":
                    # Alterna la visualización del esqueleto
                    analyzer.show_skeleton = not analyzer.show_skeleton

def draw_rectangle_with_text(image, x1, y1, x2, y2, text, active=False):
    """Dibuja un botón con un rectángulo y texto."""
    # Colores: Si está activo se pinta con verde, sino gris
    color_rect = (50, 150, 50) if active else (100, 100, 100)
    color_text = (255, 255, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color_rect, -1)
    cv2.putText(image, text, (x1+10, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)

def draw_button(image, btn_name, active=False):
    """Dibuja un botón según su posición en 'buttons'."""
    global buttons
    if btn_name in buttons:
        (x1, y1, x2, y2) = buttons[btn_name]
        if btn_name == "PAUSE":
            text = "Resume" if paused else "Pause"
        elif btn_name == "SKELETON":
            text = "Skeleton ON" if analyzer.show_skeleton else "Skeleton OFF"
        else:
            text = btn_name
        draw_rectangle_with_text(image, x1, y1, x2, y2, text, active)

def process_video_file(video_path, show_skeleton=True):
    """Procesa un video con botones para pausar y mostrar/ocultar esqueleto,
       además de guardar coordenadas y ángulos en CSV."""
    global paused, buttons, analyzer
    paused = False

    if not os.path.exists(video_path):
        print(f"Error: El archivo {video_path} no existe.")
        return

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}.")
        return

    analyzer = BiomechanicsAnalyzer(show_skeleton=show_skeleton)

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    time_delta = 1.0 / fps

    # Configurar salida de video
    output_dir = "biomechanics_processed"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{base_name}_processed_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Configurar archivos CSV para coordenadas y ángulos
    coord_path = os.path.join("output", "coordinates.csv")
    angle_path = os.path.join("output", "angles.csv")
    os.makedirs("output", exist_ok=True)

    with open(coord_path, "w", newline="", encoding="utf-8") as cfile, \
         open(angle_path, "w", newline="", encoding="utf-8") as afile:
        coord_writer = csv.writer(cfile)
        angle_writer = csv.writer(afile)
        coord_writer.writerow(["frame", "joint_name", "x", "y"])
        angle_writer.writerow(["frame", "angle_name", "angle_value", "range_status"])

        window_name = "Video - Controles"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        # Definir posiciones de los botones (puedes ajustar estos valores)
        buttons = {
            "PAUSE":   (10,  10, 130,  50),
            "SKELETON":(150, 10, 300,  50),
        }

        frame_count = 0
        panel_width = 300  # Panel lateral para ángulos
        # Precreamos un panel de ángulos base
        angle_panel = np.full((frame_height, panel_width, 3), 50, dtype=np.uint8)

        print(f"Procesando video: {video_path}")
        while True:
            if not paused:
                ret, frame = capture.read()
                if not ret:
                    break
                processed_frame, angles_dict = analyzer.process_frame(frame, time_delta, return_angles=True)
                out.write(processed_frame)
                frame_count += 1
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"\rProcesando... {progress:.1f}% completado", end="")

                # Guardar coordenadas
                for joint_name, pos in analyzer.position_history.items():
                    x, y = pos
                    coord_writer.writerow([frame_count, joint_name, x, y])
                    cfile.flush()

                # Guardar ángulos
                for angle_name, val in angles_dict.items():
                    rng = ANGLE_RANGES.get(angle_name, (0, 180))
                    status = "FUERA_DE_RANGO" if (val < rng[0] or val > rng[1]) else "OK"
                    angle_writer.writerow([frame_count, angle_name, f"{val:.2f}", status])
                    afile.flush()

                # Dibujar panel de ángulos
                angle_panel[:] = 50
                y0 = 70
                dy = 30
                for i, (name, angle_val) in enumerate(angles_dict.items()):
                    rng = ANGLE_RANGES.get(name, (0, 180))
                    if angle_val < rng[0] or angle_val > rng[1]:
                        status = "FUERA_DE_RANGO"
                        color = (0, 0, 255)  # rojo
                    else:
                        status = "OK"
                        color = (255, 255, 255)
                    text = f"{name}: {angle_val:.1f}° [{status}]"
                    cv2.putText(angle_panel, text, (10, y0 + i * dy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Dibujar botones en el frame
                draw_button(processed_frame, "PAUSE")
                draw_button(processed_frame, "SKELETON")

                # Combinar el frame con el panel de ángulos
                combined = cv2.hconcat([processed_frame, angle_panel])
                cv2.imshow(window_name, combined)
            else:
                # En pausa, se muestra el último frame con botones y panel
                if 'processed_frame' in locals():
                    draw_button(processed_frame, "PAUSE")
                    draw_button(processed_frame, "SKELETON")
                    combined = cv2.hconcat([processed_frame, angle_panel])
                    cv2.imshow(window_name, combined)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("\nProcesamiento interrumpido por el usuario.")
                break

        capture.release()
        out.release()
        cv2.destroyAllWindows()
        analyzer.close()

        print(f"\nVideo procesado guardado en: {output_path}")
        print(f"Coordenadas guardadas en: {coord_path}")
        print(f"Ángulos guardados en: {angle_path}")

def process_webcam(show_skeleton=True):
    """Implementa el modo webcam si lo necesitas (similar a process_video_file)."""
    pass

def process_ip_camera(ip_url, show_skeleton=True):
    global paused, buttons, analyzer
    paused = False

    capture = cv2.VideoCapture(ip_url)
    if not capture.isOpened():
        print(f"Error: No se pudo abrir la cámara IP en {ip_url}")
        return

    analyzer = BiomechanicsAnalyzer(show_skeleton=show_skeleton)

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Por defecto para cámaras IP
    time_delta = 1.0 / fps

    window_name = "IP Camera - Controles"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    buttons = {
        "PAUSE": (10, 10, 130, 50),
        "SKELETON": (150, 10, 300, 50),
    }

    print(f"Conectado a cámara IP: {ip_url}")

    while True:
        if not paused:
            ret, frame = capture.read()
            if not ret:
                print("Error: No se pudo obtener el frame de la cámara IP.")
                break

            processed_frame, angles_dict = analyzer.process_frame(frame, time_delta, return_angles=True)

            # Dibujar panel de ángulos
            panel_width = 300
            angle_panel = np.full((frame_height, panel_width, 3), 50, dtype=np.uint8)
            y0, dy = 70, 30
            for i, (name, angle_val) in enumerate(angles_dict.items()):
                rng = ANGLE_RANGES.get(name, (0, 180))
                status = "FUERA_DE_RANGO" if (angle_val < rng[0] or angle_val > rng[1]) else "OK"
                color = (0, 0, 255) if status == "FUERA_DE_RANGO" else (255, 255, 255)
                text = f"{name}: {angle_val:.1f}° [{status}]"
                cv2.putText(angle_panel, text, (10, y0 + i * dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Dibujar botones en el frame
            draw_button(processed_frame, "PAUSE")
            draw_button(processed_frame, "SKELETON")

            combined = cv2.hconcat([processed_frame, angle_panel])
            cv2.imshow(window_name, combined)
        else:
            # Mantiene último frame visible en pausa
            draw_button(processed_frame, "PAUSE")
            draw_button(processed_frame, "SKELETON")
            combined = cv2.hconcat([processed_frame, angle_panel])
            cv2.imshow(window_name, combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("\nProcesamiento detenido por usuario.")
            break

    capture.release()
    cv2.destroyAllWindows()
    analyzer.close()
