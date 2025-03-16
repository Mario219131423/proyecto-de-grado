import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import argparse
import glob
from datetime import datetime

class BiomechanicsAnalyzer:
    def __init__(self):
        # Inicializar MediaPipe Pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Configuración para guardar video
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
        
        # Lista de articulaciones principales para analizar
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
        
        # Almacenamiento de datos históricos para análisis de velocidad y aceleración
        self.position_history = {}
        self.velocity_history = {}
        self.last_frame_time = time.time()
        
    def calculate_angle(self, a, b, c):
        """Calcula el ángulo entre tres puntos (articulaciones)"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        # Cálculo del ángulo utilizando la ley de cosenos
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        # Ajustar el ángulo para que siempre esté entre 0 y 180 grados
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def calculate_velocity(self, current_pos, prev_pos, time_delta):
        """Calcula la velocidad vectorial entre dos posiciones"""
        if prev_pos is None or time_delta == 0:
            return None
        
        dx = (current_pos[0] - prev_pos[0]) / time_delta
        dy = (current_pos[1] - prev_pos[1]) / time_delta
        
        # Magnitud de la velocidad (pixels/segundo)
        velocity = math.sqrt(dx*dx + dy*dy)
        return velocity
    
    def start_recording(self, frame_width, frame_height, fps=30):
        """Inicia la grabación del video"""
        if not self.recording:
            # Crear directorio de salida si no existe
            output_dir = "biomechanics_recordings"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = f"{output_dir}/biomechanics_{timestamp}.mp4"
            
            # Configurar codificador de video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_video = cv2.VideoWriter(
                self.output_path, 
                fourcc, 
                fps, 
                (frame_width, frame_height)
            )
            self.recording = True
            return True
        return False
    
    def stop_recording(self):
        """Detiene la grabación del video"""
        if self.recording and self.output_video is not None:
            self.output_video.release()
            self.recording = False
            result_path = self.output_path
            self.output_path = None
            self.output_video = None
            return result_path
        return None
    
    def process_frame(self, frame, time_delta=None):
        """Procesa un frame para análisis biomecánico"""
        # Convertir frame a RGB para MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Obtener timestamp actual para cálculos de velocidad si no se proporciona
        if time_delta is None:
            current_time = time.time()
            time_delta = current_time - self.last_frame_time
            self.last_frame_time = current_time
        
        # Desactivar escrituras en el frame para procesamiento
        image.flags.writeable = False
        
        # Procesar imagen con MediaPipe Pose
        results = self.pose.process(image)
        
        # Reactivar escrituras en el frame para dibujo
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not results.pose_landmarks:
            # Si no se detectaron landmarks, devolver frame original
            return frame
        
        # Dibujar esqueleto básico
        self.mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=self.colors['white'], thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=self.colors['blue'], thickness=2)
        )
        
        # Landmarks para análisis
        landmarks = results.pose_landmarks.landmark
        
        # Dimensiones del frame
        h, w, _ = image.shape
        
        # Procesar principales articulaciones
        for joint_name, joint_index in self.main_joints.items():
            idx = joint_index[0]
            
            # Extraer coordenadas
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * w)
            
            # Almacenar posición actual
            current_pos = (x, y)
            
            # Recuperar posición anterior para calcular velocidad
            prev_pos = self.position_history.get(joint_name)
            
            # Calcular velocidad si hay datos previos
            velocity = self.calculate_velocity(current_pos, prev_pos, time_delta)
            
            # Actualizar historiales
            self.position_history[joint_name] = current_pos
            if velocity is not None:
                self.velocity_history[joint_name] = velocity
            
            # Dibujar punto de articulación con mayor énfasis
            cv2.circle(image, (x, y), 10, self.colors['red'], -1)
            
            # Mostrar nombre de articulación
            cv2.putText(image, joint_name, (x-10, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1, cv2.LINE_AA)
            
            # Mostrar velocidad si está disponible
            if joint_name in self.velocity_history:
                vel_text = f"V: {self.velocity_history[joint_name]:.1f} px/s"
                cv2.putText(image, vel_text, (x-10, y+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['green'], 1, cv2.LINE_AA)
        
        # Calcular y mostrar ángulos
        for angle_name, points in self.angle_joints.items():
            if all(landmarks[i].visibility > 0.5 for i in points):
                # Obtener los tres puntos para calcular el ángulo
                angle = self.calculate_angle(
                    landmarks[points[0]], 
                    landmarks[points[1]], 
                    landmarks[points[2]]
                )
                
                # Coordenadas del punto central (articulación)
                x = int(landmarks[points[1]].x * w)
                y = int(landmarks[points[1]].y * w)
                
                # Mostrar valor del ángulo
                cv2.putText(image, f"{angle:.1f}°", (x+10, y+10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['yellow'], 2, cv2.LINE_AA)
        
        # Añadir información general
        cv2.putText(image, "ANALISIS BIOMECANICO", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['yellow'], 2, cv2.LINE_AA)
        
        # Indicador de grabación
        if self.recording:
            cv2.putText(image, "REC", (w-70, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['red'], 2, cv2.LINE_AA)
            cv2.circle(image, (w-30, 20), 10, self.colors['red'], -1)
        
        # Si estamos grabando, escribir el frame
        if self.recording and self.output_video is not None:
            self.output_video.write(image)
            
        return image
    
    def close(self):
        """Libera recursos"""
        if self.recording:
            self.stop_recording()
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()


def process_webcam():
    """Función para procesar video en tiempo real desde la webcam"""
    # Inicializar analizador biomecánico
    analyzer = BiomechanicsAnalyzer()
    
    # Intentar abrir la cámara
    capture = cv2.VideoCapture(0)  # 0 para webcam principal
    
    if not capture.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return
    
    # Obtener propiedades de la cámara
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    if fps == 0:  # A veces la cámara reporta 0 FPS
        fps = 30
    
    print(f"Iniciando análisis biomecánico con resolución {frame_width}x{frame_height} a {fps} FPS")
    print("Presiona 'r' para iniciar/detener la grabación")
    print("Presiona 'q' para salir")
    
    try:
        while True:
            # Leer frame de la cámara
            ret, frame = capture.read()
            
            if not ret:
                print("Error: No se pudo leer el frame.")
                break
            
            # Procesar el frame
            processed_frame = analyzer.process_frame(frame)
            
            try:
                # Mostrar el resultado
                cv2.imshow("Análisis Biomecánico", processed_frame)
            except Exception as e:
                print(f"Error al mostrar imagen: {e}")
                print("Ejecutando en modo sin interfaz gráfica...")
                # Iniciar grabación automática si no podemos mostrar imágenes
                if not analyzer.recording:
                    analyzer.start_recording(frame_width, frame_height, fps)
                    print("Grabación iniciada automáticamente...")
                # Salir después de 30 segundos si no hay interfaz gráfica
                if time.time() - analyzer.last_frame_time > 30:
                    break
             # Capturar teclas
            key = cv2.waitKey(1) & 0xFF
            
            # 'r' para iniciar/detener grabación
            if key == ord('r'):
                if analyzer.recording:
                    output_path = analyzer.stop_recording()
                    print(f"Grabación guardada en: {output_path}")
                else:
                    if analyzer.start_recording(frame_width, frame_height, fps):
                        print("Grabación iniciada...")
            # 'q' para salir
            elif key == ord('q'):
                break
    finally:
        # Liberar recursos
        if analyzer.recording:
            output_path = analyzer.stop_recording()
            print(f"Grabación guardada en: {output_path}")
        
        capture.release()
        cv2.destroyAllWindows()
        analyzer.close()
        
        print("Análisis biomecánico finalizado.")
def process_video_file(video_path):
    """Función para procesar un archivo de video pregrabado"""
    if not os.path.exists(video_path):
        print(f"Error: El archivo de video {video_path} no existe.")
        return
    
    # Inicializar analizador biomecánico
    analyzer = BiomechanicsAnalyzer()
    
    # Abrir el video
    capture = cv2.VideoCapture(video_path)
    
    if not capture.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}.")
        return
    
    # Obtener propiedades del video
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Procesando video: {video_path}")
    print(f"Resolución: {frame_width}x{frame_height}, FPS: {fps}, Duración: {duration:.2f} segundos")
    
    # Crear directorio de salida si no existe
    output_dir = "biomechanics_processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generar nombre de archivo de salida
    video_name = os.path.basename(video_path)
    base_name, _ = os.path.splitext(video_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/{base_name}_processed_{timestamp}.mp4"
    
    # Configurar escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Calcular tiempo entre frames para el análisis de velocidad
    time_delta = 1.0 / fps if fps > 0 else 0.033  # Default a 30fps si fps es 0
    
    try:
        frame_count = 0
        while True:
            # Leer frame del video
            ret, frame = capture.read()
            
            if not ret:
                break
            
            # Procesar el frame
            processed_frame = analyzer.process_frame(frame, time_delta)
            
            # Escribir frame procesado al video de salida
            out.write(processed_frame)
            
            # Mostrar progreso
            frame_count += 1
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"\rProcesando... {progress:.1f}% completado ({frame_count}/{total_frames} frames)", end="")
            
            try:
                # Mostrar el resultado (si hay GUI disponible)
                cv2.imshow("Procesando Video", processed_frame)
                
                # Permitir interrumpir con la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcesamiento interrumpido por el usuario.")
                    break
            except Exception as e:
                # Si no podemos mostrar la imagen, solo continuamos procesando
                if frame_count == 1:  # Solo mostrar error la primera vez
                    print(f"\nError al mostrar imagen: {e}")
                    print("Continuando procesamiento sin interfaz gráfica...")
    finally:
        # Liberar recursos
        capture.release()
        out.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        analyzer.close()
        
        print(f"\nVideo procesado guardado en: {output_path}")
        print("Análisis biomecánico finalizado.")


def list_recorded_videos():
    """Lista todos los videos grabados y procesados por el sistema"""
    print("\n=== VIDEOS DISPONIBLES ===")
    
    # Buscar en carpeta de grabaciones en tiempo real
    recordings_dir = "biomechanics_recordings"
    if os.path.exists(recordings_dir):
        recordings = glob.glob(f"{recordings_dir}/*.mp4")
        if recordings:
            print("\nGrabaciones en tiempo real:")
            for i, video in enumerate(recordings, 1):
                print(f"  {i}. {os.path.basename(video)}")
        else:
            print("\nNo hay grabaciones en tiempo real disponibles.")
    
    # Buscar en carpeta de videos procesados
    processed_dir = "biomechanics_processed"
    if os.path.exists(processed_dir):
        processed = glob.glob(f"{processed_dir}/*.mp4")
        if processed:
            print("\nVideos procesados:")
            for i, video in enumerate(processed, len(recordings) + 1 if 'recordings' in locals() else 1):
                print(f"  {i}. {os.path.basename(video)}")
        else:
            print("\nNo hay videos procesados disponibles.")
    
    # Combinar todos los videos encontrados
    all_videos = []
    if os.path.exists(recordings_dir):
        all_videos.extend(glob.glob(f"{recordings_dir}/*.mp4"))
    if os.path.exists(processed_dir):
        all_videos.extend(glob.glob(f"{processed_dir}/*.mp4"))
    
    return all_videos

def play_video(video_path):
    """Reproduce un video utilizando OpenCV"""
    print(f"\nReproduciendo: {os.path.basename(video_path)}")
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return
    
    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Si no se puede detectar, usar 30 FPS por defecto
    
    # Calcular el tiempo de espera entre frames
    frame_delay = int(1000 / fps)
    
    try:
        # Intentar mostrar el video con OpenCV
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            cv2.imshow("Reproducción de Video", frame)
            
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q') or key == 27:  # q o ESC para salir
                break
            elif key == ord(' '):  # Espacio para pausar/continuar
                cv2.waitKey(0)
        
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nError al reproducir con OpenCV: {e}")
        print("Intentando abrir el video con el reproductor predeterminado del sistema...")
        
        # Alternativa: abrir con el reproductor predeterminado del sistema
        cap.release()
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == 'Darwin':  # macOS
                subprocess.call(('open', video_path))
            elif system == 'Windows':
                os.startfile(video_path)
            else:  # Linux y otros
                subprocess.call(('xdg-open', video_path))
                
            print("Video abierto en el reproductor predeterminado del sistema.")
        except Exception as e2:
            print(f"Error al abrir con el reproductor del sistema: {e2}")
            print(f"Ruta del video para abrir manualmente: {os.path.abspath(video_path)}")

def view_recorded_videos():
    """Función para ver los videos grabados previamente"""
    all_videos = list_recorded_videos()
    
    if not all_videos:
        print("\nNo se encontraron videos. Graba o procesa videos primero.")
        return
    
    while True:
        try:
            selection = input("\nIngresa el número del video a reproducir (o 'q' para salir): ")
            
            if selection.lower() == 'q':
                break
                
            index = int(selection) - 1
            if 0 <= index < len(all_videos):
                play_video(all_videos[index])
            else:
                print("Selección inválida. Intenta de nuevo.")
                
        except ValueError:
            print("Por favor ingresa un número válido.")
        except Exception as e:
            print(f"Error: {e}")

# Añadir esta función al archivo principal y modificar el main() así:
def main():
    """Función principal que maneja los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Sistema de Análisis Biomecánico')
    parser.add_argument('--video', type=str, help='Ruta al archivo de video a procesar (opcional)')
    parser.add_argument('--view', action='store_true', help='Ver videos grabados anteriormente')
    args = parser.parse_args()
    
    if args.view:
        # Ver videos grabados
        view_recorded_videos()
    elif args.video:
        # Procesar video desde archivo
        process_video_file(args.video)
    else:
        # Usar webcam en tiempo real
        process_webcam()

if __name__ == "__main__":
    main()  