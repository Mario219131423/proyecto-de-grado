import argparse
import os
from modules.biomechanics_analyzer import process_webcam, process_ip_camera, process_video_file

def main():
    parser = argparse.ArgumentParser(
        description="SwimMotionApp - Análisis Biomecánico de Nadadores"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--video", type=str, help="Ruta al archivo de video a procesar.")
    group.add_argument("--ipcam", type=str, help="URL de cámara IP para análisis en tiempo real.")
    group.add_argument("--webcam", action="store_true", help="Usar webcam local (por defecto).")
    parser.add_argument("--hide-skeleton", action="store_true", help="Inicia sin mostrar el esqueleto.")
    args = parser.parse_args()

    show_skeleton = not args.hide_skeleton

    if args.video:
        if os.path.exists(args.video):
            process_video_file(args.video, show_skeleton=show_skeleton)
        else:
            print(f"El archivo {args.video} no existe.")
    elif args.ipcam:
        process_ip_camera(args.ipcam, show_skeleton=show_skeleton)
    else:
        process_webcam(show_skeleton=show_skeleton)

if __name__ == "__main__":
    main()
