import cv2
import numpy as np

def detect_black_frames(video_path, threshold=10, stable_frames=2):
    """
    Détecte les séquences de frames noires dans une vidéo.
    :param video_path: Chemin vers la vidéo à analyser.
    :param threshold: Seuil de luminance pour définir une frame comme noire.
    :param stable_frames: Nombre minimum de frames consécutives pour qu'une séquence soit considérée comme noire.
    :return: Liste des plages de frames noires détectées.
    """
    cap = cv2.VideoCapture(video_path)
    black_frames = []
    start_frame = None
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la frame en niveaux de gris et calculer la luminance moyenne
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_luminance = gray_frame.mean()

        if mean_luminance < threshold:
            if start_frame is None:
                start_frame = frame_index
        else:
            if start_frame is not None:
                duration = frame_index - start_frame
                if duration >= stable_frames:  # Vérifie si la séquence est suffisamment stable
                    black_frames.append((start_frame, frame_index - 1))
                start_frame = None

        frame_index += 1

    # Ajouter une dernière séquence noire si la vidéo se termine en noir
    if start_frame is not None:
        duration = frame_index - start_frame
        if duration >= stable_frames:
            black_frames.append((start_frame, frame_index - 1))

    cap.release()
    return black_frames


def detect_fades(video_path, fade_threshold=10, variation_threshold=3, min_duration=5):
    """
    Détecte les séquences de fondu (fades) dans une vidéo.
    :param video_path: Chemin vers la vidéo à analyser.
    :param fade_threshold: Seuil de luminance pour considérer une variation comme un fondu.
    :param variation_threshold: L'écart maximal de luminance entre deux frames successives.
    :param min_duration: Durée minimale pour qu'une séquence soit considérée comme un fondu.
    :return: Liste des plages de frames fondues détectées.
    """
    cap = cv2.VideoCapture(video_path)
    fades = []
    frame_index = 0
    previous_mean = None
    fade_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la frame en niveaux de gris et calculer la luminance moyenne
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_mean = gray_frame.mean()

        # Détecter les variations progressives de luminance
        if previous_mean is not None:
            diff = abs(current_mean - previous_mean)
            if diff <= variation_threshold:  # La variation est progressive
                if fade_start is None:
                    fade_start = frame_index
            else:
                if fade_start is not None:
                    duration = frame_index - fade_start
                    if duration >= min_duration:  # Vérifie la durée minimale
                        fades.append((fade_start, frame_index - 1))
                    fade_start = None

        previous_mean = current_mean
        frame_index += 1

    # Ajouter une dernière séquence de fondu si la vidéo se termine
    if fade_start is not None:
        duration = frame_index - fade_start
        if duration >= min_duration:
            fades.append((fade_start, frame_index - 1))

    cap.release()
    return fades


def detect_transitions_to_file(video_path, output_file, black_threshold=10, fade_threshold=10, variation_threshold=3, stable_frames=2, min_duration=5):
    """
    Détecte les séquences de black frames et de fades, et écrit les résultats dans un fichier texte.
    :param video_path: Chemin vers la vidéo à analyser.
    :param output_file: Chemin du fichier texte pour écrire les résultats.
    :param black_threshold: Seuil pour détecter les frames noires.
    :param fade_threshold: Seuil pour détecter les fades.
    :param variation_threshold: Variation maximale pour un fade.
    :param stable_frames: Frames minimales pour considérer un black frame.
    :param min_duration: Durée minimale pour considérer un fade.
    """
    # Détecter les frames noires et les fades
    black_frames = detect_black_frames(video_path, threshold=black_threshold, stable_frames=stable_frames)
    fades = detect_fades(video_path, fade_threshold=fade_threshold, variation_threshold=variation_threshold, min_duration=min_duration)

    # Écrire les résultats dans un fichier
    with open(output_file, 'w') as file:
        file.write("Detected Transitions:\n")
        file.write("Black Frames (start, end):\n")
        for start, end in black_frames:
            file.write(f"{start}\t{end}\n")
        
        file.write("\nFades (start, end):\n")
        for start, end in fades:
            file.write(f"{start}\t{end}\n")



# Exemple d'utilisation
video_path = 'Pub_C+_352_288_1.mp4'
output_file = "black_frames_output.txt"
black_frames_detected = detect_black_frames(video_path)
output_file = "transitions_output.txt"
detect_transitions_to_file(
    video_path,
    output_file,
    black_threshold=10,
    fade_threshold=10,
    variation_threshold=3,
    stable_frames=3,
    min_duration=5
)
print("Black frames détectés :", black_frames_detected)
print(f"Les séquences de black frames ont été écrites dans {output_file}")