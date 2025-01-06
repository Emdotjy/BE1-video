import cv2



def play_video(video_frames, fps=24):
    """
    Joue une séquence vidéo à partir d'une liste de frames avec un nombre d'images par seconde (FPS) spécifié.

    Cette fonction utilise OpenCV pour afficher les frames de la vidéo dans une fenêtre appelée 'Contours épaissis'.
    L'utilisateur peut quitter la lecture en appuyant sur la touche 'q'.

    Parameters:
    -----------
    video_frames : list of numpy.ndarray
        Liste des frames vidéo, où chaque frame est une image représentée sous forme de tableau NumPy.
    fps : int, optional
        Nombre d'images par seconde pour la lecture de la vidéo. La valeur par défaut est 24.

    Notes:
    ------
    - La fenêtre d'affichage peut être fermée en appuyant sur la touche 'q'.
    - Assurez-vous que les frames dans `video_frames` sont compatibles avec OpenCV (par exemple, format BGR).
    """
    for frame in video_frames:
        # Afficher chaque frame
        cv2.imshow('Contours épaissis', frame)
        
        # Attendre selon le FPS
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



def afficher_frame_avec_timecode(video, frame_number, fps=24):
    """
    Affiche une frame spécifique d'une vidéo, calcule et affiche son timecode en secondes.

    Cette fonction utilise OpenCV pour afficher une frame donnée d'une vidéo et attend que 
    l'utilisateur appuie sur une touche avant de fermer la fenêtre. Elle calcule également 
    le timecode correspondant en fonction du nombre d'images par seconde (FPS).

    Parameters:
    -----------
    video : list of numpy.ndarray
        Liste représentant les frames de la vidéo, où chaque frame est une image sous 
        forme de tableau NumPy.
    frame_number : int
        Numéro de la frame à afficher. Doit être un index valide dans la liste `video`.
    fps : int, optional
        Nombre d'images par seconde de la vidéo, utilisé pour calculer le timecode.
        La valeur par défaut est 24.

    Notes:
    ------
    - La fenêtre d'affichage est fermée lorsque l'utilisateur appuie sur une touche.
    - Le timecode est affiché en secondes, avec deux décimales.
    - Assurez-vous que `frame_number` est un index valide dans la liste `video`.

    """
    frame = video[frame_number]
    cv2.imshow(f'Frame {frame_number}', frame)
    cv2.waitKey(0)  # Attendre que l'utilisateur appuie sur une touche
    cv2.destroyAllWindows()  # Fermer la fenêtre d'affichage

    # Calculer et afficher le timecode
    timecode = obtenir_timecode(frame_number, fps)
    print(f"Frame numéro : {frame_number}, Timecode : {timecode:.2f} secondes")



def convertir_video_en_array(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return
    
    ret, previous_frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture de la première frame.")
        cap.release()
        return
    
    video = []
    while True:
        # Lire la frame suivante
        ret, frame = cap.read()
        if not ret:
            break
        

        video.append(frame)
    
    cap.release()
    return video


# Fonction pour convertir le numéro de frame en timecode (en secondes)
def obtenir_timecode(frame_number, fps=24):
    # Calculer le temps en secondes
    time_in_seconds = frame_number / fps
    return time_in_seconds

# importing libraries
import cv2
import numpy as np

def play_video(file):

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(file)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video  file")
    i = 0
    # Read until video is completed
    while(cap.isOpened()):
            
        # Capture frame-by-frame
        ret, frame = cap.read()
        i+= 1
        if i ==10:
            break

        if ret == True:
        
            # Display the resulting frame
            cv2.imshow('Frame', frame)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break
        
        # When everything done, release 
        # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()



