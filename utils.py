import cv2

def play_video_frames(video_frames, fps=24):
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
    - Il faut s'assurer que que les frames dans 'video_frames' sont compatibles avec OpenCV (par exemple, format BGR).
    """
    for frame in video_frames:
        # Afficher chaque frame
        cv2.imshow('Contours épaissis', frame)
        
        # Attendre selon le FPS
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def obtenir_timecode(frame_number, fps=24):
    """
    Convertit un numéro de frame en un timecode exprimé en secondes.

    Cette fonction calcule le temps correspondant à un numéro de frame donné,
    en supposant un nombre d'images par seconde (FPS) constant.

    Parameters:
    -----------
    frame_number : int
        Numéro de la frame à convertir.
    fps : int, optional
        Nombre d'images par seconde de la vidéo. Par défaut, la valeur est 24.

    Returns:
    --------
    float
        Le timecode en secondes correspondant au numéro de frame.

    Notes:
    ------
    - Le calcul est basé sur la formule : 'time_in_seconds = frame_number / fps'.
    - Il faut s'assurer que que 'fps' est un entier positif pour éviter une division par zéro.

    Example:
    --------
    >>> obtenir_timecode(48, fps=24)
    2.0
    >>> obtenir_timecode(120, fps=30)
    4.0
    """
    # Calculer le temps en secondes
    time_in_seconds = frame_number / fps
    return time_in_seconds


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
        Numéro de la frame à afficher. Doit être un index valide dans la liste 'video'.
    fps : int, optional
        Nombre d'images par seconde de la vidéo, utilisé pour calculer le timecode.
        La valeur par défaut est 24.

    Notes:
    ------
    - La fenêtre d'affichage est fermée lorsque l'utilisateur appuie sur une touche.
    - Le timecode est affiché en secondes, avec deux décimales.
    - Il faut s'assurer que que 'frame_number' est un index valide dans la liste 'video'.

    """
    frame = video[frame_number]
    cv2.imshow(f'Frame {frame_number}', frame)
    cv2.waitKey(0)  # Attendre que l'utilisateur appuie sur une touche
    cv2.destroyAllWindows()  # Fermer la fenêtre d'affichage

    # Calculer et afficher le timecode
    timecode = obtenir_timecode(frame_number, fps)
    print(f"Frame numéro : {frame_number}, Timecode : {timecode:.2f} secondes")


def afficher_frame(frame_number):
    """
    Affiche une frame spécifique d'une vidéo

    Cette fonction utilise OpenCV pour afficher une frame donnée d'une vidéo.

    Parameters:
    -----------
    video : list of numpy.ndarray
        Liste représentant les frames de la vidéo, où chaque frame est une image sous 
        forme de tableau NumPy.
    frame_number : int
        Numéro de la frame à afficher. Doit être un index valide dans la liste 'video'.

    Notes:
    ------
    - Il faut s'assurer que que 'frame_number' est un index valide dans la liste 'video'.
    """
    print(f"Frame numéro : {frame_number}")



def convertir_video_en_array(video_path):
    """
    Convertit une vidéo en un tableau de frames (images).

    Cette fonction lit une vidéo à partir du chemin spécifié et extrait chaque frame 
    pour les stocker dans une liste. Chaque frame est une image représentée sous forme 
    de tableau NumPy.

    Parameters:
    -----------
    video_path : str
        Chemin vers le fichier vidéo à convertir.

    Returns:
    --------
    list of numpy.ndarray
        Une liste contenant toutes les frames de la vidéo sous forme de tableaux NumPy.
        Si une erreur survient, la fonction retourne 'None'.

    Notes:
    ------
    - La fonction utilise OpenCV pour lire la vidéo.
    - Il faut s'assurer que que le chemin de la vidéo est valide et que le fichier est lisible.
    - Les frames sont stockées dans l'ordre d'apparition dans la vidéo.

    """
    # Ouvrir le fichier vidéo
    cap = cv2.VideoCapture(video_path)

    # Vérifier si la vidéo a été ouverte correctement
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return
    
    # Lire la première frame pour initialiser
    ret, previous_frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture de la première frame.")
        cap.release()  # Libérer les ressources
        return
    
    # Liste pour stocker les frames de la vidéo
    video = []
    while True:
        # Lire la frame suivante
        ret, frame = cap.read()
        if not ret:  # Arrêter si aucune frame n'est lue (fin de la vidéo)
            break
        
        # Ajouter la frame à la liste
        video.append(frame)
    
    # Libérer les ressources vidéo
    cap.release()

    # Retourner la liste des frames
    return video


def play_video(file):
    """
    Lit et affiche les frames d'un fichier vidéo.

    Cette fonction utilise OpenCV pour lire un fichier vidéo, afficher chaque frame dans une fenêtre, 
    et s'arrête automatiquement après 10 frames ou si l'utilisateur appuie sur la touche 'q'. 

    Parameters:
    -----------
    file : str
        Chemin vers le fichier vidéo à lire.

    Notes:
    ------
    - La fenêtre d'affichage peut être fermée en appuyant sur la touche 'q'.
    - La lecture s'arrête automatiquement après 10 frames pour des raisons de démonstration.
    - Il faut s'assurer que que le chemin du fichier est valide et que le fichier est lisible.

    Example:
    --------
    >>> play_video("chemin/vers/video.mp4")
    """
    # Crée un objet VideoCapture pour lire le fichier vidéo
    cap = cv2.VideoCapture(file)
    
    # Vérifie si le fichier vidéo a été ouvert avec succès
    if not cap.isOpened(): 
        print("Erreur : Impossible d'ouvrir le fichier vidéo.")
        return

    i = 0  # Compteur pour limiter à 10 frames

    # Boucle jusqu'à ce que la vidéo soit terminée ou que l'utilisateur arrête
    while cap.isOpened():
        # Capture chaque frame de la vidéo
        ret, frame = cap.read()
        i += 1
        
        # Arrêter la lecture après 10 frames
        if i == 10:
            break

        # Si une frame est lue avec succès
        if ret:
            # Affiche la frame dans une fenêtre
            cv2.imshow('Frame', frame)
        
            # Quitte la lecture si 'q' est pressé
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            # Si aucune frame n'est lue, termine la boucle
            break

    # Libère l'objet VideoCapture
    cap.release()

    # Ferme toutes les fenêtres d'affichage
    cv2.destroyAllWindows()
