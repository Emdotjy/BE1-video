import cv2
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

