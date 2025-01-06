import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from utils import afficher_frame_avec_timecode
from utils import play_video
from utils import convertir_video_en_array
from utils import obtenir_timecode
from utils import play_video



def is_black(frame):
    return (np.mean(frame) <= 5)


def segmentation_spot_pub(video):
    black_frames = [is_black(frame) for frame in video]
    debuts = [0]
    fins = []

    for i in range(len(black_frames)-1):
        if not black_frames[i] and black_frames[i+1]:
            j = 1
            while i+j <len(black_frames) and  black_frames[i+j]:
                j+=1 
            if j <= 16 and j>=7:
                debuts.append(i+j+1)
                fins.append(i+1)

    # On enlève la dernière valeur de début qui correspond à la fin de séquence pub 
    debuts.pop()          

    #on de même, on enlève le début
    debuts.pop(0)
    fins.pop(0)
    pub_list = [video[debuts[i]:fins[i]] for i in range(len(debuts))]    
    for num_pub, pub in enumerate(pub_list):
        print(f"Sequence pub n°{num_pub+1}, démarre à {debuts[num_pub]} et se termine à {fins[num_pub]}")

    return pub_list

# Fonction pour calculer l'histogramme d'une frame
def calculer_histogramme(frame):
    hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
    return hist_b, hist_g, hist_r

# Fonction pour calculer la similarité couleur entre deux frames
def calculer_similarite_frames_couleur(frame1, frame2):
    # Calculer les histogrammes des deux frames
    hist_b1, hist_g1, hist_r1 = calculer_histogramme(frame1)
    hist_b2, hist_g2, hist_r2 = calculer_histogramme(frame2)
    
    # Calculer la similarité en utilisant la corrélation pour chaque canal (B, G, R)
    similarite_b = cv2.compareHist(hist_b1, hist_b2, cv2.HISTCMP_INTERSECT)
    similarite_g = cv2.compareHist(hist_g1, hist_g2, cv2.HISTCMP_INTERSECT)
    similarite_r = cv2.compareHist(hist_r1, hist_r2, cv2.HISTCMP_INTERSECT)
    
    # Retourner la moyenne des similarités entre les trois canaux
    similarite_moyenne = (similarite_b + similarite_g + similarite_r) / 3
    return similarite_moyenne


def tracer_contours(video_frames):

    # Noyau pour épaissir les contours
    #kernel = np.ones((5, 5), np.uint8)
    kernel_size = 15  # Taille du noyau, ajustable pour un dégradé plus ou moins large
    gradient_kernel = cv2.getGaussianKernel(kernel_size, sigma=5)
    gradient_kernel = gradient_kernel * gradient_kernel.T 
    contour_frames = []

    for frame in video_frames:
        # Conversion en niveaux de gris et détection des contours

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        
        edges = cv2.Canny(blurred, 100, 200)
        
        
        # Épaissir les contours
        thickened_edges = cv2.dilate(edges, gradient_kernel, iterations=1)
        
        
        # Ajouter la frame de contours épaissis à la liste
        contour_frames.append(thickened_edges)

    return contour_frames


def calculer_similarite_forme(video_contour_frames):
    """
    Calcule la similarité entre chaque paire de frames consécutives dans une vidéo pré-traitée (liste de frames avec contours épaissis).
    
    Paramètres:
    - video_contour_frames (list): Liste de frames de contours épaissis (images en niveaux de gris de même shape).
    
    Retour:
    - list: Liste de similarités entre chaque paire de frames consécutives, avec des valeurs entre -1 et 1.
    """
    similarities = []
    
    for i in range(len(video_contour_frames) - 1):
        # Prendre deux frames consécutives
        frame1 = video_contour_frames[i]
        frame2 = video_contour_frames[i + 1]
        
        # Aplatir les frames pour une comparaison des pixels
        flat1 = frame1.flatten()
        flat2 = frame2.flatten()
        
        # Calcul de la corrélation de Pearson
        correlation = np.corrcoef(flat1, flat2)[0, 1]
        
        # Ajouter la similarité à la liste
        similarities.append(correlation)
    
    return similarities

# Fonction pour analyser la vidéo et calculer les similarités entre les frames successives
def calculer_similarite_couleur(video):

    similarites = []
    current_frame = video[0] 
 
    for frame_number in range(1,len(video)):
        previous_frame = current_frame
        current_frame = video[frame_number]
        # Calculer la similarité entre la frame précédente et la frame actuelle
        similarite = calculer_similarite_frames_couleur(previous_frame, current_frame)
        similarites.append(similarite)
        
        # Passer à la frame suivante

 
    
    # Convertir la liste en numpy array pour calculer les statistiques
    similarites = np.array(similarites)
    return similarites








def detection_transition(similarites, silent): 
    
    # calcul moyenne et écart type glissant
    longueur_frame = 100
    moyenne_similarites=np.zeros(len(similarites-longueur_frame))
    ecart_type_similarites=np.zeros(len(similarites-longueur_frame))
    for i in range(len(similarites)):
        moyenne_similarites[i] = np.mean(similarites[i:i+longueur_frame])
        ecart_type_similarites[i] = np.std(similarites[i:i+longueur_frame])
        
    # Différence entre la moyenne et l'écart type
    seuil = 3
    difference_moyenne_ecart_type = moyenne_similarites - seuil*ecart_type_similarites
    
    
    frame_transition = similarites < difference_moyenne_ecart_type
    frame_transition_numbers =[]
    print(f"on a détécté {sum(frame_transition)} frames de transition")
    for i in range(len(frame_transition)):
        if frame_transition[i]:
            frame_transition_numbers.append(i)
            #on supprime les éventuelles frames de transitions succéssives
            j = 1
            while j +i <= len(frame_transition) and frame_transition[i+j]:
                frame_transition[i+j] = False
                j+=1
            if not silent:     
                afficher_frame_avec_timecode(video, i, fps=24)

    print(f"Moyenne des similarités : {moyenne_similarites:.4f}")
    print(f"Écart type des similarités : {ecart_type_similarites:.4f}")
    print(f"Différence (Moyenne - Écart type) : {difference_moyenne_ecart_type:.4f}")
    

    return moyenne_similarites, ecart_type_similarites, difference_moyenne_ecart_type


def trace_similarites(similarites):
    moyenne_similarites, ecart_type_similarites, difference_moyenne_ecart_type = detection_transition(similarites, silent=False)
    # Tracer les similarités
    plt.plot(similarites, label='Similarité entre frames')
    plt.plot(moyenne_similarites, color='r', linestyle='--', label='Moyenne des similarités')
    plt.plot(difference_moyenne_ecart_type, color='g', linestyle='--', label='Moyenne - Écart type')
    plt.xlabel('Numéro de frame')
    plt.ylabel('Similarité')
    plt.title('Similarité entre frames successives dans la vidéo')
    plt.legend()
    plt.show()
    
    print(f"Moyenne des similarités : {moyenne_similarites:.4f}")
    print(f"Écart type des similarités : {ecart_type_similarites:.4f}")
    print(f"Différence (Moyenne - Écart type) : {difference_moyenne_ecart_type:.4f}")






if __name__ == "__main__":
    video_path = 'Pub_C+_352_288_1.mp4'  # Chemin de la vidéo
    video = convertir_video_en_array(video_path)
    print(f"la vidéo est de longueur {len(video)} et les frames sont de shape {video[0].shape}")
    similarite_couleur = calculer_similarite_couleur(video)
    video_contour = tracer_contours(video)
    similarite_forme = calculer_similarite_forme(video_contour)
    

    #play_video(video_contour)
    #detection_transition(similarite_couleur+similarite_forme)
    simil_couleur_normal = (similarite_couleur-np.mean(similarite_couleur))/np.std(similarite_couleur)
    simil_forme_normal = (similarite_forme-np.mean(similarite_forme))/np.std(similarite_forme)
    silent = False
    detection_transition(simil_couleur_normal+simil_forme_normal, silent)
    trace_similarites(simil_couleur_normal+simil_forme_normal)