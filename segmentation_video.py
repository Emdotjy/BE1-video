import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from utils import afficher_frame_avec_timecode
from utils import convertir_video_en_array

def standardize_frame(frame,min_values,max_values):
    # Normalisation : (frame - min) / (max - min) * 255
    standardized = (frame - min_values) / (max_values - min_values) * 255
    return np.clip(standardized, 0, 255).astype(np.uint8)  

def standardize_video_color(video_frames):
    if len(video_frames) == 0:
        print("Warning: No video frames provided.")
        return video_frames  # Return an empty list
    stacked_frames = np.stack(video_frames, axis=0)
    
    # Compute min and max values across all frames
    min_values = min([np.mean(frame) for frame in stacked_frames])
    max_values = max([np.mean(frame) for frame in stacked_frames])

    # Safeguard against division by zero
    if min_values == max_values:
        print("Warning: min and max values are identical, skipping normalization.")
        return video_frames  # Return frames unprocessed if no range exists

    standardized_frames = [standardize_frame(frame, min_values, max_values) for frame in video_frames]

    return standardized_frames

def is_black(frame):
    """
    Vérifie si une frame vidéo est presque entièrement noire.

    Cette fonction calcule la moyenne des intensités des pixels dans la frame.
    Si la moyenne est inférieure ou égale à 5, la frame est considérée comme noire.

    Parameters:
    -----------
    frame : numpy.ndarray
        Une frame vidéo représentée sous forme d'un tableau NumPy, où chaque pixel est une intensité
        ou une combinaison de canaux (par exemple, BGR ou RGB).

    Returns:
    --------
    bool
        Retourne `True` si la frame est considérée comme noire, sinon `False`. 
    """
    return (np.mean(frame) <= 5)


def segmentation_spot_pub(video):
    """
    Segmente les séquences publicitaires dans une vidéo en fonction des frames noires.

    Cette fonction identifie les segments publicitaires dans une vidéo en détectant 
    les transitions entre des séquences noires et des séquences de contenu vidéo. 
    Elle utilise des critères spécifiques pour déterminer les débuts et les fins 
    des séquences publicitaires.

    Parameters:
    -----------
    video : list of numpy.ndarray
        Liste de frames vidéo, où chaque frame est une image représentée sous forme 
        de tableau NumPy.

    Returns:
    --------
    list of list of numpy.ndarray
        Liste des séquences publicitaires, où chaque séquence est une sous-liste 
        des frames correspondant à une publicité.

    Notes:
    ------
    - La détection repose sur la fonction `is_black` pour identifier les frames noires.
    - Une séquence est considérée comme publicitaire si elle est encadrée par 
      une période de frames noires de 7 à 16 frames.
    - Les frames de début et de fin de chaque séquence sont ajustées pour exclure 
      les séquences complètement noires.
    """
    # Détecter les frames noires dans la vidéo
    black_frames = [is_black(frame) for frame in video]
    debuts = [0]  # Liste des indices de début des séquences publicitaires
    fins = []     # Liste des indices de fin des séquences publicitaires

    # Parcourir les frames pour détecter les transitions entre noir et contenu
    for i in range(len(black_frames) - 1):
        if not black_frames[i] and black_frames[i + 1]:  # Transition vers une séquence noire
            j = 1
            while i + j < len(black_frames) and black_frames[i + j]:  # Compter les frames noires
                j += 1
            # Vérifier si la durée des frames noires est entre 7 et 16 (critère pour une pub)
            if 7 <= j <= 16:
                debuts.append(i + j + 1)  # Début de la prochaine séquence après les frames noires
                fins.append(i + 1)       # Fin de la séquence précédente

    # Supprimer la dernière valeur de début (fin de la séquence de pub)
    debuts.pop()          

    # Supprimer les débuts inutiles (première séquence noire)
    debuts.pop(0)
    fins.pop(0)

    # Extraire les séquences publicitaires à partir des indices de début et de fin
    pub_list = [video[debuts[i]:fins[i]] for i in range(len(debuts))]    

    # Afficher les informations sur les séquences détectées
    for num_pub, pub in enumerate(pub_list):
        print(f"Sequence pub n°{num_pub+1}, démarre à {debuts[num_pub]} et se termine à {fins[num_pub]}")
    print(f"On a détecté {len(pub_list)} publicités.")

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
    print(f"On a détécté {sum(frame_transition)} frames de transition")
    for i in range(len(frame_transition)):
        if frame_transition[i]:
            frame_transition_numbers.append(i)
            #on supprime les éventuelles frames de transitions successives
            j = 1
            while j +i <= len(frame_transition) and frame_transition[i+j]:
                frame_transition[i+j] = False
                j+=1
            if not silent:     
                #afficher_frame_avec_timecode(video, i, fps=24)
                pass
    
    print(frame_transition_numbers)
    #print(f"Moyenne des similarités : {moyenne_similarites:.4f}")
    #print(f"Écart type des similarités : {ecart_type_similarites:.4f}")
    #print(f"Différence (Moyenne - Écart type) : {difference_moyenne_ecart_type:.4f}")
    

    return frame_transition


def detection_transition_list(similarites): 
    
    # calcul moyenne et écart type glissant
    longueur_frame = 100
    moyenne_similarites=np.zeros(len(similarites-longueur_frame))
    ecart_type_similarites=np.zeros(len(similarites-longueur_frame))
    for i in range(len(similarites)):
        moyenne_similarites[i] = np.mean(similarites[i:i+longueur_frame])
        ecart_type_similarites[i] = np.std(similarites[i:i+longueur_frame])
        
    # Différence entre la moyenne et l'écart type
    seuil = 3.5
    difference_moyenne_ecart_type = moyenne_similarites - seuil*ecart_type_similarites
    
    
    frame_transition = similarites < difference_moyenne_ecart_type
    frame_transition_numbers =[]
    epsilon = 2
    print(f"On a détécté {sum(frame_transition)} frames de transition")
    for i in range(len(frame_transition)):
        if frame_transition[i]:
            frame_transition_numbers.append(i+epsilon)
            #on supprime les éventuelles frames de transitions successives
            j = 1
            while j +i <= len(frame_transition) and frame_transition[i+j]:
                frame_transition[i+j] = False
                j+=1
    print(frame_transition_numbers)
    return frame_transition_numbers





'''
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
'''



def comparison(similarites):
    # Vérité Terrain de Pub_C+_352_288_1_
    nombre_changement_plans_verite = 77
    frames_changement_plans_verite = [0, 42, 52, 142, 163, 187, 200, 221, 248, 256, 268, 307, 485, 526, 561, 582, 
                           595, 615, 635, 664, 690, 705, 720, 746, 821, 853, 903, 956, 975, 998, 
                           1027, 1062, 1099, 1120, 1144, 1177, 1220, 1255, 1293, 1335, 1367, 1444, 
                           1582, 1655, 1735, 1812, 1871, 1895, 1909, 1960, 2016, 2106, 2147, 2184,
                           2243, 2487, 2526, 2617, 2688, 2775, 2808, 2829, 2858, 2881, 2917, 2934,
                           2962, 2978, 3011, 3086, 3179, 3287]
    frames_changement_plans_par_notre_code = detection_transition_list(similarites)
    
    # Convertir en ensembles pour faciliter les comparaisons
    verite_set = set(frames_changement_plans_verite)
    detections_set = set(frames_changement_plans_par_notre_code)

    # Calcul des métriques
    true_positives = len(verite_set.intersection(detections_set))  # Changements correctement détectés
    false_positives = len(detections_set - verite_set)  # Détections incorrectes
    false_negatives = len(verite_set - detections_set)  # Changements manqués

    # Calcul de la précision, du rappel et du F1-score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Résultats
    print(f"Précision: {precision:.2f}")
    print(f"Rappel: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")



if __name__ == "__main__":
    video_path = 'pub/Pub_C+_352_288_1.mp4'  # Chemin de la vidéo
    video = convertir_video_en_array(video_path)
    print(f"La vidéo est de longueur {len(video)} et les frames sont de shape {video[0].shape}")
    video_standart = standardize_video_color(video)
    segmentation_spot_pub(video_standart)

    
    similarite_couleur = calculer_similarite_couleur(video)
    video_contour = tracer_contours(video)
    similarite_forme = calculer_similarite_forme(video_contour)
    #play_video(video_contour)
    #detection_transition(similarite_couleur+similarite_forme)
    simil_couleur_normal = (similarite_couleur-np.mean(similarite_couleur))/np.std(similarite_couleur)
    simil_forme_normal = (similarite_forme-np.mean(similarite_forme))/np.std(similarite_forme)
    silent = False
    detection_transition_list(simil_couleur_normal+simil_forme_normal)
    comparison(simil_couleur_normal+simil_forme_normal)
    #trace_similarites(simil_couleur_normal+simil_forme_normal)
    