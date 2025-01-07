import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from utils import afficher_frame_avec_timecode
from utils import convertir_video_en_array

#region standardize_frame
def standardize_frame(frame, min_values, max_values):
    """
    Standardise les valeurs de pixels d'une image (ou d'une frame) dans une plage de 0 à 255.

    Cette fonction applique une normalisation linéaire pour transformer les valeurs 
    des pixels de la frame d'entrée en fonction des valeurs minimum et maximum fournies. 
    Le résultat est limité à la plage [0, 255] et converti au format entier non signé 8 bits (uint8).

    Paramètres :
    ------------
    frame : numpy.ndarray
        L'image ou frame à standardiser. Il peut s'agir d'un tableau 2D (image en niveaux de gris) 
        ou 3D (image couleur).
    min_values : numpy.ndarray ou scalaire
        Les valeurs minimums utilisées pour la normalisation. Elles doivent correspondre à la 
        forme de 'frame' ou être un scalaire appliqué uniformément à tous les pixels.
    max_values : numpy.ndarray ou scalaire
        Les valeurs maximums utilisées pour la normalisation. Elles doivent correspondre à la 
        forme de 'frame' ou être un scalaire appliqué uniformément à tous les pixels.

    Return :
    ---------
    numpy.ndarray
        La frame standardisée avec des valeurs de pixels dans la plage [0, 255], 
        convertie au format uint8.

    Notes :
    -------
    - Assurez-vous que 'min_values' est strictement inférieur à 'max_values' pour éviter 
      une division par zéro.
    - Toute valeur de pixel en dehors de la plage [0, 255] après normalisation est 
      automatiquement ramenée aux limites les plus proches (0 ou 255).

    Exemple :
    ---------
    >>> import numpy as np
    >>> frame = np.array([[50, 100], [150, 200]])
    >>> min_values = 0
    >>> max_values = 255
    >>> standardized_frame = standardize_frame(frame, min_values, max_values)
    >>> print(standardized_frame)
    [[ 50 100]
     [150 200]]
    """
    # Normalisation de la frame avec la formule : (frame - min) / (max - min) * 255
    standardized = (frame - min_values) / (max_values - min_values) * 255
    
    # Limite les valeurs normalisées à la plage [0, 255] pour assurer des intensités valides
    # Convertit les valeurs au format entier non signé 8 bits (uint8) pour la représentation d'image
    return np.clip(standardized, 0, 255).astype(np.uint8)
 
#region standardize_video_color
def standardize_video_color(video_frames):
    """
    Standardise les couleurs des frames d'une vidéo en normalisant leurs valeurs de pixels.

    Cette fonction calcule les valeurs minimales et maximales moyennes sur toutes 
    les frames de la vidéo pour normaliser chaque frame selon une plage de 0 à 255. 
    Si les frames ne contiennent aucune variation (min == max), la normalisation 
    est ignorée pour éviter des erreurs de calcul.

    Paramètres :
    ------------
    video_frames : list of numpy.ndarray
        Une liste de frames (images) représentant une vidéo. Chaque frame peut 
        être une image en niveaux de gris (2D) ou en couleur (3D).

    Return :
    ---------
    list of numpy.ndarray
        Une liste de frames standardisées avec des valeurs de pixels comprises 
        dans la plage [0, 255], ou la liste d'origine si aucune normalisation n'a 
        été appliquée.

    Notes :
    -------
    - Les valeurs minimales et maximales sont calculées comme les moyennes des 
      valeurs de chaque frame, ce qui peut entraîner une perte de détails si les 
      frames contiennent des variations importantes.
    - Si aucune frame n'est fournie (liste vide), la fonction renvoie une liste vide 
      et affiche un avertissement.
    - Si toutes les frames ont la même moyenne, aucune normalisation n'est effectuée.

    Exemple :
    ---------
    >>> import numpy as np
    >>> video_frames = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(10)]
    >>> standardized_frames = standardize_video_color(video_frames)
    >>> print(len(standardized_frames))  # Affiche le nombre de frames dans la vidéo
    10
    """
    # Vérifie si la liste des frames est vide
    if len(video_frames) == 0:
        print("Avertissement : Aucune frame de vidéo fournie.")
        return video_frames  # Retourne une liste vide
    
    # Empile les frames pour calculer les statistiques globales
    stacked_frames = np.stack(video_frames, axis=0)
    
    # Calcule les valeurs minimales et maximales moyennes sur toutes les frames
    min_values = min([np.mean(frame) for frame in stacked_frames])
    max_values = max([np.mean(frame) for frame in stacked_frames])

    # Vérifie si les valeurs min et max sont identiques pour éviter une division par zéro
    if min_values == max_values:
        print("Avertissement : Les valeurs min et max sont identiques, normalisation ignorée.")
        return video_frames  # Retourne les frames d'origine si aucune plage n'existe

    # Applique la normalisation à chaque frame
    standardized_frames = [standardize_frame(frame, min_values, max_values) for frame in video_frames]

    return standardized_frames

#region is_black
def is_black(frame):
    """
    Vérifie si une frame vidéo est presque entièrement noire.

    Cette fonction calcule la moyenne des intensités des pixels dans la frame.
    Si la moyenne est inférieure ou égale à 5, la frame est considérée comme noire.

    Parameters:
    -----------
    frame : numpy.ndarray
        Une frame vidéo représentée sous forme d'un tableau NumPy, où chaque pixel est une intensité
        ou une combinaison de canaux.

    Returns:
    --------
    bool
        Retourne 'True' si la frame est considérée comme noire, sinon 'False'. 
    """
    return (np.mean(frame) <= 5)

#region segmentation_spot_pub
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
    - La détection repose sur la fonction 'is_black' pour identifier les frames noires.
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

#region calculer_histogramme
# Fonction pour calculer l'histogramme d'une frame
def calculer_histogramme(frame):
    """
    Calcule les histogrammes des canaux de couleur (Rouge, Vert, Bleu) d'une image.

    Cette fonction utilise la bibliothèque OpenCV pour calculer les histogrammes 
    des valeurs d'intensité des pixels pour chaque canal de couleur dans une image 
    au format RGB (Rouge, Vert, Bleu).

    Paramètres :
    ------------
    frame : numpy.ndarray
        L'image pour laquelle les histogrammes doivent être calculés. Elle doit être 
        au format RGB (image couleur) avec 3 canaux.

    Return :
    ---------
    tuple of numpy.ndarray
        Un tuple contenant trois histogrammes :
        - hist_b : Histogramme pour le canal Bleu.
        - hist_g : Histogramme pour le canal Vert.
        - hist_r : Histogramme pour le canal Rouge.

    Notes :
    -------
    - Chaque histogramme représente la distribution des intensités des pixels 
      dans la plage [0, 255] pour le canal correspondant.
    - Les histogrammes sont calculés sans masque (couvrant toute l'image).

    Exemple :
    ---------
    >>> import cv2
    >>> frame = cv2.imread("image.jpg")  # Charge une image au format BGR
    >>> hist_b, hist_g, hist_r = calculer_histogramme(frame)
    >>> print(hist_b.shape, hist_g.shape, hist_r.shape)
    (256, 1) (256, 1) (256, 1)
    """
    # Calcul de l'histogramme pour le canal Bleu (indice 0)
    hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
    
    # Calcul de l'histogramme pour le canal Vert (indice 1)
    hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
    
    # Calcul de l'histogramme pour le canal Rouge (indice 2)
    hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
    
    return hist_b, hist_g, hist_r

#region calculer_similarite_frames_couleur
# Fonction pour calculer la similarité couleur entre deux frames
def calculer_similarite_frames_couleur(frame1, frame2):
    """
    Calcule la similarité des couleurs entre deux frames en comparant leurs histogrammes.

    Cette fonction utilise la méthode de corrélation d'histogrammes pour évaluer la similarité 
    entre deux frames (images) au format RGB (Rouge, Vert, Bleu). La similarité est calculée 
    indépendamment pour chaque canal de couleur, puis la moyenne des similarités est retournée.

    Paramètres :
    ------------
    frame1 : numpy.ndarray
        La première frame à comparer. Elle doit être au format BGR avec 3 canaux.
    frame2 : numpy.ndarray
        La deuxième frame à comparer. Elle doit être au format BGR avec 3 canaux.

    Return :
    ---------
    float
        La similarité moyenne des couleurs entre les deux frames, calculée sur les 
        trois canaux (Rouge, Vert, Bleu).

    Notes :
    -------
    - La méthode utilisée pour la comparaison est 'cv2.HISTCMP_INTERSECT', qui mesure 
      le degré de correspondance entre les histogrammes.
    - Une valeur de similarité plus élevée indique une plus grande correspondance 
      des couleurs entre les frames.

    Exemple :
    ---------
    >>> import cv2
    >>> frame1 = cv2.imread("image1.jpg")
    >>> frame2 = cv2.imread("image2.jpg")
    >>> similarite = calculer_similarite_frames_couleur(frame1, frame2)
    >>> print(f"Similarité moyenne : {similarite:.2f}")
    """
    # Calculer les histogrammes pour les trois canaux (B, G, R) de la première frame
    hist_b1, hist_g1, hist_r1 = calculer_histogramme(frame1)
    # Calculer les histogrammes pour les trois canaux (B, G, R) de la deuxième frame
    hist_b2, hist_g2, hist_r2 = calculer_histogramme(frame2)
    
    # Comparer les histogrammes des canaux Bleu, Vert et Rouge en utilisant la corrélation
    similarite_b = cv2.compareHist(hist_b1, hist_b2, cv2.HISTCMP_INTERSECT)
    similarite_g = cv2.compareHist(hist_g1, hist_g2, cv2.HISTCMP_INTERSECT)
    similarite_r = cv2.compareHist(hist_r1, hist_r2, cv2.HISTCMP_INTERSECT)
    
    # Calculer la moyenne des similarités pour les trois canaux
    similarite_moyenne = (similarite_b + similarite_g + similarite_r) / 3
    
    return similarite_moyenne


#region tracer_contour
def tracer_contours(video_frames):
    """
    Détecte et trace les contours sur une série de frames d'une vidéo.

    Cette fonction convertit chaque frame en niveaux de gris, applique un flou gaussien 
    pour réduire le bruit, détecte les contours à l'aide de l'algorithme Canny, puis 
    épaissit ces contours en utilisant une dilatation avec un noyau gaussien personnalisé.

    Paramètres :
    ------------
    video_frames : list of numpy.ndarray
        Une liste de frames (images) de la vidéo au format BGR.

    Retourne :
    ---------
    list of numpy.ndarray
        Une liste de frames où seuls les contours détectés sont visibles, après 
        épaississement.

    Notes :
    -------
    - Le noyau gaussien utilisé pour épaissir les contours peut être ajusté via sa taille 
      ('kernel_size') et son écart-type ('sigma').
    - Cette fonction est utile pour extraire des contours saillants dans une vidéo, par exemple 
      pour un traitement d'image ou un effet visuel.

    Exemple :
    ---------
    >>> import cv2
    >>> video_frames = [cv2.imread(f"frame_{i}.jpg") for i in range(10)]
    >>> contour_frames = tracer_contours(video_frames)
    >>> for i, contour_frame in enumerate(contour_frames):
    >>>     cv2.imwrite(f"contour_frame_{i}.jpg", contour_frame)
    """
    # Définir la taille du noyau et créer un noyau gaussien pour épaissir les contours
    kernel_size = 2  # Taille du noyau ajustable pour un épaississement variable
    gradient_kernel = cv2.getGaussianKernel(kernel_size, sigma=5)
    gradient_kernel = gradient_kernel * gradient_kernel.T  # Transformation en noyau 2D

    contour_frames = []  # Liste pour stocker les frames avec contours

    for frame in video_frames:
        # Conversion de la frame en niveaux de gris
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Réduction du bruit avec un flou gaussien
        blurred = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        
        # Détection des contours avec l'algorithme Canny
        edges = cv2.Canny(blurred, 100, 200)
        
        # Épaissir les contours avec une dilatation utilisant le noyau gaussien
        thickened_edges = cv2.dilate(edges, gradient_kernel, iterations=1)
        
        # Ajouter la frame de contours épaissis à la liste
        contour_frames.append(thickened_edges)

    return contour_frames


#region calculer_similarite_forme
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

#region calculer_similarite_couleur
# Fonction pour analyser la vidéo et calculer les similarités entre les frames successives
def calculer_similarite_couleur(video):
    """
    Analyse une vidéo et calcule les similarités de couleur entre les frames successives.

    Cette fonction parcourt une vidéo (sous forme de liste de frames) et calcule la 
    similarité de couleur entre chaque frame et la suivante, en utilisant une méthode 
    de comparaison d'histogrammes. Les similarités sont ensuite renvoyées sous forme 
    de tableau numpy.

    Paramètres :
    ------------
    video : list of numpy.ndarray
        Une liste de frames représentant une vidéo. Chaque frame doit être au format BGR.

    Return :
    ---------
    numpy.ndarray
        Un tableau numpy contenant les similarités entre les frames successives.

    Notes :
    -------
    - La méthode de similarité utilisée dépend de la fonction 'calculer_similarite_frames_couleur'.
    - Les valeurs de similarité sont calculées pour chaque paire successive de frames dans la vidéo.
    - Si la vidéo ne contient qu'une seule frame ou est vide, la liste des similarités sera vide.

    Exemple :
    ---------
    >>> import cv2
    >>> video = [cv2.imread(f"frame_{i}.jpg") for i in range(10)]
    >>> similarites = calculer_similarite_couleur(video)
    >>> print(f"Similarités entre frames successives : {similarites}")
    """
    similarites = []  # Liste pour stocker les similarités entre frames successives
    
    # Initialiser la frame courante avec la première frame de la vidéo
    current_frame = video[0] 

    # Parcourir toutes les frames de la vidéo à partir de la deuxième
    for frame_number in range(1, len(video)):
        previous_frame = current_frame  # Frame précédente
        current_frame = video[frame_number]  # Frame actuelle
        
        # Calculer la similarité entre la frame précédente et la frame actuelle
        similarite = calculer_similarite_frames_couleur(previous_frame, current_frame)
        similarites.append(similarite)  # Ajouter la similarité à la liste

    # Convertir la liste des similarités en un tableau numpy pour faciliter les calculs statistiques
    similarites = np.array(similarites)
    
    return similarites

#region visualiser_detection_forme
def visualiser_detection_forme(video_path):
    """
    Visualise la détection de contours sur une vidéo en utilisant la fonction tracer_contours.

    Paramètres :
    ------------
    video_path : str
        Chemin de la vidéo à traiter.

    Retourne :
    ---------
    None
    """
    # Charger la vidéo et la convertir en array
    video = convertir_video_en_array(video_path)
    
    # Appliquer la détection de contours avec la fonction existante
    video_contours = tracer_contours(video)
    
    # Initialiser une fenêtre pour afficher les résultats
    cv2.namedWindow('Détection de contours', cv2.WINDOW_NORMAL)
    
    for i, frame in enumerate(video_contours):
        # Convertir les contours détectés (niveaux de gris) en une image couleur pour l'affichage
        display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Ajouter un texte indiquant le numéro de la frame
        cv2.putText(display_frame, f"Frame: {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Afficher la frame dans la fenêtre
        cv2.imshow('Détection de contours', display_frame)
        
        # Attendre 25 ms entre les frames (simule environ 40 FPS)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  # Quitter la boucle si 'q' est pressé
    
    # Libérer les ressources et fermer la fenêtre
    cv2.destroyAllWindows()








#region detection_transition
def detection_transition(similarites, silent): 
    
    # calcul moyenne et écart type glissant
    longueur_frame = 100
    moyenne_similarites=np.zeros(len(similarites-longueur_frame))
    ecart_type_similarites=np.zeros(len(similarites-longueur_frame))
    for i in range(len(similarites)):
        moyenne_similarites[i] = np.mean(similarites[i:i+longueur_frame])
        ecart_type_similarites[i] = np.std(similarites[i:i+longueur_frame])
        
    # Différence entre la moyenne et l'écart type
    seuil = 3.2
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
    seuil = 3.2
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

#region trace_similarites

def trace_similarites(similarites):
    """
    Trace les similarités entre les frames successives d'une vidéo et affiche des statistiques.

    Cette fonction génère un graphique montrant les similarités entre frames successives,
    ainsi que les courbes représentant la moyenne des similarités et la différence entre
    la moyenne et l'écart type.

    Paramètres :
    ------------
    similarites : numpy.ndarray
        Un tableau contenant les similarités calculées entre les frames successives.

    Retourne :
    ---------
    None
    """
    # Longueur de la fenêtre glissante
    longueur_frame = 100

    # Initialisation des tableaux pour la moyenne et l'écart type
    moyenne_similarites = np.zeros(len(similarites) - longueur_frame)
    ecart_type_similarites = np.zeros(len(similarites) - longueur_frame)

    # Calcul de la moyenne et de l'écart type glissant
    for i in range(len(moyenne_similarites)):
        moyenne_similarites[i] = np.mean(similarites[i:i+longueur_frame])
        ecart_type_similarites[i] = np.std(similarites[i:i+longueur_frame])

    # Différence entre la moyenne et l'écart type
    seuil = 3
    difference_moyenne_ecart_type = moyenne_similarites - seuil * ecart_type_similarites

    # Tracer les similarités entre frames
    plt.plot(similarites, label='Similarité entre frames', color='b')

    # Tracer la courbe de la moyenne des similarités
    plt.plot(range(len(moyenne_similarites)), moyenne_similarites, color='r', linestyle='--', label='Moyenne des similarités')

    # Tracer la courbe moyenne - écart type
    plt.plot(range(len(moyenne_similarites)), difference_moyenne_ecart_type, color='g', linestyle='--', label='Moyenne - Écart type')

    # Ajouter des détails au graphique
    plt.xlabel('Numéro de frame')  # Légende de l'axe des x
    plt.ylabel('Similarité')  # Légende de l'axe des y
    plt.title('Similarité entre frames successives dans la vidéo')  # Titre du graphique
    plt.legend()  # Afficher la légende des courbes
    plt.show()  # Afficher le graphique

    # Affichage des statistiques dans la console
    print(f"Moyenne des similarités : {moyenne_similarites.mean():.2f}")
    print(f"Écart type des similarités : {ecart_type_similarites.mean():.2f}")
    print(f"Différence (Moyenne - Écart type) : {difference_moyenne_ecart_type.mean():.2f}")




#region comparison
def comparison(similarites):
    """
    Compare les résultats de détection des changements de plans avec la vérité terrain.

    Cette fonction utilise une liste de frames détectées par un algorithme de détection 
    de transitions et les compare à une vérité terrain. Elle calcule les métriques 
    de précision, rappel et F1-score pour évaluer la performance de l'algorithme.

    Paramètres :
    ------------
    similarites : numpy.ndarray
        Tableau contenant les similarités entre les frames successives, utilisé pour 
        détecter les transitions.

    Return :
    ---------
    None
        Les résultats (précision, rappel, F1-score) sont affichés dans la console.

    Notes :
    -------
    - La fonction utilise une vérité terrain pour une vidéo spécifique, définie par 
      'frames_changement_plans_verite'.
    - Les métriques sont calculées en comparant les frames détectées par l'algorithme 
      avec celles de la vérité terrain.

    Exemple :
    ---------
    >>> similarites = np.random.rand(1000)  # Similarités simulées
    >>> comparison(similarites)
    Précision: 0.85
    Rappel: 0.90
    F1-score: 0.87
    """
    # Vérité terrain pour la vidéo spécifique
    nombre_changement_plans_verite = 77
    frames_changement_plans_verite = [0, 42, 52, 142, 163, 187, 200, 221, 248, 256, 268, 307, 485, 526, 561, 582, 
                                      595, 615, 635, 664, 690, 705, 720, 746, 821, 853, 903, 956, 975, 998, 
                                      1027, 1062, 1099, 1120, 1144, 1177, 1220, 1255, 1293, 1335, 1367, 1444, 
                                      1582, 1655, 1735, 1812, 1871, 1895, 1909, 1960, 2016, 2106, 2147, 2184,
                                      2243, 2487, 2526, 2617, 2688, 2775, 2808, 2829, 2858, 2881, 2917, 2934,
                                      2962, 2978, 3011, 3086, 3179, 3287]
    
    # Frames détectées par l'algorithme
    frames_changement_plans_par_notre_code = detection_transition_list(similarites)
    
    # Convertir en ensembles pour simplifier les comparaisons
    verite_set = set(frames_changement_plans_verite)
    detections_set = set(frames_changement_plans_par_notre_code)

    # Calcul des métriques de performance
    true_positives = len(verite_set.intersection(detections_set))  # Changements correctement détectés
    false_positives = len(detections_set - verite_set)  # Changements détectés mais incorrects
    false_negatives = len(verite_set - detections_set)  # Changements manqués

    # Calcul des scores : précision, rappel, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Afficher les résultats
    print(f"Précision: {precision:.2f}")
    print(f"Rappel: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

def lire_et_tracer_contours(video_path):
    """
    Lit une vidéo, applique la détection de contours avec la fonction `tracer_contours`, 
    et affiche les frames avec contours détectés en temps réel.

    Paramètres :
    ------------
    video_path : str
        Chemin de la vidéo à traiter.
    """
    # Ouvrir la vidéo avec OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erreur : Impossible de lire la vidéo.")
        return

    # Lire les frames une par une
    while True:
        ret, frame = cap.read()
        if not ret:  # Fin de la vidéo
            break
        
        # Appliquer la détection de contours directement sur la frame
        contours = tracer_contours([frame])[0]
        
        # Convertir les contours en image couleur pour affichage
        display_frame = cv2.cvtColor(contours, cv2.COLOR_GRAY2BGR)
        
        # Afficher la frame avec contours
        cv2.imshow('Contours détectés', display_frame)
        
        # Attendre 25ms (40 FPS) ou quitter si 'q' est pressé
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()




#region main
if __name__ == "__main__":
    video_path = 'pub/Pub_C+_352_288_1.mp4'  # Chemin de la vidéo
    #video = convertir_video_en_array(video_path)
    #visualiser_detection_forme(video_path)
    #lire_et_tracer_contours(video_path)
    
    video = convertir_video_en_array(video_path)
    print(f"La vidéo est de longueur {len(video)} et les frames sont de shape {video[0].shape}")
    video_standart = standardize_video_color(video)
    segmentation_spot_pub(video_standart)

    
    similarite_couleur = calculer_similarite_couleur(video)
    video_contour = tracer_contours(video)
    similarite_forme = calculer_similarite_forme(video_contour)
    #play_video(video_contour)
    silent = False
    detection_transition(similarite_couleur+similarite_forme, silent)
    simil_couleur_normal = (similarite_couleur-np.mean(similarite_couleur))/np.std(similarite_couleur)
    simil_forme_normal = (similarite_forme-np.mean(similarite_forme))/np.std(similarite_forme)
    
    #detection_transition_list(simil_couleur_normal+simil_forme_normal)
    #comparison(simil_couleur_normal+simil_forme_normal)
    trace_similarites(simil_couleur_normal+simil_forme_normal)
    
    