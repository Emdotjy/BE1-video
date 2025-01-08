import cv2
from utils import *
import numpy as np
import os
import json
import random
from segmentation_video import *
from utils import afficher_frame_avec_timecode
from utils import play_video
from utils import convertir_video_en_array



def standardize_frame(frame, min_values, max_values):
    """
    Standardise une image (frame) en normalisant ses valeurs entre 0 et 255.

    La normalisation suit la formule suivante :
    (frame - min_values) / (max_values - min_values) * 255

    Les valeurs normalisées sont ensuite limitées entre 0 et 255 
    pour garantir leur compatibilité avec le type uint8.

    Args:
        frame (numpy.ndarray): La trame d'image à normaliser.
        min_values (numpy.ndarray): Les valeurs minimales pour la normalisation, de même forme que 'frame'.
        max_values (numpy.ndarray): Les valeurs maximales pour la normalisation, de même forme que 'frame'.

    Returns:
        numpy.ndarray: La trame normalisée avec des valeurs entre 0 et 255 (type uint8).
    """
    # Calcul de la normalisation : ajuste les valeurs de 'frame' entre 0 et 255
    standardized = (frame - min_values) / (max_values - min_values) * 255
    
    # Clip pour s'assurer que les valeurs sont dans l'intervalle [0, 255] et conversion en uint8
    return np.clip(standardized, 0, 255).astype(np.uint8)


def standardize_video_color(video_frames):
    """
    Standardise les couleurs d'une liste de frames vidéo en normalisant leurs valeurs de pixel.

    Cette fonction calcule les valeurs minimales et maximales des moyennes de pixel
    parmi toutes les frames fournies, puis normalise chaque frame en utilisant ces valeurs.
    Si les valeurs minimale et maximale sont identiques, la normalisation est ignorée.

    Args:
        video_frames (list of numpy.ndarray): Liste des frames vidéo, chaque frame étant une matrice numpy.

    Returns:
        list of numpy.ndarray: Liste des frames normalisées, ou la liste originale si aucune normalisation n'a été appliquée.
    """
    # Vérifie si la liste de frames est vide
    if len(video_frames) == 0:
        print("Avertissement : aucune frame vidéo n'a été fournie.")
        return video_frames  # Retourne une liste vide

    # Empile toutes les frames pour faciliter les calculs
    stacked_frames = np.stack(video_frames, axis=0)

    # Calcul des valeurs minimales et maximales des moyennes des frames
    min_values = min([np.mean(frame) for frame in stacked_frames])
    max_values = max([np.mean(frame) for frame in stacked_frames])

    # Vérifie si les valeurs minimale et maximale sont identiques pour éviter une division par zéro
    if min_values == max_values:
        print("Avertissement : les valeurs min et max sont identiques, normalisation ignorée.")
        return video_frames  # Retourne les frames originales si la normalisation est impossible

    # Applique la normalisation à chaque frame
    standardized_frames = [standardize_frame(frame, min_values, max_values) for frame in video_frames]

    return standardized_frames



def is_black(frame):
    """
    Vérifie si une frame d'image peut être considérée comme "noire".

    Une frame est définie comme "noire" si la moyenne de ses valeurs de pixel est inférieure ou égale à 5.

    Args:
        frame (numpy.ndarray): Une frame d'image représentée comme une matrice numpy.

    Returns:
        bool: Retourne True si la frame est noire, False sinon.
    """
    # Calcul de la moyenne des valeurs de pixel et comparaison avec le seuil de noirceur
    return np.mean(frame) <= 5


def preprocess_image(image, target_size=(224, 224)):
    """
    Pré-traite une image en la redimensionnant et en la normalisant.

    Cette fonction redimensionne l'image à une taille standard spécifiée,
    puis normalise ses valeurs de pixel dans l'intervalle [0, 1].

    Args:
        image (numpy.ndarray): L'image d'entrée sous forme de matrice numpy.
        target_size (tuple): Un tuple (hauteur, largeur) spécifiant la taille de redimensionnement.

    Returns:
        numpy.ndarray: L'image pré-traitée, redimensionnée et normalisée.
    """
    # Redimensionne l'image à la taille spécifiée tout en préservant la plage de valeurs
    resized = cv2.resize(image, target_size, anti_aliasing=True, preserve_range=True)
    
    # Convertit les valeurs de l'image en type float et normalise dans l'intervalle [0, 1]
    normalized = resized.astype(float) / 255.0
    
    # Retourne l'image pré-traitée
    return normalized


def segmentation_spot_pub(video):
    """
    Segmente un ensemble de frames vidéo pour extraire les séquences publicitaires.

    Cette fonction identifie les séquences publicitaires dans une vidéo en repérant
    les transitions entre des frames "non-noires" et des séries de frames "noires"
    (indiquant potentiellement des transitions publicitaires).

    Args:
        video (list of numpy.ndarray): Liste des frames vidéo, chaque frame étant une matrice numpy.

    Returns:
        list of list of numpy.ndarray: Liste des séquences publicitaires, 
        chaque séquence étant une sous-liste de frames correspondant à une publicité.
    """
    # Détecte les frames noires dans la vidéo
    black_frames = [is_black(frame) for frame in video]
    
    # Initialise les indices des débuts et des fins de séquences
    debuts = [0]
    fins = []

    # Parcours des frames pour détecter les transitions publicitaires
    for i in range(len(black_frames) - 1):
        # Transition : une frame non-noire suivie d'une frame noire
        if not black_frames[i] and black_frames[i + 1]:
            j = 1
            # Compte les frames noires consécutives
            while i + j < len(black_frames) and black_frames[i + j]:
                j += 1
            # Vérifie si la durée de la séquence noire est acceptable pour une publicité
            if 7 <= j <= 16:
                debuts.append(i + j + 1)  # Ajoute le début de la séquence publicitaire
                fins.append(i + 1)       # Ajoute la fin de la séquence publicitaire

    # Enlève la dernière valeur des débuts qui correspond à la fin de la séquence publicitaire
    debuts.pop()

    # Supprime les premiers indices pour éviter des décalages
    debuts.pop(0)
    fins.pop(0)

    # Segmente les frames publicitaires à partir des indices détectés
    pub_list = [video[debuts[i]:fins[i]] for i in range(len(debuts))]

    # Affiche les informations sur chaque séquence publicitaire
    for num_pub, pub in enumerate(pub_list):
        print(f"Séquence pub n°{num_pub + 1}, démarre à {debuts[num_pub]} et se termine à {fins[num_pub]}")

    return pub_list


def detection_transitions_from_pub(video):
    """
    Détecte les transitions dans une vidéo, en particulier celles liées aux séquences publicitaires.

    Cette fonction calcule deux types de similarités entre les frames vidéo : 
    la similarité de couleur et la similarité de forme. Elle combine ces deux métriques 
    pour détecter les transitions significatives.

    Args:
        video (list of numpy.ndarray): Liste des frames vidéo, chaque frame étant une matrice numpy.

    Returns:
        list: Liste des indices ou positions des transitions détectées dans la vidéo.
    """
    # Calcule la similarité basée sur les couleurs entre les frames de la vidéo
    similarite_couleur = calculer_similarite_couleur(video)
    # Génère une version de la vidéo avec des contours pour détecter les formes
    video_contour = tracer_contours(video)
    # Calcule la similarité basée sur les formes entre les frames de la vidéo
    similarite_forme = calculer_similarite_forme(video_contour)
    # Variable pour activer ou désactiver le mode silencieux dans la détection
    silent = True
    # Combine les similarités de couleur et de forme pour détecter les transitions
    return detection_transition(similarite_couleur + similarite_forme, silent)

    #simil_couleur_normal = (similarite_couleur-np.mean(similarite_couleur))/np.std(similarite_couleur)
    #simil_forme_normal = (similarite_forme-np.mean(similarite_forme))/np.std(similarite_forme)
    #silent = True
    #return detection_transition(simil_couleur_normal+simil_forme_normal, silent)
    
def get_rpz_images(pub, silent):
    """
    Génère une liste d'images représentatives pour chaque séquence identifiée dans une publicité.

    Cette fonction détecte les transitions au sein de la publicité et calcule une image moyenne
    pour chaque segment entre deux transitions. Les images représentatives peuvent être affichées
    à l'écran si le mode silencieux est désactivé.

    Args:
        pub (list of numpy.ndarray): Liste des frames vidéo d'une publicité.
        silent (bool): Si True, désactive l'affichage des images. Si False, les images sont affichées.

    Returns:
        list of numpy.ndarray: Liste d'images représentatives pour chaque segment détecté.
    """
    images_rpz = []  # Liste pour stocker les images représentatives
    pub_transitions = [0]  # Initialisation avec le début de la publicité

    # Ajout de la dernière frame répétée pour éviter des détections incorrectes en fin de vidéo
    pub_transitions += detection_transitions_from_pub(pub + pub[-1] * 100)

    # Supprime les transitions détectées au-delà de la longueur réelle de la vidéo
    for i in range(len(pub_transitions) - 1, 0, -1):
        if pub_transitions[i] >= len(pub):
            pub_transitions.pop(-1)

    # Ajoute la dernière frame comme limite finale
    pub_transitions.append(len(pub) - 1)

    # Parcours des segments pour calculer les images représentatives
    for i in range(len(pub_transitions) - 1):
        # Calcul de l'image moyenne pour le segment courant
        images_rpz.append(np.mean(pub[pub_transitions[i]:pub_transitions[i + 1]], axis=0).astype(np.uint8))

        # Affiche l'image si le mode silencieux est désactivé
        if not silent:
            cv2.imshow(f'Séquence n°{i}', images_rpz[-1])
            cv2.waitKey(0)  # Attente d'une action de l'utilisateur
            cv2.destroyAllWindows()

    return images_rpz


def add_pub_to_bdd(pub):
    """
    Ajoute une publicité (pub) à une base de données locale en sauvegardant ses images représentatives
    et ses métadonnées.

    La fonction extrait les images représentatives de la publicité, les enregistre dans un dossier spécifique 
    dans une base de données locale, et met à jour les métadonnées associées à cette publicité.

    Args:
        pub (list of numpy.ndarray): Liste des frames vidéo de la publicité.

    Returns:
        None
    """
    silent = True  # Mode silencieux pour l'extraction des images représentatives
    rpz_images = get_rpz_images(pub, silent)  # Obtenir les images représentatives

    # --------- Enregistrement dans la base de données --------------------------------------------------------

    # Vérifie et crée le dossier de base de données s'il n'existe pas
    bdd_path = 'bdd'
    if not os.path.exists(bdd_path):       
        os.makedirs(bdd_path)

    # Liste les dossiers existants dans la base de données
    folders = [f for f in os.listdir(bdd_path) if os.path.isdir(os.path.join(bdd_path, f))]
    num_pub = len(folders)  # Numéro de la nouvelle publicité

    # Chemin du dossier spécifique à la nouvelle publicité
    pub_path = bdd_path + f'/images_representatives_pub_n{str(num_pub + 1)}'
    if not os.path.exists(pub_path):       
        os.makedirs(pub_path)  # Crée le dossier pour cette publicité

    # Enregistre chaque image représentative dans le dossier
    for num_image, image in enumerate(rpz_images):
        cv2.imwrite(pub_path + f'/sequence_n{num_image + 1}.png', image)

    # Gestion des métadonnées
    metadata_path = os.path.join(pub_path, "metadata.json")
    metadata = {}

    # Si un fichier de métadonnées existe déjà, on le charge pour le modifier
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
    
    # Mise à jour des métadonnées avec la longueur de la publicité
    metadata['pub_length'] = len(pub)

    # Enregistrement des métadonnées mises à jour dans un fichier JSON
    with open(metadata_path, 'w') as file:
        json.dump(metadata, file, indent=4)
    
    print(f"Métadonnées enregistrées : {metadata_path}")


def argmin(d):
    """
    Trouve la clé associée à la valeur minimale dans un dictionnaire.

    Cette fonction retourne la première clé correspondant à la valeur minimale
    du dictionnaire. Si le dictionnaire est vide, elle retourne 'None'.

    Args:
        d (dict): Un dictionnaire où les clés sont associées à des valeurs numériques.

    Returns:
        hashable: La clé associée à la valeur minimale, ou 'None' si le dictionnaire est vide.
    """
    # Vérifie si le dictionnaire est vide
    if not d:
        return None

    # Trouve la valeur minimale parmi les valeurs du dictionnaire
    min_val = min(d.values())

    # Retourne la première clé dont la valeur correspond à la valeur minimale
    return [k for k in d if d[k] == min_val][0]


def compare_images(img1, img2):
    """
    Compare deux images en calculant une mesure de distance normalisée entre elles.

    La fonction redimensionne la plus grande image pour qu'elle corresponde 
    à la taille de la plus petite, puis calcule la norme euclidienne (distance de Frobenius)
    entre les deux images. La distance est ensuite normalisée par la racine carrée du nombre de pixels.

    Args:
        img1 (numpy.ndarray): Première image à comparer.
        img2 (numpy.ndarray): Deuxième image à comparer.

    Returns:
        float: Une mesure de distance normalisée entre les deux images.
    """
    # Obtenir les dimensions des deux images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
        
    # Redimensionne la plus grande image à la taille de la plus petite
    if h1 * w1 > h2 * w2:
        img1_resized = cv2.resize(img1, (w2, h2))  # Redimensionne img1 à la taille de img2
        # Calcule la distance normalisée
        return np.linalg.norm(img1_resized - img2) / np.sqrt(h2 * w2)
    else:
        img2_resized = cv2.resize(img2, (w1, h1))  # Redimensionne img2 à la taille de img1
        # Calcule la distance normalisée
        return np.linalg.norm(img1 - img2_resized) / np.sqrt(h1 * w1)


def compute_elastic_distance(seq1, seq2, max_warp=3):
    """
    Calcule la distance élastique entre deux séquences d'images avec une fenêtre de déformation limitée.

    La distance élastique (inspirée de la Dynamic Time Warping - DTW) est utilisée pour comparer
    deux séquences en alignant de manière optimale leurs éléments tout en permettant des déformations
    dans les alignements. Une fenêtre de déformation limite les écarts possibles entre les éléments alignés.

    Args:
        seq1 (list of numpy.ndarray): Première séquence d'images.
        seq2 (list of numpy.ndarray): Deuxième séquence d'images.
        max_warp (int): Taille maximale de la fenêtre de déformation (par défaut 3).

    Returns:
        float: Distance élastique normalisée entre les deux séquences.
    """
    n, m = len(seq1), len(seq2)

    # Initialisation de la matrice des coûts avec des valeurs infinies
    cost_matrix = np.full((n + 1, m + 1), np.inf)
    cost_matrix[0, 0] = 0  # Le coût initial est de 0

    # Remplissage de la matrice des coûts
    for i in range(1, n + 1):
        # Limite la fenêtre de déformation autour de la diagonale
        start_j = max(1, i - max_warp)
        end_j = min(m + 1, i + max_warp + 1)

        for j in range(start_j, end_j):
            # Calcule le coût entre les images correspondantes des deux séquences
            cost = compare_images(seq1[i - 1], seq2[j - 1])
            # Mise à jour de la matrice des coûts en tenant compte des trois chemins possibles :
            # insertion, suppression, ou correspondance
            cost_matrix[i, j] = cost + min(
                cost_matrix[i - 1, j],    # insertion
                cost_matrix[i, j - 1],    # suppression
                cost_matrix[i - 1, j - 1] # correspondance
            )

    # Retourne la distance normalisée par la longueur totale des séquences
    return cost_matrix[n, m] / (n + m)


def compute_best_fit_distance(seq1, seq2):
    """
    Calcule la distance d'ajustement optimal ("best-fit") entre deux séquences d'images.

    Cette fonction compare chaque image de la première séquence ('seq1') à toutes les images
    de la deuxième séquence ('seq2') et retient la distance minimale pour chaque image.
    La distance d'ajustement optimal est la moyenne de ces distances minimales.

    Args:
        seq1 (list of numpy.ndarray): Première séquence d'images.
        seq2 (list of numpy.ndarray): Deuxième séquence d'images.

    Returns:
        float: Moyenne des distances minimales pour chaque image de 'seq1' par rapport à 'seq2'.
    """
    # Pour chaque image dans seq1 calculer la distance entre l'image actuelle de seq1 et chaque image de seq2
    return np.mean([min([compare_images(img1, img2) for img2 in seq2])for img1 in seq1])

    
def recognise_pub_in_bdd(video):
    """
    Identifie une publicité dans la base de données en comparant les marqueurs visuels
    d'une vidéo donnée avec ceux des publicités enregistrées.

    Cette fonction vérifie d'abord la compatibilité de la longueur de la vidéo avec les publicités
    dans la base de données, puis compare les images représentatives (marqueurs) pour trouver
    la publicité la plus proche dans la base de données.

    Args:
        video (list of numpy.ndarray): Liste des frames vidéo représentant la publicité à identifier.

    Returns:
        None: Affiche le chemin de la publicité la plus proche dans la base de données et sa distance.
    """
    # Sélection de 5 frames aléatoires de la vidéo pour échantillonnage futur 
    frame_list = random.choices(video,k= 5)
    #  Liste des chemins des dossiers dans la base de données
    bdd_paths = os.listdir('./bdd')
    # Stocke les longueurs des publicités enregistrées
    length = dict()
    # Liste des publicités compatibles en termes de longueur
    length_compatible_pub = []
    # Vérification de la compatibilité de la longueur des publicités
    for path in bdd_paths:
        # Charge les métadonnées pour récupérer la longueur de la publicité
        with open('./bdd/'+path+'/metadata.json', 'r') as file:
            length[path] = json.load(file)['pub_length']    
        #Si les deux pubs on à-peu près la même longueur (à +/- 120 frames près)
        if abs(length[path] -len(video)) <= 120 :  
            length_compatible_pub.append('./bdd/'+path )

    # Extraction des images représentatives pour la vidéo donnée
    rpz_images_test = get_rpz_images(video, True)
    # Dictionnaire pour stocker les distances entre la vidéo et les publicités compatibles
    distances_to_pubs_in_bdd = dict()
    # Comparaison avec chaque publicité compatible
    for pub_path in length_compatible_pub:
        # Liste des chemins des images représentatives dans la publicité courante
        rpz_images_bdd_paths = [path for path in os.listdir(pub_path)]
        # Supprime les métadonnées des chemins (en général le fichier metadata.json)
        rpz_images_bdd_paths.pop(0)
        # Charge les images représentatives de la publicité courante
        rpz_images_bdd = [np.array(cv2.imread(pub_path+'/'+path)) for path in rpz_images_bdd_paths]

        # Comparaison des marqueurs visuels de la vidéo à ceux de la publicité courante
        distance_to_pub = compute_best_fit_distance(rpz_images_test, rpz_images_bdd)
        #distance_to_pub = compute_elastic_distance(rpz_images_test, rpz_images_bdd,5) 
        # # Stocke la distance calculée     
        distances_to_pubs_in_bdd[pub_path] = (distance_to_pub)
    
    # Trouve la publicité avec la distance minimale
    best_fit_path = argmin(distances_to_pubs_in_bdd)

    # Affiche les distances et le meilleur résultat
    print(distances_to_pubs_in_bdd)
    print(f"the best fit within the database was with pub was found with {best_fit_path}, with a distance of {distances_to_pubs_in_bdd[best_fit_path]}")

          


if __name__ == "__main__":


    video_path = 'Pub_C+_352_288_1.mp4'  # Chemin de la vidéo
    video = convertir_video_en_array(video_path)
    video_standart = standardize_video_color(video)
    pub_list = segmentation_spot_pub(video_standart)


    
    # à décommenter une fois pour crééer la base de données, puis à recommenter
    for pub in pub_list:
        add_pub_to_bdd(pub)

    video_test_path = 'Pub_C+_176_144.mp4'    
    video_test = standardize_video_color(convertir_video_en_array(video_test_path))
    pub_list_test = segmentation_spot_pub(video_test)

    for pub_test in pub_list_test:
        recognise_pub_in_bdd(pub_test)













