import cv2
from obtain_array import *
import numpy as np
import os
import json
import random
from segmentation_video import *
import matplotlib.pyplot as plt 

def standardize_frame(frame,min_values,max_values):
    # Normalisation : (frame - min) / (max - min) * 255
    standardized = (frame - min_values) / (max_values - min_values) * 255
    return np.clip(standardized, 0, 255).astype(np.uint8)  
def standardize_video_color(video_frames):

    stacked_frames = np.stack(video_frames, axis=0)
    
    # Trouver les valeurs min et max pour chaque canal RGB
    min_values = min([np.mean(frame ) for frame in stacked_frames])
    max_values = max([np.mean(frame ) for frame in stacked_frames])

    standardized_frames = [standardize_frame(frame,min_values,max_values) for frame in video_frames]

    return standardized_frames

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
            print(j)    
            if j <= 16 and j>=7:
                debuts.append(i+j+1)
                fins.append(i+1)
    print(debuts)            
    # On enlève la dernière valeur de début qui correspond à la fin de séquence pub 
    debuts.pop()          

    #on de même, on enlève le début
    debuts.pop(0)
    fins.pop(0)
    pub_list = [video[debuts[i]:fins[i]] for i in range(len(debuts))]    
    for num_pub, pub in enumerate(pub_list):
        print(f"Sequence pub n°{num_pub+1}, démarre à {debuts[num_pub]} et se termine à {fins[num_pub]}")

    return pub_list

def detection_transitions_from_pub(video):
    similarite_couleur = calculer_similarite_couleur(video)
    video_contour = calculate_contour_frames(video)
    similarite_forme = calculate_similarity(video_contour)

    #play_video(video_contour)
    
    silent = True
    return detection_transition(similarite_couleur+similarite_forme,silent)
    #simil_couleur_normal = (similarite_couleur-np.mean(similarite_couleur))/np.std(similarite_couleur)
    #simil_forme_normal = (similarite_forme-np.mean(similarite_forme))/np.std(similarite_forme)
    #silent = True
    #return detection_transition(simil_couleur_normal+simil_forme_normal, silent)
    
def get_rpz_images(pub,silent):
    images_rpz = []
    pub_transitions = [0]
    #rajout de la dernière frame pour éviter les détections due à la fin de la vidéo 
    pub_transitions += detection_transitions_from_pub(pub + pub[-1]*100)
    for i in range(len(pub_transitions)-1,0,-1):
        if pub_transitions[i] >= len(pub):
            pub_transitions.pop(-1)
    pub_transitions.append(len(pub)-1)
    print(pub_transitions)
    for i in range(len(pub_transitions)-1):

        images_rpz.append( np.mean(pub[pub_transitions[i]:pub_transitions[i+1]],axis=0).astype(np.uint8))
        if not silent:
            cv2.imshow(f'sequence n°{i}',images_rpz[-1])
            cv2.waitKey(0)  # Attendre que l'utilisateur appuie sur une touche
            cv2.destroyAllWindows()
    return images_rpz


def add_pub_to_bdd(pub):

    silent = True
    rpz_images = get_rpz_images(pub, silent)
    #---------Enregistrement dans la base de donnée --------------------------------------------------------

    # Vérifier et créer le dossier si nécessaire
    bdd_path = 'bdd'
    if not os.path.exists(bdd_path):       
        os.makedirs(bdd_path)

    folders = [f for f in os.listdir(bdd_path) if os.path.isdir(os.path.join(bdd_path, f))]
    num_pub = len(folders)

    num_pub, pub
        # Vérifier et créer le dossier spécifique pour la pub, identifié avec son numéro
    pub_path = bdd_path+f'/images_representatives_pub_n{str(num_pub+1)}'
    if not os.path.exists(pub_path):       
        os.makedirs(pub_path)            
        #enregistrer les images et le deltat
    for num_image,image in enumerate(rpz_images):

        cv2.imwrite(pub_path+f'/sequence n{num_image+1}.png',image)

    metadata_path = os.path.join(pub_path, "metadata.json")
    metadata = {}
    #si metadata existe déja, on le modifie
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
            
    metadata['pub_length'] = len(pub)
            
    with open(metadata_path, 'w') as file:
        json.dump(metadata, file, indent=4)
    print(f"Metadata saved: {metadata_path}")


def argmin(d):
    if not d: return None
    min_val = min(d.values())
    return [k for k in d if d[k] == min_val][0]


def recognise_pub_in_bdd(video):
    frame_list = random.choices(video,k= 5)
    bdd_paths = os.listdir('./bdd')
    length = dict()
    length_compatible_pub = []
    for path in bdd_paths:
        with open('./bdd/'+path+'/metadata.json', 'r') as file:
            length[path] = json.load(file)['pub_length']    
        #Si les deux pubs on à-peu près la même longueur, à 60 frames près
        if abs(length[path] -len(video)) <= 6000000 :  
            length_compatible_pub.append('./bdd/'+path )


    rpz_images_test = get_rpz_images(video, True)
    distances_to_pubs_in_bdd = dict()
    for pub_path in length_compatible_pub:
        #on fait un pop pour enleveer les metadata
        rpz_images_bdd_paths = [path for path in os.listdir(pub_path)]
        rpz_images_bdd_paths.pop(0)
        rpz_images_bdd = [np.array(cv2.imread(pub_path+'/'+path)) for path in rpz_images_bdd_paths]

        #On a chargé les marqueurs de la pub dans la bdd stockées dans pub_path, maintenant on les compares au marqueur de la pub à identifier
        distance_to_pub = np.mean([
                min(
                    [np.linalg.norm(rpz_image_test - rpz_image_bdd) / np.sqrt(rpz_image_test.size)
                    for rpz_image_bdd in rpz_images_bdd ]) 
            for rpz_image_test in rpz_images_test ])
        distances_to_pubs_in_bdd[pub_path] = (distance_to_pub)
    best_fit_path = argmin(distances_to_pubs_in_bdd)
    print(distances_to_pubs_in_bdd)
    print(f"the best fit within the database was with pub was found with {best_fit_path}, with a distance of {distances_to_pubs_in_bdd[best_fit_path]}")

          


if __name__ == "__main__":


    video_path = 'Pub_C+_352_288_1.mp4'  # Chemin de la vidéo
    video = convertir_video_en_array(video_path)
    video_standart = standardize_video_color(video)
    pub_list = segmentation_spot_pub(video_standart)


    
    # à décommenter une fois pour crééer la base de données, puis à recommenter
    #for pub in pub_list:
    #    add_pub_to_bdd(pub)

    video_test_path = 'Pub_C+_352_288_2_.mp4'    
    video_test = standardize_video_color(convertir_video_en_array(video_test_path))
    pub_list_test = segmentation_spot_pub(video_test)

    for pub_test in pub_list_test:
        recognise_pub_in_bdd(pub_test)













