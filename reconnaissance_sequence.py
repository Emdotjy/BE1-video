import cv2
from obtain_array import *
import numpy as np
import os
import json

from segmentation_video import *
def is_black(frame):
    return (np.mean(frame) <= 10)



def segmentation_spot_pub(video):
    black_frames = [is_black(frame) for frame in video]
    debuts = [0]
    fins = []

    for i in range(len(black_frames)-1):
        if not black_frames[i] and black_frames[i+1]:
            j = 1
            while i+j <len(black_frames) and  black_frames[i+j]:
                j+=1
            if j <=60 and j>=5:
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



def recognise_pub_in_bdd(video_path):
    pass


if __name__ == "__main__":
    video_path = 'Pub_C+_352_288_1.mp4'  # Chemin de la vidéo
    video = convertir_video_en_array(video_path)

    pub_list = segmentation_spot_pub(video)
    

    
    
    for pub in pub_list:
        add_pub_to_bdd(pub)
    video_test_path = 'Pub_C+_352_288_2.mp4'    
    video = convertir_video_en_array(video_path)
    pub_list = segmentation_spot_pub(video)










