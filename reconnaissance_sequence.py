import cv2
from obtain_array import *
import numpy as np

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

    return debuts,fins

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


if __name__ == "__main__":
    video_path = 'Pub_C+_352_288_1.mp4'  # Chemin de la vidéo
    video = convertir_video_en_array(video_path)
    debuts, fins = segmentation_spot_pub(video)

    pub_list = [video[debuts[i]:fins[i]] for i in range(len(debuts))]    

    pub_rpz_images_dict = {}

    silent = True
    for num_pub, pub in enumerate(pub_list):
        print(f"Sequence pub n°{num_pub}, démarre à {debuts[num_pub]} et se termine à {fins[num_pub]}")
        rpz_images = get_rpz_images(pub, silent)
        pub_rpz_images_dict[f"pub n°{num_pub} from the file {video_path}"] = rpz_images







