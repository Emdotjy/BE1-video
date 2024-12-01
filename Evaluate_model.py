from segmentation_video import *

import csv

def get_first_column_values(file_path):
    first_column_values = []
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:  
                first_column_values.append(int(row[0]))
    return first_column_values




if __name__ == "__main__":
    video_path = 'Pub_C+_352_288_1.mp4'  # Chemin de la vidéo
    video = convertir_video_en_array(video_path)
    labels_path = "Vérité_terrain_Pub_C+_352_288_1_.csv" 

    print(f"la vidéo est de longueur {len(video)} et les frames sont de shape {video[0].shape}")
    similarite_couleur = calculer_similarite_couleur(video)
    video_contour = calculate_contour_frames(video)
    similarite_forme = calculate_similarity(video_contour)

    #play_video(video_contour)
    #detection_transition(similarite_couleur+similarite_forme)
    simil_couleur_normal = (similarite_couleur-np.mean(similarite_couleur))/np.std(similarite_couleur)
    simil_forme_normal = (similarite_forme-np.mean(similarite_forme))/np.std(similarite_forme)
    silent = True
    transition_frames_numbers = detection_transition(simil_couleur_normal+simil_forme_normal,silent)
    labels = get_first_column_values(labels_path)
    print(transition_frames_numbers)
    print(labels)
    TP = 0
    TN = 0
    FP = 0
    FN = 0 
    for detection in transition_frames_numbers:
        if detection in labels:
            TP+=1
        else:
            FP += 1
    for label in labels:
        if not label in transition_frames_numbers:
            FN+=1
    print('TPfdsqfdsqf')        



