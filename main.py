import cv2
import numpy as np

import os
import glob
import imageio # Pour les GIFs
from scipy.optimize import curve_fit

import argparse

from mask1_NN import predict_mask_with_resizing
from mask_script import get_surface_lines_from_mask
from plot3d_script import transform_surface_points_to_3d, plot_3d_surface_matplotlib

# --- Configuration ---
FUNDUS_IMAGE_WIDTH = 495
BOTTOM_CROP_PIXELS_OCT = 100 
BOTTOM_CROP_PIXELS_MASK = 0 
PLOT_3D_AS_MESH = False  # Mettre à False pour un nuage de points rapide, True pour un maillage
INITIAL_ANGLE_DEGREES = 0.0
ANGLE_INCREMENT_DEGREES = 3.75

NN_INPUT_HEIGHT = 128  
NN_INPUT_WIDTH = 256   

MODEL_ILM_PATH = 'oct_segmentation_mask_1.keras' 
MODEL_HRC_PATH = 'oct_segmentation_mask_2.keras' 
import tensorflow as tf

try:
    def dice_coef_keras(y_true, y_pred, smooth=1e-6): 
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    loaded_ILM_model = tf.keras.models.load_model(MODEL_ILM_PATH, custom_objects={'dice_coef_keras': dice_coef_keras} if 'dice_coef_keras' in locals() else None)
    print("Modèle ILM chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle ILM: {e}")
    exit()

try:
    def dice_coef_keras(y_true, y_pred, smooth=1e-6):
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    loaded_HRC_model = tf.keras.models.load_model(MODEL_HRC_PATH, custom_objects={'dice_coef_keras': dice_coef_keras} if 'dice_coef_keras' in locals() else None)
    print("Modèle chargé HRC avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle HRC: {e}")
    exit()

# Pour l'extraction de la surface du masque
PIXEL_OF_INTEREST_IN_MASK = 255 # Valeur des pixels d'intérêt dans le masque

GIF_FPS = 5


def preprocess_oct_image(full_bgr_image):
    if full_bgr_image is None: return None, None
    # Extraire la partie OCT B-scan
    oct_b_scan_full = full_bgr_image[:, FUNDUS_IMAGE_WIDTH:]
    
    # Rogner le bas
    processed_oct_bgr = oct_b_scan_full[:-BOTTOM_CROP_PIXELS_OCT, :] if oct_b_scan_full.shape[0] > BOTTOM_CROP_PIXELS_OCT else oct_b_scan_full.copy()
    
    if processed_oct_bgr.size == 0: return None, None
    gray_oct_processed = cv2.cvtColor(processed_oct_bgr, cv2.COLOR_BGR2GRAY)
    return gray_oct_processed, processed_oct_bgr



def gaussian_2d_model(xy_tuple, amplitude, xo, yo, sigma_x, sigma_y, offset):
    (x, y) = xy_tuple
    xo = float(xo); yo = float(yo)
    # Z est recalé, donc offset ~ 0, amplitude = profondeur
    g = offset + amplitude * np.exp(-(((x - xo)**2 / (2 * sigma_x**2)) + ((y - yo)**2 / (2 * sigma_y**2))))
    return g

# Variable globale pour stocker le chemin du masque en cours de traitement (pour les messages d'erreur)
mask_path_being_processed = ""

# --- Fonction principale de traitement ---
def process_oct_series(series_dir_path, output_dir="output_project_final"):
    global mask_path_being_processed # Pour utiliser dans preprocess_mask

    if not os.path.isdir(series_dir_path):
        print(f"Erreur: Le répertoire d'images '{series_dir_path}' n'existe pas.")
        return


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = sorted(glob.glob(os.path.join(series_dir_path, '*.png')))
    # Adapte pour .tif, .jpg si nécessaire
    if not image_files: image_files = sorted(glob.glob(os.path.join(series_dir_path, '*.tif')))
    if not image_files: image_files = sorted(glob.glob(os.path.join(series_dir_path, '*.jpg')))


    if not image_files:
        print(f"Aucune image trouvée dans {series_dir_path}")
        return

    num_scans = len(image_files)
    print(f"Traitement de {num_scans} images de '{os.path.basename(series_dir_path)}'")

    all_angles_rad = []
    all_oct_widths = []
    all_processed_bgr_for_gif = []

    all_upper_coords_m1, all_lower_coords_m1 = [], []
    all_min_rows_upper_m1 = [] # Pour recalage Z de la surface supérieure du masque 1

    all_upper_coords_m2, all_lower_coords_m2 = [], []
    # all_min_rows_upper_m2 = [] # Si tu veux reconstruire la surface sup du masque 2 aussi

    for i, img_path in enumerate(image_files):
        current_angle_deg = INITIAL_ANGLE_DEGREES + (i * ANGLE_INCREMENT_DEGREES)
        all_angles_rad.append(np.deg2rad(current_angle_deg))
        base_filename = os.path.basename(img_path)
        print(f"  Processing {base_filename} (Angle: {current_angle_deg:.2f}°, Image {i+1}/{num_scans})")

        full_bgr_oct_original = cv2.imread(img_path)
        if full_bgr_oct_original is None:
            print(f"    Impossible de charger l'image OCT.")
            # Remplir toutes les listes avec None pour cette itération
            all_oct_widths.append(None); all_upper_coords_m1.append(None); all_lower_coords_m1.append(None)
            all_min_rows_upper_m1.append(None); all_upper_coords_m2.append(None); all_lower_coords_m2.append(None)
            all_processed_bgr_for_gif.append(None)
            continue

        gray_oct_processed, bgr_oct_processed = preprocess_oct_image(full_bgr_oct_original)
        if gray_oct_processed is None:
            print(f"    Erreur de prétraitement de l'image OCT.")
            all_oct_widths.append(None); all_upper_coords_m1.append(None); all_lower_coords_m1.append(None)
            all_min_rows_upper_m1.append(None); all_upper_coords_m2.append(None); all_lower_coords_m2.append(None)
            all_processed_bgr_for_gif.append(None)
            continue
        
        all_oct_widths.append(gray_oct_processed.shape[1])
        target_shape_for_masks = gray_oct_processed.shape # (height, width)

        # Variables pour cette itération
        upper_m1, lower_m1, min_row_um1 = None, None, None
        upper_m2, lower_m2 = None, None # min_row_um2 si besoin

        # ------- TRAITEMENT MASQUE 1 (ex: ILM) --------
        _,mask1,_ = predict_mask_with_resizing(image_path=img_path,model= loaded_ILM_model,nn_h= NN_INPUT_HEIGHT,nn_w= NN_INPUT_WIDTH)
  

        

        upper_m1, lower_m1 = get_surface_lines_from_mask(mask1, PIXEL_OF_INTEREST_IN_MASK)
        if upper_m1 is not None:
            valid_upper_m1_pts = upper_m1[upper_m1 != -1]
            if valid_upper_m1_pts.size > 0:
                min_row_um1 = np.min(valid_upper_m1_pts)

        
        all_upper_coords_m1.append(upper_m1)
        all_lower_coords_m1.append(lower_m1)
        all_min_rows_upper_m1.append(min_row_um1)

        # ------- TRAITEMENT MASQUE 2 (ex: HRC ou autre limite) --------
        _,mask2,_ = predict_mask_with_resizing(image_path=img_path,model= loaded_HRC_model,nn_h= NN_INPUT_HEIGHT,nn_w= NN_INPUT_WIDTH)
  

        

        upper_m2, lower_m2 = get_surface_lines_from_mask(mask2, PIXEL_OF_INTEREST_IN_MASK)


        
        all_upper_coords_m2.append(upper_m2)
        all_lower_coords_m2.append(lower_m2)


# ** Préparation pour le GIF **
        frame_for_gif = bgr_oct_processed.copy() # Image sur laquelle on va dessiner

        # Dessiner surface supérieure du masque 1 (ex: ILM) en vert
        if upper_m1 is not None: # upper_m1 vient de get_surface_lines_from_mask
            print(f"    GIF: Masque 1 - Surface Supérieure: {len(np.where(upper_m1 != -1)[0])} points valides.") # DEBUG
            for col_idx, row_val in enumerate(upper_m1):
                if row_val != -1: # Dessiner seulement si un point a été trouvé pour cette colonne
                    # Vérifier si (col_idx, row_val) est dans les limites de frame_for_gif
                    if 0 <= col_idx < frame_for_gif.shape[1] and 0 <= row_val < frame_for_gif.shape[0]:
                        cv2.circle(frame_for_gif, (col_idx, row_val), 1, (0, 255, 0), -1) # Vert
                    # else: # DEBUG
                    #     print(f"      GIF M1 Sup: Point hors limites ({col_idx}, {row_val}) pour image {frame_for_gif.shape}")

        # Dessiner surface inférieure du masque 1 en vert clair
        if lower_m1 is not None:
            print(f"    GIF: Masque 1 - Surface Inférieure: {len(np.where(lower_m1 != -1)[0])} points valides.") # DEBUG
            for col_idx, row_val in enumerate(lower_m1):
                #if row_val != -1:
                    #if 0 <= col_idx < frame_for_gif.shape[1] and 0 <= row_val < frame_for_gif.shape[0]:
                        cv2.circle(frame_for_gif, (col_idx, row_val), 1, (100, 200, 100), -1) # Vert clair
                    # else: # DEBUG
                    #     print(f"      GIF M1 Inf: Point hors limites ({col_idx}, {row_val}) pour image {frame_for_gif.shape}")

        # Dessiner surface supérieure du masque 2 en rouge
        if upper_m2 is not None:
            print(f"    GIF: Masque 2 - Surface Supérieure: {len(np.where(upper_m2 != -1)[0])} points valides.") # DEBUG
            for col_idx, row_val in enumerate(upper_m2):
                if row_val != -1:
                    if 0 <= col_idx < frame_for_gif.shape[1] and 0 <= row_val < frame_for_gif.shape[0]:
                        cv2.circle(frame_for_gif, (col_idx, row_val), 1, (255, 0, 0), -1) # Rouge
                    # else: # DEBUG
                    #     print(f"      GIF M2 Sup: Point hors limites ({col_idx}, {row_val}) pour image {frame_for_gif.shape}")
        
        # Dessiner surface inférieure du masque 2 en rose
        if lower_m2 is not None:
            print(f"    GIF: Masque 2 - Surface Inférieure: {len(np.where(lower_m2 != -1)[0])} points valides.") # DEBUG
            for col_idx, row_val in enumerate(lower_m2):
                if row_val != -1:
                    if 0 <= col_idx < frame_for_gif.shape[1] and 0 <= row_val < frame_for_gif.shape[0]:
                        cv2.circle(frame_for_gif, (col_idx, row_val), 1, (200, 100, 100), -1) # Rose
                    # else: # DEBUG
                    #     print(f"      GIF M2 Inf: Point hors limites ({col_idx}, {row_val}) pour image {frame_for_gif.shape}")
        
        all_processed_bgr_for_gif.append(cv2.cvtColor(frame_for_gif, cv2.COLOR_BGR2RGB))
    print(upper_m1)
    print(upper_m2)

    # --- Fin de la boucle sur les images ---


    # ** Génération du GIF Animé **
    valid_gif_frames = [frame for frame in all_processed_bgr_for_gif if frame is not None]
    if valid_gif_frames:
        gif_name = f"{os.path.basename(series_dir_path)}_surfaces_from_masks.gif"
        gif_path = os.path.join(output_dir, gif_name)
        imageio.mimsave(gif_path, valid_gif_frames, fps=GIF_FPS, loop=0) # loop=0 pour boucle infinie
        print(f"GIF animé sauvegardé : {gif_path}")

   # ** Reconstruction 3D et Graphe de la Surface Supérieure du Masque 1 (ILM) **
    surface_to_reconstruct_name = "Surface Supérieure Masque 1 (ILM)"
    curves_for_3d = all_upper_coords_m1
    # Si tu n'utilises pas de recalage Z basé sur min_rows, tu n'as pas besoin de all_min_rows_upper_m1 ici.
    
    # Filtrer les données pour la reconstruction
    valid_curves_for_3d, valid_angles_for_3d, valid_widths_for_3d = [], [], [] # Pas besoin de valid_min_rows_for_recal_3d
    for i, curve in enumerate(curves_for_3d): # curves_for_3d est all_upper_coords_m1
        if curve is not None and \
           all_oct_widths[i] is not None and \
           all_angles_rad[i] is not None and \
           len(curve) == all_oct_widths[i]: # Assure-toi que all_oct_widths est bien rempli
            valid_curves_for_3d.append(curve)
            valid_angles_for_3d.append(all_angles_rad[i])
            valid_widths_for_3d.append(all_oct_widths[i])
            
    if not valid_curves_for_3d:
        print(f"Aucune courbe valide pour '{surface_to_reconstruct_name}' pour la reconstruction 3D.")
    else:
        print(f"Reconstruction 3D de '{surface_to_reconstruct_name}' avec {len(valid_curves_for_3d)} courbes.")
        
        # Appel à la fonction de transformation SANS recalage Z interne
        points_3d_reconstructed = transform_surface_points_to_3d(
            valid_curves_for_3d, 
            valid_angles_for_3d, 
            valid_widths_for_3d
        )
        
        if points_3d_reconstructed.size > 0:
            MAX_POINTS_FOR_3D_PLOT = 5000 # Par exemple, limite à 50 000 points pour l'affichage
            
            if points_3d_reconstructed.shape[0] > MAX_POINTS_FOR_3D_PLOT:
                print(f"  Sous-échantillonnage des points 3D pour l'affichage ({points_3d_reconstructed.shape[0]} -> {MAX_POINTS_FOR_3D_PLOT}).")
                random_indices = np.random.choice(points_3d_reconstructed.shape[0], MAX_POINTS_FOR_3D_PLOT, replace=False)
                points_for_plot = points_3d_reconstructed[random_indices, :]
            else:
                points_for_plot = points_3d_reconstructed

            plot_title = f"{surface_to_reconstruct_name} 3D - {os.path.basename(series_dir_path)}"
            plot_3d_surface_matplotlib(
                points_for_plot, # Utiliser les points sous-échantillonnés
                title=plot_title, 
                is_mesh=PLOT_3D_AS_MESH,  
                grid_resolution_factor=0.8 # Ce facteur sera maintenant appliqué sur moins de points
            )

            # ** Modèle Mathématique **
            print(f"\nAjustement du modèle Gaussien sur '{surface_to_reconstruct_name}'...")
            x_data = points_3d_reconstructed[:, 0]
            y_data = points_3d_reconstructed[:, 1]
            z_data = points_3d_reconstructed[:, 2] # Ces Z ne sont PAS recalés pour avoir leur sommet à 0

            if z_data.size > 10 :
                # Pour le guess initial, il faut maintenant considérer que Z n'est pas recalé
                # L'offset sera la hauteur moyenne ou min des Z.
                # L'amplitude sera la différence entre le max et le min des Z (la profondeur de la fovéa).
                z_min_guess = np.min(z_data)
                z_max_guess = np.max(z_data) # Le "sommet" de la surface environnante
                
                # Si c'est une dépression, z_min est le fond, z_max est le bord.
                # Amplitude = z_max - z_min (si l'exponentielle est négative et représente la chute)
                # Ou si le modèle est Z = offset_fond + Amplitude_bosse * exp(...)
                # Pour notre modèle: Z(x,y) = offset_bord + Amplitude_creux * exp(...)
                # Amplitude_creux sera négative si Z augmente avec la profondeur (row plus grande)
                # Ou positive si Z diminue avec la profondeur (row plus petite).
                # Dans notre cas, Z = row, donc Z augmente avec la profondeur.
                # Le modèle est offset_base_surface + Amplitude_gaussienne_positive * exp(...)
                # où offset_base_surface est la valeur Z des bords de la fovéa, et Amplitude_gaussienne_positive est la profondeur.
                
                # Nouveau guess:
                # offset: la valeur Z "autour" de la fovéa (plus petites valeurs de row, donc plus petits Z)
                # amplitude: la profondeur de la fovéa (différence entre le fond et les bords)
                # Notre modèle : Z = offset + A * exp(...). Si Z est la coordonnée row,
                # offset sera la valeur de row au bord de la fovéa, et A sera la profondeur (positive).
                
                offset_guess = np.percentile(z_data, 10) # Estimer le Z des bords (plus petites valeurs de Z si row est Z)
                amplitude_guess = np.percentile(z_data, 90) - offset_guess # Profondeur
                if amplitude_guess <=0: amplitude_guess = 50 # S'assurer qu'elle est positive

                xo_guess = np.mean(x_data); yo_guess = np.mean(y_data)
                sx_guess = np.std(x_data) if np.std(x_data) > 1 else 50
                sy_guess = np.std(y_data) if np.std(y_data) > 1 else 50
                
                initial_guess = [amplitude_guess, xo_guess, yo_guess, sx_guess, sy_guess, offset_guess]
                print(f"  Guess initial (Z non recalé): {initial_guess}")
                
                bounds_lower = [0,       -np.inf,      -np.inf,      1e-1,    1e-1,    -np.inf] # Offset peut être n'importe quelle valeur Z
                bounds_upper = [np.inf,  np.inf,       np.inf,       np.inf,  np.inf,  np.inf]
                
                try:
                    popt, pcov = curve_fit(gaussian_2d_model, (x_data, y_data), z_data, 
                                           p0=initial_guess, maxfev=10000, bounds=(bounds_lower, bounds_upper))
                    
                    amp, xo, yo, sx, sy, off = popt
                    print("\nParamètres du modèle Gaussien ajusté:")
                    print(f"  Amplitude (profondeur relative à l'offset) = {amp:.2f} pixels") # Interprétation change
                    print(f"  Centre Xo = {xo:.2f}, Yo = {yo:.2f} pixels")
                    print(f"  Sigma X = {sx:.2f}, Sigma Y = {sy:.2f} pixels")
                    print(f"  Offset Z (valeur de base de la surface) = {off:.2f} pixels") # Interprétation change
                    print("\nModèle approximatif:")
                    print(f"  Z(x,y) = {off:.2f} + {amp:.2f} * exp( - ( ((x-{xo:.2f})^2 / (2*{sx:.2f}^2)) + ((y-{yo:.2f})^2 / (2*{sy:.2f}^2)) ) )")

                except RuntimeError as e:
                    print(f"  Erreur: L'ajustement n'a pas convergé. {e}")
                except Exception as e:
                    print(f"  Erreur lors de l'ajustement du modèle: {e}")
            else:
                print("  Pas assez de points 3D pour l'ajustement du modèle.")
        else:
            print(f"Aucun point 3D pour '{surface_to_reconstruct_name}' pour la modélisation.")
            
    print(f"\nTraitement de la série '{os.path.basename(series_dir_path)}' terminé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traite une série d'images OCT et leurs masques pour modéliser la dépression fovéolaire.")
    parser.add_argument("series_name", type=str, help="Nom de la série (ex: 01_CAB_OD).")
    parser.add_argument("--base_image_dir", type=str, default="IMAGES/IMAGES", help="Répertoire de base des images OCT.")
    parser.add_argument("--output_dir", type=str, default="output_final", help="Répertoire de sortie.")
    
    args = parser.parse_args()

    series_dir_path = os.path.join(args.base_image_dir, args.series_name)

    
    process_oct_series(series_dir_path, args.output_dir)