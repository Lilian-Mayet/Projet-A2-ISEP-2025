import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import imageio # Pour les GIFs
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import argparse # Pour prendre le répertoire en argument de ligne de commande
import random # Si on garde le sous-échantillonnage

# --- Configuration ---
FUNDUS_IMAGE_WIDTH = 495
BOTTOM_CROP_PIXELS_OCT = 100 # Pour l'image OCT
BOTTOM_CROP_PIXELS_MASK = 0 # Si les masques sont déjà à la bonne hauteur après découpage OCT

INITIAL_ANGLE_DEGREES = 0.0
ANGLE_INCREMENT_DEGREES = 3.75

# Pour l'extraction de la surface du masque
PIXEL_OF_INTEREST_IN_MASK = 255 # Valeur des pixels d'intérêt dans le masque

GIF_FPS = 5

# --- Fonctions Utilitaires ---

def preprocess_oct_image(full_bgr_image):
    if full_bgr_image is None: return None, None
    # Extraire la partie OCT B-scan
    oct_b_scan_full = full_bgr_image[:, FUNDUS_IMAGE_WIDTH:]
    
    # Rogner le bas
    processed_oct_bgr = oct_b_scan_full[:-BOTTOM_CROP_PIXELS_OCT, :] if oct_b_scan_full.shape[0] > BOTTOM_CROP_PIXELS_OCT else oct_b_scan_full.copy()
    
    if processed_oct_bgr.size == 0: return None, None
    gray_oct_processed = cv2.cvtColor(processed_oct_bgr, cv2.COLOR_BGR2GRAY)
    return gray_oct_processed, processed_oct_bgr

def preprocess_mask(mask_original_loaded, target_shape_from_oct=None):
    """Prétraite le masque (rognage si nécessaire, et s'assure qu'il correspond à la taille de l'OCT)."""
    if mask_original_loaded is None: return None
    
    mask_processed = mask_original_loaded.copy()

    # 1. Rogner le bas du masque si configuré
    if BOTTOM_CROP_PIXELS_MASK > 0:
        if mask_processed.shape[0] > BOTTOM_CROP_PIXELS_MASK:
            mask_processed = mask_processed[:-BOTTOM_CROP_PIXELS_MASK, :]
        else:
            print(f"  Alerte: Masque ({os.path.basename(mask_path_being_processed)}) trop petit ({mask_processed.shape[0]}px) pour rogner {BOTTOM_CROP_PIXELS_MASK}px.")
    
    if mask_processed.size == 0: return None

    # 2. Vérifier et ajuster la taille pour correspondre à l'OCT traitée (target_shape_from_oct)
    if target_shape_from_oct is not None:
        if mask_processed.shape != target_shape_from_oct:
            print(f"  Alerte: Le masque ({os.path.basename(mask_path_being_processed)}) de taille {mask_processed.shape} ne correspond pas à l'OCT traitée {target_shape_from_oct}. Tentative de redimensionnement.")
            # Tenter un redimensionnement. INTER_NEAREST est bon pour les masques binaires.
            mask_processed = cv2.resize(mask_processed, (target_shape_from_oct[1], target_shape_from_oct[0]), interpolation=cv2.INTER_NEAREST)
            if mask_processed.shape != target_shape_from_oct: # Double vérification
                print(f"    Erreur de redimensionnement du masque. Masque ignoré.")
                return None
    return mask_processed

def transform_surface_points_to_3d(surface_y_coords_list, angles_rad, image_widths_list):
    """
    Transforme une liste de courbes de surface 2D (coordonnées y pour chaque x) 
    en un nuage de points 3D. Pas de recalage Z interne ici.

    Args:
        surface_y_coords_list (list): Liste d'arrays NumPy, où chaque array contient les 
                                      coordonnées y de la surface pour une coupe. 
                                      Peut contenir -1 pour les points non valides.
        angles_rad (list): Liste des angles de scan en radians, correspondant à chaque courbe.
        image_widths_list (list): Liste des largeurs des images/masques originaux,
                                  correspondant à chaque courbe.

    Returns:
        np.array: Array NumPy de points (N, 3) [x, y, z] ou un array vide si aucun point.
    """
    points_3d = []
    for i, surface_y_coords in enumerate(surface_y_coords_list):
        # Vérifier si toutes les données pour cet index sont valides
        if surface_y_coords is None or \
           image_widths_list[i] is None or \
           angles_rad[i] is None:
            continue
        
        # Filtrer les y_coords non valides (ex: -1 si utilisé comme indicateur)
        valid_cols = np.where(surface_y_coords != -1)[0]
        if not valid_cols.size > 0:
            continue

        angle = angles_rad[i]
        width = image_widths_list[i]
        center_x = width / 2.0

        for col in valid_cols:
            row_val = surface_y_coords[col] # C'est la coordonnée y (verticale) dans la coupe 2D
            
            u = col - center_x  # Coordonnée horizontale relative au centre de la coupe
            
            # Transformation en 3D
            x_3d = u * np.cos(angle)
            y_3d = u * np.sin(angle)
            z_3d = row_val  # La coordonnée 'row' devient directement Z
            
            points_3d.append([x_3d, y_3d, z_3d])
            
    return np.array(points_3d)
# --- Fonction pour obtenir les lignes de surface du MASQUE ---
def get_surface_lines_from_mask(processed_mask_image, pixel_value):
    if processed_mask_image is None:
        return None, None

    height, width = processed_mask_image.shape
    upper_y_coords = np.full(width, -1, dtype=int) # -1 indique non trouvé
    lower_y_coords = np.full(width, -1, dtype=int)

    for col in range(width):
        rows_with_pixel = np.where(processed_mask_image[:, col] == pixel_value)[0]
        if rows_with_pixel.size > 0:
            upper_y_coords[col] = np.min(rows_with_pixel)
            lower_y_coords[col] = np.max(rows_with_pixel)
            
    if np.all(upper_y_coords == -1): # Si aucun point du tout (par exemple, masque vide)
        return None, None
        
    return upper_y_coords, lower_y_coords


def plot_3d_surface_matplotlib(points_3d, title="Surface 3D", is_mesh=True, grid_resolution_factor=1.0):
    """
    Affiche les points 3D soit comme un nuage de points, soit comme une surface maillée interpolée.
    
    Args:
        points_3d (np.array): Array de points (N, 3) [x, y, z].
        title (str): Titre du graphique.
        is_mesh (bool): Si True, tente de créer une surface maillée. Sinon, affiche un nuage de points.
        grid_resolution_factor (float): Facteur pour contrôler la résolution de la grille d'interpolation.
                                        1.0 est une résolution standard. >1 plus dense, <1 moins dense.
                                        Une valeur autour de 0.5 ou 0.7 peut être plus rapide pour le test.
    """
    #if points_3d is None or points_3d.ndim != 2 or points_3d.shape[0] < 4 or points_3d.shape[1] != 3: # Besoin d'au moins 4 points pour griddata
    #    print(f"Pas assez de points 3D valides ou format incorrect pour afficher '{title}'. Points: {points_3d.shape if points_3d is not None else 'None'}")
    #    return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_coords = points_3d[:, 0]
    y_coords = points_3d[:, 1]
    z_coords = points_3d[:, 2]

    if is_mesh:
        print(f"  Génération du maillage pour '{title}'...")
        # Déterminer la résolution de la grille
        # Utiliser la plage des données et un nombre de points raisonnable
        num_points_x = int(len(np.unique(x_coords)) * grid_resolution_factor)
        num_points_y = int(len(np.unique(y_coords)) * grid_resolution_factor)
        
        # Assurer un minimum de points pour la grille si peu de points uniques
        num_points_x = max(num_points_x, 50) # Au moins 50 points sur l'axe x de la grille
        num_points_y = max(num_points_y, 50) # Au moins 50 points sur l'axe y de la grille

        # Créer une grille sur laquelle interpoler
        xi = np.linspace(x_coords.min(), x_coords.max(), num_points_x)
        yi = np.linspace(y_coords.min(), y_coords.max(), num_points_y)
        X_grid, Y_grid = np.meshgrid(xi, yi)

        # Interpoler les valeurs Z sur la grille
        # 'linear' est souvent un bon compromis. 'cubic' peut être plus lisse mais plus lent/instable.
        # 'nearest' est rapide mais donne un aspect en blocs.
        try:
            Z_grid = griddata((x_coords, y_coords), z_coords, (X_grid, Y_grid), method='linear')
            print("    Interpolation griddata terminée.")

            # plot_surface n'aime pas les NaNs, bien que Matplotlib essaie de les gérer.
            # On peut masquer les NaNs pour un meilleur rendu, mais cela peut créer des trous.
            # Z_grid_masked = np.ma.masked_invalid(Z_grid) # Optionnel

            # Afficher la surface maillée
            # `rstride` et `cstride` contrôlent la finesse du maillage affiché (pas de l'interpolation)
            surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none', rstride=1, cstride=1, antialiased=True, shade=True, linewidth=0)
            ax.axes.set_zlim3d(bottom=0, top=500)
            # fig.colorbar(surf, shrink=0.5, aspect=10, label="Profondeur Z (recalée)") # Optionnel
            print("    Surface maillée tracée.")
        except Exception as e:
            print(f"    Erreur lors de la création du maillage ou de l'affichage de la surface: {e}")
            print("    Affichage en nuage de points à la place.")
            ax.scatter(x_coords, y_coords, z_coords, s=1, c=z_coords, cmap='viridis', alpha=0.6)
            
    else: # Afficher comme un nuage de points
        ax.scatter(x_coords, y_coords, z_coords, s=1, c=z_coords, cmap='viridis', alpha=0.6)

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_zlabel("Profondeur Z (pixels relatifs)")
    ax.set_title(title)
    ax.invert_zaxis()
    
    # Ajuster l'angle de vue pour une meilleure perspective
    ax.view_init(elev=20, azim=-65) # Tu peux jouer avec ces valeurs (élévation, azimut)
    
    plt.show()

def gaussian_2d_model(xy_tuple, amplitude, xo, yo, sigma_x, sigma_y, offset):
    (x, y) = xy_tuple
    xo = float(xo); yo = float(yo)
    # Z est recalé, donc offset ~ 0, amplitude = profondeur
    g = offset + amplitude * np.exp(-(((x - xo)**2 / (2 * sigma_x**2)) + ((y - yo)**2 / (2 * sigma_y**2))))
    return g

# Variable globale pour stocker le chemin du masque en cours de traitement (pour les messages d'erreur)
mask_path_being_processed = ""

# --- Fonction principale de traitement ---
def process_oct_series(series_dir_path, mask1_dir_path, mask2_dir_path, output_dir="output_project_final"):
    global mask_path_being_processed # Pour utiliser dans preprocess_mask

    if not os.path.isdir(series_dir_path):
        print(f"Erreur: Le répertoire d'images '{series_dir_path}' n'existe pas.")
        return
    if not os.path.isdir(mask1_dir_path):
        print(f"Erreur: Le répertoire de masques 1 '{mask1_dir_path}' n'existe pas.")
        return
    if not os.path.isdir(mask2_dir_path):
        print(f"Erreur: Le répertoire de masques 2 '{mask2_dir_path}' n'existe pas.")
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
        mask_path_being_processed = os.path.join(mask1_dir_path, base_filename)
        # Tenter avec _BIN1.png si le nom direct ne marche pas
        if not os.path.exists(mask_path_being_processed):
            mask_path_png_alt = os.path.join(mask1_dir_path, os.path.splitext(base_filename)[0] + "_BIN1.png")
            if os.path.exists(mask_path_png_alt): mask_path_being_processed = mask_path_png_alt
        
        if os.path.exists(mask_path_being_processed):
            mask1_loaded = cv2.imread(mask_path_being_processed, cv2.IMREAD_GRAYSCALE)
            mask1_processed = preprocess_mask(mask1_loaded, target_shape_for_masks)
            if mask1_processed is not None:
                upper_m1, lower_m1 = get_surface_lines_from_mask(mask1_processed, PIXEL_OF_INTEREST_IN_MASK)
                if upper_m1 is not None:
                    valid_upper_m1_pts = upper_m1[upper_m1 != -1]
                    if valid_upper_m1_pts.size > 0:
                        min_row_um1 = np.min(valid_upper_m1_pts)
        else:
            print(f"    Masque 1 non trouvé: {mask_path_being_processed}")
        
        all_upper_coords_m1.append(upper_m1)
        all_lower_coords_m1.append(lower_m1)
        all_min_rows_upper_m1.append(min_row_um1)

        # ------- TRAITEMENT MASQUE 2 (ex: HRC ou autre limite) --------
        mask_path_being_processed = os.path.join(mask2_dir_path, base_filename)
        if not os.path.exists(mask_path_being_processed):
            mask_path_png_alt = os.path.join(mask2_dir_path, os.path.splitext(base_filename)[0] + "_BIN2.png")
            if os.path.exists(mask_path_png_alt): mask_path_being_processed = mask_path_png_alt

        if os.path.exists(mask_path_being_processed):
            mask2_loaded = cv2.imread(mask_path_being_processed, cv2.IMREAD_GRAYSCALE)
            mask2_processed = preprocess_mask(mask2_loaded, target_shape_for_masks)
            if mask2_processed is not None:
                upper_m2, lower_m2 = get_surface_lines_from_mask(mask2_processed, PIXEL_OF_INTEREST_IN_MASK)
                # if upper_m2 is not None: # Calculer min_row_um2 si on reconstruit cette surface
                #     valid_upper_m2_pts = upper_m2[upper_m2 != -1]
                #     if valid_upper_m2_pts.size > 0: min_row_um2 = np.min(valid_upper_m2_pts)
        else:
            print(f"    Masque 2 non trouvé: {mask_path_being_processed}")

        all_upper_coords_m2.append(upper_m2)
        all_lower_coords_m2.append(lower_m2)
        # all_min_rows_upper_m2.append(min_row_um2) # Si besoin

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
            MAX_POINTS_FOR_3D_PLOT = 1000 # Par exemple, limite à 50 000 points pour l'affichage
            
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
                is_mesh=True, 
                grid_resolution_factor=0.7 # Ce facteur sera maintenant appliqué sur moins de points
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
    parser.add_argument("--base_mask1_dir", type=str, default="R_BIN1/R_BIN1", help="Répertoire de base pour le premier jeu de masques.")
    parser.add_argument("--base_mask2_dir", type=str, default="R_BIN2/R_BIN2", help="Répertoire de base pour le second jeu de masques.")
    parser.add_argument("--output_dir", type=str, default="output_final", help="Répertoire de sortie.")
    
    args = parser.parse_args()

    series_dir_path = os.path.join(args.base_image_dir, args.series_name)
    mask1_dir_path = os.path.join(args.base_mask1_dir, args.series_name)
    mask2_dir_path = os.path.join(args.base_mask2_dir, args.series_name)
    
    process_oct_series(series_dir_path, mask1_dir_path, mask2_dir_path, args.output_dir)