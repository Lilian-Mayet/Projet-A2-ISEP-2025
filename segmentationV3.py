import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import random # Pour le sous-échantillonnage aléatoire

# --- Configuration ---
BASE_MASK_DIR = 'R_BIN1/R_BIN1' # Ou 'R_BIN2/R_BIN2'
PATIENT_SERIES_SUBDIR = '01_CAB_OD'

MASK_SERIES_DIR = os.path.join(BASE_MASK_DIR, PATIENT_SERIES_SUBDIR)

BOTTOM_CROP_PIXELS_MASK = 0 # Si les masques doivent être rognés en bas

INITIAL_ANGLE_DEGREES = 0
ANGLE_INCREMENT_DEGREES = 3.75

PIXEL_OF_INTEREST_VALUE = 255 # La valeur des pixels qui constituent la forme dans le masque
# Sous-échantillonnage
MAX_POINTS_PER_MASK = 200

# --- Fonctions ---

def get_uppermost_surface_pixels(mask_image, pixel_value_of_interest, max_points_to_return, mask_width_for_sampling_ref):
    """
    Pour chaque colonne du masque, trouve le pixel le plus haut (plus petite coordonnée y/row)
    ayant la valeur `pixel_value_of_interest`.
    Retourne une liste de points (col, row).
    Sous-échantillonne si le nombre de points trouvés dépasse max_points_to_return.
    """
    if mask_image is None:
        return []

    height, width = mask_image.shape
    uppermost_points = []

    for col in range(width):
        # Récupère toutes les coordonnées 'row' des pixels d'intérêt dans cette colonne
        rows_with_pixel = np.where(mask_image[:, col] == pixel_value_of_interest)[0]
        
        if rows_with_pixel.size > 0:
            # Le premier pixel blanc en partant du haut est celui avec le min(row)
            uppermost_row_in_col = np.min(rows_with_pixel)
            uppermost_points.append((col, uppermost_row_in_col))
            
    if not uppermost_points:
        return []

    # Sous-échantillonnage si nécessaire (le nombre de points sera au max 'width')
    if len(uppermost_points) > max_points_to_return:
        # print(f"    Sous-échantillonnage de la surface supérieure de {len(uppermost_points)} à {max_points_to_return} points.")
        sampled_points = random.sample(uppermost_points, max_points_to_return)
        return sampled_points
    else:
        return uppermost_points
def transform_2d_mask_points_to_3d(points_2d_list, angle_rad, mask_width):
    """
    Transforme une liste de points 2D (col, row) d'un masque en points 3D,
    en considérant une rotation autour de la colonne centrale du masque.

    Args:
        points_2d_list (list): Liste de tuples (col, row) des points du masque.
                               'col' est la coordonnée horizontale dans le masque.
                               'row' est la coordonnée verticale (profondeur/hauteur) dans le masque.
        angle_rad (float): Angle de la coupe radiale en radians.
        mask_width (int): Largeur totale du masque 2D.

    Returns:
        list: Liste de listes [x_3d, y_3d, z_3d] représentant les points en 3D.
    """
    points_3d = []
    
    # 1. Déterminer le centre de rotation horizontal dans le masque
    # C'est la colonne de pixels du milieu.
    # Si mask_width est 768, center_x_mask est 384.0.
    # Un pixel à la colonne 0 aura u = 0 - 384 = -384.
    # Un pixel à la colonne 384 aura u = 384 - 384 = 0.
    # Un pixel à la colonne 767 aura u = 767 - 384 = 383.
    center_x_mask = mask_width / 2.0
    total_Z = 0
    for p_2d in points_2d_list:
        col_in_mask, row_in_mask = p_2d

        # 2. Calculer 'u': la distance horizontale du pixel par rapport au centre de rotation.
        # C'est le "rayon" dans le plan de la coupe 2D pour ce pixel.
        # Si col_in_mask < center_x_mask, u est négatif.
        # Si col_in_mask > center_x_mask, u est positif.
        u = col_in_mask - center_x_mask

        # 3. La coordonnée 'row_in_mask' (verticale dans le masque) devient directement la coordonnée Z (hauteur/profondeur) en 3D.
        # C'est ce qui crée les "ondulations, les hauts et les bas" de la surface.
        # Une valeur de 'row' plus petite signifie plus "haut" dans l'image du masque (généralement plus proche du vitré).
        # Une valeur de 'row' plus grande signifie plus "bas" (plus profond dans la rétine).
        # L'inversion de l'axe Z dans plot_all_3d_points gère l'aspect visuel de la "dépression".
        z_3d = row_in_mask
        total_Z+=z_3d

        # 4. Projeter 'u' dans le plan XY 3D en utilisant l'angle de la coupe.
        # C'est la rotation de la "baguette" de longueur 'u' (qui est sur l'axe X local de la coupe)
        # autour de l'axe Z global.
        x_3d = u * np.cos(angle_rad)
        y_3d = u * np.sin(angle_rad)
        
        points_3d.append([x_3d, y_3d, z_3d])
    print("AVERAGE Z VALUES")
    print(total_Z/len(points_2d_list))
    return points_3d

def plot_all_3d_points(all_points_3d, title="Visualisation 3D des Contours de Masques"):
    """Affiche tous les points 3D collectés."""
    if not all_points_3d:
        print("Aucun point 3D à afficher.")
        return

    np_all_points_3d = np.array(all_points_3d)
    if np_all_points_3d.ndim != 2 or np_all_points_3d.shape[0] == 0 or np_all_points_3d.shape[1] != 3:
        print(f"Format de points 3D incorrect pour l'affichage. Shape: {np_all_points_3d.shape}")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np_all_points_3d[:, 0], np_all_points_3d[:, 1], np_all_points_3d[:, 2],
               s=1, c=np_all_points_3d[:, 2], cmap='viridis', alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Profondeur Z (pixels dans le masque)")
    ax.set_title(title)
    ax.invert_zaxis()
    plt.show()

# --- Script Principal ---
if __name__ == "__main__":
    mask_files = sorted(glob.glob(os.path.join(MASK_SERIES_DIR, '*.png')))
    mask_files = mask_files
    if not mask_files: mask_files = sorted(glob.glob(os.path.join(MASK_SERIES_DIR, '*.tif')))

    if not mask_files:
        print(f"Aucun fichier de masque trouvé dans {MASK_SERIES_DIR}")
        exit()

    num_masks = len(mask_files)
    print(f"Trouvé {num_masks} fichiers de masque dans {MASK_SERIES_DIR}")

    collected_3d_points = []
    current_mask_width_ref = None

    for i, mask_path in enumerate(mask_files):
        current_angle_deg = INITIAL_ANGLE_DEGREES + (i * ANGLE_INCREMENT_DEGREES)
        current_angle_rad = np.deg2rad(current_angle_deg)
        
        print(f"Traitement du masque {os.path.basename(mask_path)} (Angle: {current_angle_deg:.2f}°)")

        mask_original = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_original is None:
            print(f"  Erreur: Impossible de charger {mask_path}")
            continue

        mask_processed = mask_original.copy()
        if BOTTOM_CROP_PIXELS_MASK > 0:
            if mask_processed.shape[0] > BOTTOM_CROP_PIXELS_MASK:
                mask_processed = mask_processed[:-BOTTOM_CROP_PIXELS_MASK, :]
            else:
                print(f"  Alerte: Masque trop petit pour rogner.")
        
        if mask_processed.ndim != 2 or mask_processed.shape[0] == 0 or mask_processed.shape[1] == 0:
            print(f"  Erreur: Masque invalide après rognage. Dimensions: {mask_processed.shape}")
            continue

        current_mask_height, current_mask_width = mask_processed.shape

        if current_mask_width_ref is None:
            current_mask_width_ref = current_mask_width
        elif current_mask_width_ref != current_mask_width:
            print(f"  Alerte: Largeur du masque changée! Attendu {current_mask_width_ref}, obtenu {current_mask_width}.")

        MAX_SURFACE_POINTS_TO_KEEP = 300
        # Extraire les points des contours du masque traité
        contour_points_2d = get_uppermost_surface_pixels(
            mask_processed,
            PIXEL_OF_INTEREST_VALUE, 
            MAX_SURFACE_POINTS_TO_KEEP,
            current_mask_width # La largeur du mask_processed
        )
        
        if not contour_points_2d:
            print(f"  Aucun contour trouvé ou extrait du masque traité.")
            continue
        print(f"  Extrait {len(contour_points_2d)} points de contour (après sous-échantillonnage si besoin).")

        points_3d_from_this_mask = transform_2d_mask_points_to_3d(contour_points_2d, 
                                                                  current_angle_rad, 
                                                                  current_mask_width)
        collected_3d_points.extend(points_3d_from_this_mask)

    if collected_3d_points:
        print(f"\nTotal de {len(collected_3d_points)} points de contour 3D collectés.")
        plot_all_3d_points(collected_3d_points)
    else:
        print("\nAucun point de contour 3D n'a été généré.")

    print("\nTerminé.")