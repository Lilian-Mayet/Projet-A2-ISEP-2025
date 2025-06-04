import numpy as np
BOTTOM_CROP_PIXELS_OCT = 100 # Pour l'image OCT
BOTTOM_CROP_PIXELS_MASK = 0 # Si les masques sont déjà à la bonne hauteur après découpage OCT

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
