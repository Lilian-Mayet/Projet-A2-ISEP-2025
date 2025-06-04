import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_IMAGE_DIR_ROOT = 'IMAGES/IMAGES'  # Dossier racine contenant les 8 sous-dossiers
BASE_MASK_DIR_ROOT = 'R_BIN1/R_BIN1' # Dossier racine des masques correspondants

# Prétraitement des images (identique à avant)
FUNDUS_IMAGE_WIDTH = 495
BOTTOM_CROP_PIXELS = 100

# Paramètres pour le réseau de neurones
IMG_RESIZE_HEIGHT = 128  # Hauteur cible pour l'entraînement
IMG_RESIZE_WIDTH = 128   # Largeur cible pour l'entraînement
CHANNELS = 1             # Niveaux de gris

EPOCHS = 10 # Nombre d'époques 
BATCH_SIZE = 4 # Taille du lot (à ajuster en fonction de la VRAM)
VALIDATION_SPLIT = 0.1 # 10% pour la validation

MODEL_SAVE_PATH = 'oct_segmentation_unet.keras' # Keras v3+ format
# MODEL_SAVE_PATH = 'oct_segmentation_unet.h5' # Ancien format HDF5

# --- Fonctions de Prétraitement des Données ---

def preprocess_for_nn(full_bgr_image, original_mask_gt, target_height, target_width):
    """
    Prétraite une image et son masque pour l'entraînement du réseau.
    1. Découpe l'image originale pour obtenir l'OCT.
    2. Rogne le bas de l'OCT et du masque.
    3. Redimensionne l'OCT et le masque.
    4. Normalise l'OCT.
    5. S'assure que le masque est binaire (0 ou 1).
    """
    # 1. Découpage
    oct_b_scan_full = full_bgr_image[:, FUNDUS_IMAGE_WIDTH:]
    
    processed_oct_bgr = None
    processed_mask = None
    original_cropped_oct_shape = None # Pour le redimensionnement inverse

    if oct_b_scan_full.shape[0] <= BOTTOM_CROP_PIXELS:
        processed_oct_bgr = oct_b_scan_full.copy()
    else:
        processed_oct_bgr = oct_b_scan_full[:-BOTTOM_CROP_PIXELS, :]
    
    original_cropped_oct_shape = processed_oct_bgr.shape[:2] # (height, width)
    gray_oct_processed = cv2.cvtColor(processed_oct_bgr, cv2.COLOR_BGR2GRAY)

    if original_mask_gt is not None:
        # Adapter le masque aux dimensions de l'OCT avant rognage du bas
        if original_mask_gt.shape[0] == oct_b_scan_full.shape[0] and \
           original_mask_gt.shape[1] == oct_b_scan_full.shape[1]:
            if original_mask_gt.shape[0] > BOTTOM_CROP_PIXELS:
                processed_mask = original_mask_gt[:-BOTTOM_CROP_PIXELS, :]
            else:
                processed_mask = original_mask_gt.copy()
        elif original_mask_gt.shape[0] == gray_oct_processed.shape[0] and \
             original_mask_gt.shape[1] == gray_oct_processed.shape[1]:
             processed_mask = original_mask_gt.copy()
        else:
            # Si dimensions incompatibles, on ne peut pas l'utiliser
            return None, None, None

    # 3. Redimensionnement
    resized_oct = cv2.resize(gray_oct_processed, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    if processed_mask is not None:
        resized_mask = cv2.resize(processed_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        # 5. S'assurer que le masque est binaire (0 ou 1) et ajouter un canal
        resized_mask = (resized_mask > 128).astype(np.float32) # Seuil à mi-chemin, convertit en 0 ou 1
        resized_mask = np.expand_dims(resized_mask, axis=-1) # (H, W, 1)
    else:
        return None, None, None # On a besoin du masque pour l'entraînement

    # 4. Normalisation de l'OCT et ajout d'un canal
    resized_oct = resized_oct.astype(np.float32) / 255.0
    resized_oct = np.expand_dims(resized_oct, axis=-1) # (H, W, 1)
            
    return resized_oct, resized_mask, original_cropped_oct_shape


def load_dataset(base_img_dir, base_mask_dir, target_height, target_width):
    images = []
    masks = []
    original_shapes = [] # Pour le redimensionnement inverse lors de la prédiction

    patient_folders = sorted([d for d in os.listdir(base_img_dir) if os.path.isdir(os.path.join(base_img_dir, d))])
    
    for patient_folder in patient_folders:
        img_folder_path = os.path.join(base_img_dir, patient_folder)
        mask_folder_path = os.path.join(base_mask_dir, patient_folder)

        if not os.path.isdir(mask_folder_path):
            print(f"Attention: Dossier de masques non trouvé pour {patient_folder}, ignoré.")
            continue

        img_files = sorted(glob.glob(os.path.join(img_folder_path, '*.png'))) # Adapte si autres formats
        if not img_files: img_files = sorted(glob.glob(os.path.join(img_folder_path, '*.tif')))
        if not img_files: img_files = sorted(glob.glob(os.path.join(img_folder_path, '*.jpg')))


        print(f"Chargement depuis: {patient_folder} ({len(img_files)} images)")
        for img_path in img_files:
            base_filename = os.path.basename(img_path)
            mask_path = os.path.join(mask_folder_path, base_filename[:-4]+"_BIN1.png")

            if not os.path.exists(mask_path):
                print(f"  Masque {mask_path} non trouvé, image ignorée.")
                continue

            full_bgr_img = cv2.imread(img_path)
            mask_gt_original = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if full_bgr_img is None or mask_gt_original is None:
                # print(f"  Erreur chargement image/masque pour {base_filename}, ignoré.")
                continue
            
            processed_oct, processed_mask, orig_shape = preprocess_for_nn(
                full_bgr_img, mask_gt_original, target_height, target_width
            )

            if processed_oct is not None and processed_mask is not None:
                images.append(processed_oct)
                masks.append(processed_mask)
                original_shapes.append(orig_shape) # On ne l'utilise pas pour l'entraînement mais bon à avoir
            # else:
                # print(f"  Erreur prétraitement pour {base_filename}, ignoré.")


    return np.array(images), np.array(masks)

# --- Définition du Modèle U-Net ---
def unet_model(input_size=(IMG_RESIZE_HEIGHT, IMG_RESIZE_WIDTH, CHANNELS)):
    inputs = Input(input_size)
    
    # Encodeur
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c_middle = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c_middle = Dropout(0.2)(c_middle)
    c_middle = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c_middle)
    
    # Décodeur
    u2 = UpSampling2D((2, 2))(c_middle)
    u2 = concatenate([u2, c2])
    c_up2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u2)
    c_up2 = Dropout(0.1)(c_up2)
    c_up2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c_up2)
    
    u1 = UpSampling2D((2, 2))(c_up2)
    u1 = concatenate([u1, c1])
    c_up1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u1)
    c_up1 = Dropout(0.1)(c_up1)
    c_up1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c_up1)
    
    # Sortie : un seul canal avec activation sigmoïde pour la segmentation binaire
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c_up1)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# --- Fonction de Prédiction ---
def predict_mask_from_image_path(image_path, model, target_height, target_width):
    full_bgr_img = cv2.imread(image_path)
    if full_bgr_img is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return None, None

    # Prétraitement (sans le masque, car c'est ce qu'on veut prédire)
    # 1. Découpage
    oct_b_scan_full = full_bgr_img[:, FUNDUS_IMAGE_WIDTH:]
    
    processed_oct_bgr = None
    if oct_b_scan_full.shape[0] <= BOTTOM_CROP_PIXELS:
        processed_oct_bgr = oct_b_scan_full.copy()
    else:
        processed_oct_bgr = oct_b_scan_full[:-BOTTOM_CROP_PIXELS, :]
    
    original_cropped_oct_shape = processed_oct_bgr.shape[:2] # (height, width)
    gray_oct_processed = cv2.cvtColor(processed_oct_bgr, cv2.COLOR_BGR2GRAY)

    # Redimensionnement pour le modèle
    resized_oct_for_model = cv2.resize(gray_oct_processed, (target_width, target_height), interpolation=cv2.INTER_AREA)
    resized_oct_for_model = resized_oct_for_model.astype(np.float32) / 255.0
    resized_oct_for_model = np.expand_dims(resized_oct_for_model, axis=-1) # (H, W, 1)
    resized_oct_for_model = np.expand_dims(resized_oct_for_model, axis=0)  # (1, H, W, 1) pour le batch

    # Prédiction
    predicted_mask_resized = model.predict(resized_oct_for_model)[0] # (H_resized, W_resized, 1)

    # Redimensionnement du masque prédit à la taille originale de l'OCT rognée
    # Seuil pour binariser avant de redimensionner (meilleure qualité souvent)
    predicted_mask_binary_resized = (predicted_mask_resized > 0.5).astype(np.uint8) * 255
    
    predicted_mask_original_size = cv2.resize(
        predicted_mask_binary_resized, 
        (original_cropped_oct_shape[1], original_cropped_oct_shape[0]), # (width, height) pour cv2.resize
        interpolation=cv2.INTER_NEAREST
    )
    
    return gray_oct_processed, predicted_mask_original_size


# --- Entraînement ---
if __name__ == "__main__":
    print("Chargement du dataset...")
    images, masks = load_dataset(BASE_IMAGE_DIR_ROOT, BASE_MASK_DIR_ROOT, IMG_RESIZE_HEIGHT, IMG_RESIZE_WIDTH)
    
    if len(images) == 0:
        print("Aucune image/masque n'a été chargé. Vérifiez les chemins et les fichiers.")
        exit()

    print(f"Dataset chargé: {len(images)} images, {len(masks)} masques.")
    print(f"Shape des images: {images.shape}, Shape des masques: {masks.shape}")

    # Division en train/validation
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=VALIDATION_SPLIT, random_state=42)
    print(f"Entraînement: {len(X_train)} échantillons, Validation: {len(X_val)} échantillons.")

    # Création et compilation du modèle
    model = unet_model(input_size=(IMG_RESIZE_HEIGHT, IMG_RESIZE_WIDTH, CHANNELS))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy']) # Tu pourrais ajouter Dice/IoU ici
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True) # Patience de 10 époques

    print("\nDébut de l'entraînement...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping]
    )

    print("\nEntraînement terminé.")
    # Le modèle est déjà sauvegardé par ModelCheckpoint s'il s'est amélioré.
    # Si tu veux forcer la sauvegarde du dernier état (même si moins bon) :
    # model.save(MODEL_SAVE_PATH)
    print(f"Modèle sauvegardé (ou meilleur modèle sauvegardé) dans: {MODEL_SAVE_PATH}")

    # --- Exemple de prédiction après entraînement ---
    print("\nExemple de prédiction sur une image de validation:")
    if len(X_val) > 0:
        # Charger le meilleur modèle sauvegardé pour la prédiction
        # (EarlyStopping avec restore_best_weights=True devrait déjà l'avoir fait,
        # mais c'est bien de le recharger explicitement pour un test propre)
        try:
            loaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            print("Modèle chargé avec succès pour la prédiction.")

            # Pour trouver le chemin original d'une image de X_val, c'est un peu compliqué ici
            # car on n'a pas gardé les chemins. Prenons une image au hasard du dataset original.
            
            # Cherchons une image de test dans le dataset original
            test_image_path = None
            patient_folders_test = sorted([d for d in os.listdir(BASE_IMAGE_DIR_ROOT) if os.path.isdir(os.path.join(BASE_IMAGE_DIR_ROOT, d))])
            if patient_folders_test:
                # Prenons la première image du premier patient qui a des images
                for pf_test in patient_folders_test:
                    img_folder_test = os.path.join(BASE_IMAGE_DIR_ROOT, pf_test)
                    img_files_test = sorted(glob.glob(os.path.join(img_folder_test, '*.png'))) # Adapte
                    if not img_files_test: img_files_test = sorted(glob.glob(os.path.join(img_folder_test, '*.tif')))
                    if not img_files_test: img_files_test = sorted(glob.glob(os.path.join(img_folder_test, '*.jpg')))
                    if img_files_test:
                        test_image_path = img_files_test[0]
                        break
            
            if test_image_path:
                print(f"Prédiction sur l'image: {test_image_path}")
                original_oct_roi, predicted_mask = predict_mask_from_image_path(
                    test_image_path,
                    loaded_model, 
                    IMG_RESIZE_HEIGHT, 
                    IMG_RESIZE_WIDTH
                )

                if original_oct_roi is not None and predicted_mask is not None:
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.imshow(original_oct_roi, cmap='gray')
                    plt.title("OCT Original (ROI)")
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.imshow(predicted_mask, cmap='gray')
                    plt.title("Masque Prédit (Taille Originale ROI)")
                    plt.axis('off')
                    plt.show()
            else:
                print("Impossible de trouver une image de test pour la prédiction.")

        except Exception as e:
            print(f"Erreur lors du chargement du modèle ou de la prédiction: {e}")
    else:
        print("Pas d'images de validation pour un exemple de prédiction.")