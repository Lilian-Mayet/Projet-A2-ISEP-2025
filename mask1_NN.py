import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam # ou AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_IMAGE_DIR_ROOT = 'IMAGES/IMAGES'
BASE_MASK_DIR_ROOT = 'R_BIN1/R_BIN1' # ou R_BIN2

FUNDUS_IMAGE_WIDTH = 495
BOTTOM_CROP_PIXELS = 100

# Taille fixe pour l'entrée du réseau de neurones
NN_INPUT_HEIGHT = 128  # EXEMPLE: Redimensionner à 128x...
NN_INPUT_WIDTH = 256   # EXEMPLE: ...x256 pour le réseau (adapte le ratio si besoin)
CHANNELS = 1

EPOCHS = 20 # Ajuste
BATCH_SIZE = 4 # Ajuste
VALIDATION_SPLIT = 0.1
MODEL_SAVE_PATH = 'oct_segmentation_mask_1.keras'

# --- Fonctions de Prétraitement des Données ---
def preprocess_for_nn_resized_input(full_bgr_image, original_mask_gt, nn_target_h, nn_target_w):
    """
    1. Découpe et rogne l'image originale et le masque GT pour obtenir l'OCT ROI.
    2. Redimensionne l'OCT ROI et son masque à nn_target_h, nn_target_w.
    3. Normalise l'OCT, binarise le masque.
    Retourne l'OCT redimensionnée, le masque redimensionné, et la shape de l'OCT ROI originale.
    """
    # 1. Obtenir l'OCT ROI originale
    oct_b_scan_full = full_bgr_image[:, FUNDUS_IMAGE_WIDTH:]
    
    cropped_oct_bgr = None
    if oct_b_scan_full.shape[0] <= BOTTOM_CROP_PIXELS:
        cropped_oct_bgr = oct_b_scan_full.copy()
    else:
        cropped_oct_bgr = oct_b_scan_full[:-BOTTOM_CROP_PIXELS, :]
    
    original_roi_shape = cropped_oct_bgr.shape[:2] # (height, width) de l'OCT ROI originale
    gray_oct_roi_original = cv2.cvtColor(cropped_oct_bgr, cv2.COLOR_BGR2GRAY)

    # Traiter le masque GT pour qu'il corresponde à l'OCT ROI originale
    cropped_mask_gt = None
    if original_mask_gt is not None:
        if original_mask_gt.shape[0] == oct_b_scan_full.shape[0] and \
           original_mask_gt.shape[1] == oct_b_scan_full.shape[1]: # Masque correspond à l'OCT full
            if original_mask_gt.shape[0] > BOTTOM_CROP_PIXELS:
                cropped_mask_gt = original_mask_gt[:-BOTTOM_CROP_PIXELS, :]
            else:
                cropped_mask_gt = original_mask_gt.copy()
        elif original_mask_gt.shape[0] == original_roi_shape[0] and \
             original_mask_gt.shape[1] == original_roi_shape[1]: # Masque correspond déjà à l'OCT ROI
             cropped_mask_gt = original_mask_gt.copy()
        else:
            # Si le masque GT ne peut pas être aligné avec l'OCT ROI, on ne peut pas l'utiliser
            print(f"  Alerte: Le masque GT original ne peut pas être aligné avec l'OCT ROI. Image ignorée pour l'entraînement.")
            return None, None, None
    else: # Pas de masque GT fourni (peut arriver si on appelle pour prédiction pure)
        pass


    # 2. Redimensionner l'OCT ROI et son masque (si présent) à la taille d'entrée du NN
    nn_input_oct_gray = cv2.resize(gray_oct_roi_original, (nn_target_w, nn_target_h), interpolation=cv2.INTER_AREA)
    
    nn_input_mask = None
    if cropped_mask_gt is not None:
        nn_input_mask = cv2.resize(cropped_mask_gt, (nn_target_w, nn_target_h), interpolation=cv2.INTER_NEAREST)
        # 3. Binariser le masque redimensionné
        nn_input_mask = (nn_input_mask > 128).astype(np.float32)
        nn_input_mask = np.expand_dims(nn_input_mask, axis=-1)
    # elif original_mask_gt is None et on est en mode prediction, nn_input_mask reste None
    
    # 3. Normaliser l'OCT redimensionnée
    nn_input_oct_gray = nn_input_oct_gray.astype(np.float32) / 255.0
    nn_input_oct_gray = np.expand_dims(nn_input_oct_gray, axis=-1)
            
    return nn_input_oct_gray, nn_input_mask, original_roi_shape


def load_dataset(base_img_dir, base_mask_dir, nn_target_h, nn_target_w):
    images_for_nn = []
    masks_for_nn = []
    # original_roi_shapes_for_prediction_later = [] # On pourrait stocker ça si on veut

    patient_folders = sorted([d for d in os.listdir(base_img_dir) if os.path.isdir(os.path.join(base_img_dir, d))])
    for patient_folder in patient_folders:
        img_folder_path = os.path.join(base_img_dir, patient_folder)
        mask_folder_path = os.path.join(base_mask_dir, patient_folder)
        if not os.path.isdir(mask_folder_path): continue

        img_files = sorted(glob.glob(os.path.join(img_folder_path, '*.png')))
        if not img_files: img_files = sorted(glob.glob(os.path.join(img_folder_path, '*.tif')))
        if not img_files: img_files = sorted(glob.glob(os.path.join(img_folder_path, '*.jpg')))

        print(f"Chargement depuis: {patient_folder} ({len(img_files)} images)")
        for img_path in img_files:
            base_filename = os.path.basename(img_path)
            mask_path = os.path.join(mask_folder_path, base_filename[:-4]+"_BIN1.png")
            if not os.path.exists(mask_path): continue

            full_bgr_img = cv2.imread(img_path)
            mask_gt_original = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if full_bgr_img is None or mask_gt_original is None: continue
            
            nn_oct, nn_mask, _ = preprocess_for_nn_resized_input( # On ignore original_roi_shape ici
                full_bgr_img, mask_gt_original, nn_target_h, nn_target_w
            )
            if nn_oct is not None and nn_mask is not None: # On a besoin des deux pour l'entraînement
                images_for_nn.append(nn_oct)
                masks_for_nn.append(nn_mask)
    return np.array(images_for_nn), np.array(masks_for_nn)


# --- Modèle U-Net (défini avec NN_INPUT_HEIGHT, NN_INPUT_WIDTH) ---
def unet_model(input_size=(NN_INPUT_HEIGHT, NN_INPUT_WIDTH, CHANNELS)):
    inputs = Input(input_size)
    # ... (architecture U-Net identique à avant) ...
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    c_middle = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c_middle = Dropout(0.2)(c_middle)
    c_middle = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c_middle)
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
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c_up1)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# --- Fonction de Prédiction ---
def predict_mask_with_resizing(image_path, model, nn_h, nn_w):
    """
    Prédit un masque. L'image est prétraitée (découpage, rognage, puis redimensionnement à nn_h, nn_w).
    Le masque prédit par le NN (taille nn_h, nn_w) est ensuite redimensionné à la taille de l'OCT ROI originale.
    Retourne: l'OCT ROI originale (pour affichage), et le masque prédit à la taille de l'OCT ROI originale.
    """
    full_bgr_img = cv2.imread(image_path)
    if full_bgr_img is None: return None, None, None

    # Prétraiter l'image pour obtenir l'entrée du NN et la forme ROI originale
    # Ici, on ne passe pas de masque GT à preprocess_for_nn_resized_input
    nn_input_oct, _, original_roi_shape = preprocess_for_nn_resized_input(
        full_bgr_img, 
        None, # Pas de masque GT pour la prédiction pure
        nn_h, 
        nn_w
    )

    if nn_input_oct is None or original_roi_shape is None:
        print(f"  Erreur lors du prétraitement de l'image {os.path.basename(image_path)} pour la prédiction.")
        # Essayons quand même de retourner l'OCT ROI si possible, même si le prétraitement pour NN a échoué
        oct_b_scan_full_temp = full_bgr_img[:, FUNDUS_IMAGE_WIDTH:]
        cropped_oct_bgr_temp = oct_b_scan_full_temp[:-BOTTOM_CROP_PIXELS, :] if oct_b_scan_full_temp.shape[0] > BOTTOM_CROP_PIXELS else oct_b_scan_full_temp.copy()
        return cv2.cvtColor(cropped_oct_bgr_temp, cv2.COLOR_BGR2GRAY), None, None


    # Prédiction (le modèle attend un batch)
    nn_input_oct_batch = np.expand_dims(nn_input_oct, axis=0)
    predicted_mask_nn_size = model.predict(nn_input_oct_batch)[0] # Sortie: (nn_h, nn_w, 1)

    # Binariser le masque prédit à la taille du NN
    predicted_mask_nn_size_binary = (predicted_mask_nn_size > 0.5).astype(np.uint8) * 255
    
    # Redimensionner le masque binaire à la taille de l'OCT ROI originale
    predicted_mask_original_roi_size = cv2.resize(
        predicted_mask_nn_size_binary,
        (original_roi_shape[1], original_roi_shape[0]), # (width, height) pour cv2.resize
        interpolation=cv2.INTER_NEAREST
    )
    
    # Obtenir l'OCT ROI originale en niveaux de gris pour l'affichage
    oct_b_scan_full = full_bgr_img[:, FUNDUS_IMAGE_WIDTH:]
    cropped_oct_bgr_display = oct_b_scan_full[:-BOTTOM_CROP_PIXELS, :] if oct_b_scan_full.shape[0] > BOTTOM_CROP_PIXELS else oct_b_scan_full.copy()
    gray_oct_roi_original_display = cv2.cvtColor(cropped_oct_bgr_display, cv2.COLOR_BGR2GRAY)
    
    return gray_oct_roi_original_display, predicted_mask_original_roi_size, original_roi_shape


# Dice Coefficient et evaluate_and_plot_prediction (adapté pour utiliser la nouvelle fonction de prédiction)
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    # ... (inchangé)
    y_true_f = (y_true > 0).flatten().astype(np.bool_)
    y_pred_f = (y_pred > 0).flatten().astype(np.bool_)
    intersection = np.sum(y_true_f & y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def evaluate_and_plot_prediction(image_path, mask_gt_path, model, alpha=0.4):
    print(f"\nÉvaluation pour: {os.path.basename(image_path)}")
    
    # Prédire le masque et obtenir l'OCT ROI originale
    oct_roi_original_gray, predicted_mask_roi_size, original_roi_shape = \
        predict_mask_with_resizing(image_path, model, NN_INPUT_HEIGHT,NN_INPUT_WIDTH)
    
    if oct_roi_original_gray is None or predicted_mask_roi_size is None or original_roi_shape is None:
        print("  Erreur lors de la prédiction ou du traitement de l'image.")
        return

    # Charger le masque GT et le préparer à la taille de l'OCT ROI originale
    mask_gt_original_loaded = cv2.imread(mask_gt_path, cv2.IMREAD_GRAYSCALE)
    full_bgr_img_for_gt = cv2.imread(image_path) # Recharger pour être sûr

    if mask_gt_original_loaded is None or full_bgr_img_for_gt is None:
        print("  Erreur: Impossible de charger l'image originale ou le masque GT pour comparaison.")
        return

    # Utiliser preprocess_for_nn_resized_input juste pour obtenir le masque GT à la bonne taille ROI originale
    # (on ne se soucie pas de la sortie redimensionnée pour le NN ici)
    _, _, _ = preprocess_for_nn_resized_input(full_bgr_img_for_gt, mask_gt_original_loaded,NN_INPUT_HEIGHT,NN_INPUT_WIDTH) # Appel juste pour la logique de découpage/rognage du masque GT

    # Plus simple: découper/rogner le masque GT manuellement pour qu'il corresponde à original_roi_shape
    oct_b_scan_full_for_gt = full_bgr_img_for_gt[:, FUNDUS_IMAGE_WIDTH:]
    mask_gt_roi_size = None
    if mask_gt_original_loaded.shape[0] == oct_b_scan_full_for_gt.shape[0] and \
       mask_gt_original_loaded.shape[1] == oct_b_scan_full_for_gt.shape[1]:
        if mask_gt_original_loaded.shape[0] > BOTTOM_CROP_PIXELS:
            temp_mask = mask_gt_original_loaded[:-BOTTOM_CROP_PIXELS, :]
        else:
            temp_mask = mask_gt_original_loaded.copy()
        # S'assurer que le masque GT rogné a la même taille que l'OCT ROI originale
        if temp_mask.shape[0] == original_roi_shape[0] and temp_mask.shape[1] == original_roi_shape[1]:
            mask_gt_roi_size = temp_mask
        else: # Si même après rognage ça ne correspond pas (ne devrait pas arriver si les données sont cohérentes)
            print(f"  Alerte: Masque GT rogné ({temp_mask.shape}) ne correspond pas à l'OCT ROI originale ({original_roi_shape}). Tentative de redimensionnement.")
            mask_gt_roi_size = cv2.resize(temp_mask, (original_roi_shape[1], original_roi_shape[0]), interpolation=cv2.INTER_NEAREST)

    elif mask_gt_original_loaded.shape[0] == original_roi_shape[0] and \
         mask_gt_original_loaded.shape[1] == original_roi_shape[1]: # Masque original est déjà à la taille ROI
         mask_gt_roi_size = mask_gt_original_loaded.copy()
    else:
        print(f"  Alerte: Masque GT original ({mask_gt_original_loaded.shape}) ne correspond pas à l'OCT ROI originale ({original_roi_shape}). Tentative de redimensionnement.")
        mask_gt_roi_size = cv2.resize(mask_gt_original_loaded, (original_roi_shape[1], original_roi_shape[0]), interpolation=cv2.INTER_NEAREST)


    if mask_gt_roi_size is None:
        print("  Erreur lors de la préparation du masque GT à la taille ROI.")
        return

    # Calculer le Dice Coefficient
    dice_score_str = "N/A"
    if predicted_mask_roi_size.shape == mask_gt_roi_size.shape:
        dice_score = dice_coefficient(mask_gt_roi_size, predicted_mask_roi_size)
        dice_score_str = f"{dice_score:.4f}"
        print(f"  Dice Coefficient: {dice_score_str}")
    else:
        print(f"  Erreur: Tailles du masque GT ({mask_gt_roi_size.shape}) et prédit ({predicted_mask_roi_size.shape}) incompatibles pour Dice.")

    # Visualisation (utilise oct_roi_original_gray, predicted_mask_roi_size, mask_gt_roi_size)
    # ... (la partie plot reste la même, mais s'assure d'utiliser les bonnes variables)
    plt.figure(figsize=(18, 6))
    plt.suptitle(f"Comparaison pour {os.path.basename(image_path)}\nDice: {dice_score_str}", fontsize=14)
    plt.subplot(1, 2, 1)
    plt.imshow(oct_roi_original_gray, cmap='gray')
    pred_mask_colored = np.zeros((*predicted_mask_roi_size.shape, 3), dtype=np.uint8)
    pred_mask_colored[predicted_mask_roi_size > 0] = [255, 0, 0]
    plt.imshow(pred_mask_colored, alpha=alpha)
    plt.title("OCT ROI + Masque Prédit")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(oct_roi_original_gray, cmap='gray')
    gt_mask_colored = np.zeros((*mask_gt_roi_size.shape, 3), dtype=np.uint8)
    gt_mask_colored[mask_gt_roi_size > 0] = [0, 255, 0]
    plt.imshow(gt_mask_colored, alpha=alpha)
    plt.title("OCT ROI + Masque GT")
    plt.axis('off')
    



    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Entraînement (partie principale) ---
if __name__ == "__main__":
    print(f"Utilisation des dimensions d'entrée pour le NN: H={NN_INPUT_HEIGHT}, W={NN_INPUT_WIDTH}")
    training = False
    # --- PARTIE ENTRAINEMENT* ---
    if training : 
        print("Chargement du dataset...")
        images_nn, masks_nn = load_dataset(BASE_IMAGE_DIR_ROOT, BASE_MASK_DIR_ROOT, NN_INPUT_HEIGHT, NN_INPUT_WIDTH)
        if len(images_nn) == 0:
            print("Aucune image/masque n'a été chargé. Vérifiez les chemins, fichiers et dimensions cibles.")
            exit()
        print(f"Dataset chargé: {len(images_nn)} images, {len(masks_nn)} masques.")
        print(f"Shape des images pour NN: {images_nn.shape}, Shape des masques pour NN: {masks_nn.shape}")
        X_train, X_val, y_train, y_val = train_test_split(images_nn, masks_nn, test_size=VALIDATION_SPLIT, random_state=42)
        
        model = unet_model(input_size=(NN_INPUT_HEIGHT, NN_INPUT_WIDTH, CHANNELS))
        def dice_coef_keras(y_true, y_pred, smooth=1e-6): # Métrique Keras
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', dice_coef_keras])
        model.summary()
        checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_dice_coef_keras', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_dice_coef_keras', patience=10, verbose=1, mode='max', restore_best_weights=True)
        print("\nDébut de l'entraînement...")
        history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])
        print("\nEntraînement terminé.")
        print(f"Modèle sauvegardé dans: {MODEL_SAVE_PATH}")
    # --- FIN PARTIE ENTRAINEMENT ---

    # --- PARTIE ÉVALUATION ET VISUALISATION ---
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Modèle non trouvé à {MODEL_SAVE_PATH}. Veuillez d'abord entraîner le modèle.")
        exit()
    print(f"\nChargement du modèle depuis {MODEL_SAVE_PATH} pour évaluation...")
    try:
        def dice_coef_keras(y_true, y_pred, smooth=1e-6): # Doit être défini si utilisé à l'entraînement
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
        loaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'dice_coef_keras': dice_coef_keras} if 'dice_coef_keras' in locals() else None)
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        exit()

    num_test_images_per_patient = 1 
    patient_folders_eval = sorted([d for d in os.listdir(BASE_IMAGE_DIR_ROOT) if os.path.isdir(os.path.join(BASE_IMAGE_DIR_ROOT, d))])
    for patient_folder in patient_folders_eval:
        img_folder_path_eval = os.path.join(BASE_IMAGE_DIR_ROOT, patient_folder)
        mask_folder_path_eval = os.path.join(BASE_MASK_DIR_ROOT, patient_folder)
        if not os.path.isdir(mask_folder_path_eval): continue
        img_files_eval = sorted(glob.glob(os.path.join(img_folder_path_eval, '*.png')))
        if not img_files_eval: img_files_eval = sorted(glob.glob(os.path.join(img_folder_path_eval, '*.tif')))
        if not img_files_eval: img_files_eval = sorted(glob.glob(os.path.join(img_folder_path_eval, '*.jpg')))

        for i in range(min(num_test_images_per_patient, len(img_files_eval))):
            test_img_path = img_files_eval[i]
            test_mask_path = os.path.join(mask_folder_path_eval, os.path.basename(test_img_path)[:-4])
            test_mask_path+="_BIN1.png"
            if os.path.exists(test_mask_path):
                evaluate_and_plot_prediction(test_img_path, test_mask_path, loaded_model)
            else:
                print(f"  Masque GT non trouvé pour {test_img_path}, ignoré pour l'évaluation.")