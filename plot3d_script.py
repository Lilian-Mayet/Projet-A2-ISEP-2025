import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

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
            Z_grid = griddata((x_coords, y_coords), z_coords, (X_grid, Y_grid), method='linear',rescale=False)
            print("    Interpolation griddata terminée.")

            # plot_surface n'aime pas les NaNs, bien que Matplotlib essaie de les gérer.
            # On peut masquer les NaNs pour un meilleur rendu, mais cela peut créer des trous.
            # Z_grid_masked = np.ma.masked_invalid(Z_grid) # Optionnel

            # Afficher la surface maillée
            # `rstride` et `cstride` contrôlent la finesse du maillage affiché (pas de l'interpolation)
            surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none', rstride=1, cstride=1, antialiased=True, shade=True, linewidth=0)
            
            # fig.colorbar(surf, shrink=0.5, aspect=10, label="Profondeur Z (recalée)") # Optionnel
            print("    Surface maillée tracée.")
        except Exception as e:
            print(f"    Erreur lors de la création du maillage ou de l'affichage de la surface: {e}")
            print("    Affichage en nuage de points à la place.")
            ax.scatter(x_coords, y_coords, z_coords, s=1, c=z_coords, cmap='viridis', alpha=0.6)
            
    else: # Afficher comme un nuage de points
        ax.scatter(x_coords, y_coords, z_coords, s=1, c=z_coords, cmap='viridis', alpha=0.6)
    ax.axes.set_zlim3d(bottom=0, top=500)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_zlabel("Profondeur Z (pixels relatifs)")
    ax.set_title(title)
    ax.invert_zaxis()
    
    # Ajuster l'angle de vue pour une meilleure perspective
    ax.view_init(elev=20, azim=-65) # Tu peux jouer avec ces valeurs (élévation, azimut)
    
    plt.show()
