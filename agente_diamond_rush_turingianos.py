import cv2
import numpy as np
from collections import deque
import pyautogui

from skimage.metrics import structural_similarity as ssim

# --- CONFIG ---
# Diccionario con objetos y su imagen base
templates_raw = {
	"door": cv2.imread("FDoor.png", cv2.IMREAD_UNCHANGED),
	"spike": cv2.imread("FSpikeBGColor.png", cv2.IMREAD_UNCHANGED),
    "diamondBG": cv2.imread("FDiamondBGColor.png", cv2.IMREAD_UNCHANGED),
    "terrain" : cv2.imread("FTerrain.png", cv2.IMREAD_UNCHANGED),
}

template_ladder = cv2.imread("FLadderBG.png", cv2.IMREAD_UNCHANGED)

# Imagen principal
#img = cv2.imread("fullscreenSS.png", cv2.IMREAD_UNCHANGED)
img = cv2.imread("SSexample.png", cv2.IMREAD_UNCHANGED)

# imagen editada
img_res = cv2.imread("SSexample.png", cv2.IMREAD_UNCHANGED)


resXOriginal = 720
resYOriginal = 1280
cellHNumber = 16
cellWNumber = 10
cellHOriginal = 1280/16
cellWOriginal = 720/10
while True:

    img = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # imagen hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    # Convertir el color hexadecimal #10191c a BGR
    target_color = (24, 20, 15)  # Color en formato BGR (Blue, Green, Red)

    target_color2 = (100, 109, 30)  # Color en formato BGR (Blue, Green, Red)

    # Crear un rango de color exacto
    lower_bound = np.array(target_color)  # Límite inferior del color
    upper_bound = np.array(target_color2)  # Límite superior del color

    # Crear una máscara para el color exacto
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    #cv2.imshow("masyyk", mask)
    result = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow("mask", result)
    #cv2.waitKey(0)
    # HAllar contornos del resultado
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dejar los 2 contornos mas grandes
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]


    # Hallar el area entre los dos contornos y dibujarla en rojo
    # Verificar que hay al menos dos contornos
    if len(contours) >= 2:
        # Obtener los rectángulos delimitadores de los dos contornos más grandes
        x1, y1, w1, h1 = cv2.boundingRect(contours[0])
        x2, y2, w2, h2 = cv2.boundingRect(contours[1])

        print(f"Contorno 1: {x1}, {y1}, {w1}, {h1}")
        print(f"Contorno 2: {x2}, {y2}, {w2}, {h2}")

        firstX = min(x1, x2)
        secondX = max(x1, x2)

        img_contours = img.copy()
        cv2.rectangle(img_contours, (firstX+w1,y1), (secondX, y2+h2), (0, 0, 255), 1)
        area = (secondX - firstX-w1, h1)
        
        
        # Dividir el área en una rejilla en 16 por 10
        rows = 15
        cols = 10
        cell_width = area[0] / cols
        cell_height = area[1] / rows
        firstGrid = (firstX + w1, y1)

        print("Área:", area)
        print("Dimensiones de celda:", cell_width, cell_height)
        print("Primera celda:", firstGrid)

        # Dibujar rejilla
        for i in range(cols + 1):  # Dibujar líneas verticales
            x = int(firstGrid[0] + i * cell_width)  # Calcular posición exacta
            cv2.line(
                img_contours,
                (x, int(firstGrid[1])),
                (x, int(firstGrid[1] + rows * cell_height)),
                (255, 0, 0),
                1,
            )

        for j in range(rows + 1):  # Dibujar líneas horizontales
            y = int(firstGrid[1] + j * cell_height)  # Calcular posición exacta
            cv2.line(
                img_contours,
                (int(firstGrid[0]), y),
                (int(firstGrid[0] + cols * cell_width), y),
                (255, 0, 0),
                1,
            )
    else:
        print("No se encontraron suficientes contornos para calcular el área.")

    # Mostrar la imagen con el área resaltada
    cv2.imshow("Area entre contornos", img_contours)

    cv2.waitKey(1)
    #cv2.destroyAllWindows()


# Hacer match template entre template ladder e img en distintas escalas entre el 50 y 150%
# scales = np.linspace(0.5, 1.5, 30)  # Escalas entre 50% y 150%
# threshold = 0.5  # Umbral de confianza

# best_match = None
# best_val = 0

# # Iterar sobre las escalas
# for scale in scales:
#     # Redimensionar el template ladder
#     resized_ladder = cv2.resize(template_ladder, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
#     h, w = resized_ladder.shape[:2]

#     # Asegurarse de que el template no sea más grande que la imagen principal
#     if h > img.shape[0] or w > img.shape[1]:
#         continue

#     # Realizar el match template
#     result = cv2.matchTemplate(img, resized_ladder, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#     cv2.imshow("res",result)
#     cv2.waitKey(0)
#     # Guardar el mejor match si supera el umbral
#     if max_val > best_val and max_val >= threshold:
#         best_val = max_val
#         best_match = (max_loc, (w, h))
#         #w,h = scale,scale

# # Si se encontró un match válido
# if best_match:
#     top_left = best_match[0]
#     w, h = best_match[1]
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     # guardar posición de ladder y tamaño
#     base_x, base_y = top_left, bottom_right
#     # Dibujar un rectángulo alrededor del match e la imagen editada
#     cv2.rectangle(img_res, top_left, bottom_right, (0, 255, 0), 2)
#     print(f"Match encontrado en posición {top_left} con tamaño ({w}, {h}) y confianza {best_val:.2f}")
#     # Escribir ladder en la imagen editada
#     cv2.putText(img_res, "Ladder", (top_left[0], top_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# else:
#     print("No se encontró un match válido.")

# # Escalar templates a ese tamaño
# templates = {
# 	nombre: cv2.resize(img_t, (w, h), interpolation=cv2.INTER_AREA)
# 	for nombre, img_t in templates_raw.items()
# }


# # Inicializar la posición inicial del ROI
# current_top_left = top_left  # Comienza desde el primer match encontrado
# matches_found = []  # Lista para almacenar los matches encontrados

# while True:
#     # Calcular el ROI en la posición actual
#     roi = img[current_top_left[1]:current_top_left[1]+h, current_top_left[0]:current_top_left[0]+w]

#     # Verificar si el ROI está dentro de los límites de la imagen
#     if roi.shape[0] == 0 or roi.shape[1] == 0:
#         break
    
#     cv2.imshow('r',roi)
#     cv2.waitKey(0)
#     # Verificar si el ROI coincide con el terreno basado en el color promedio
#     roi_mean_color = cv2.mean(roi)[:3]  # Ignorar el canal alfa si existe
#     terrain_mean_color = cv2.mean(templates["terrain"])[:3]
#     color_diff = np.linalg.norm(np.array(roi_mean_color) - np.array(terrain_mean_color))
#     color_threshold = 10  # Ajusta este valor según sea necesario

#     if color_diff < color_threshold:
#         print(f"Terreno encontrado en posición {current_top_left} basado en el color promedio.")
#         matches_found.append({"type": "terrain", "position": current_top_left, "confidence": color_diff})

#         # Dibujar un rectángulo alrededor del terreno en la imagen editada
#         bottom_right = (current_top_left[0] + w, current_top_left[1] + h)
#         cv2.rectangle(img_res, current_top_left, bottom_right, (255, 0, 0), 2)
#         cv2.putText(img_res, "Terrain", (current_top_left[0], current_top_left[1] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#     else:
#         # Verificar match con todos los templates y seleccionar el mejor
#         best_match = None
#         best_val = 0
#         best_name = None

#         for name, template in templates.items():
#             if name == "terrain":  # Saltar el terreno, ya fue verificado
#                 continue


#             result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
#             max_val = np.max(result)

#             # Si el resultado supera el umbral y es el mejor hasta ahora, registrar el match
#             if max_val >= threshold and max_val > best_val:
#                 best_val = max_val
#                 best_match = current_top_left
#                 best_name = name

#         # Si se encontró un mejor match, registrar y dibujar
#         if best_match:
#             print(f"{best_name.capitalize()} encontrado en posición {best_match} con confianza {best_val:.2f}")
#             matches_found.append({"type": best_name, "position": best_match, "confidence": best_val})

#             # Dibujar un rectángulo alrededor del mejor match en la imagen editada
#             bottom_right = (best_match[0] + w, best_match[1] + h)
#             cv2.rectangle(img_res, best_match, bottom_right, (0, 255, 0), 2)
#             cv2.putText(img_res, best_name.capitalize(), (best_match[0], best_match[1] - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#         else:
#             print("No se encontró un match válido en la posición actual.")

#     # Mover hacia abajo para el siguiente cuadro
#     current_top_left = (current_top_left[0], current_top_left[1] + h)

# # Imprimir todos los matches encontrados
# print("Matches encontrados:")
# for match in matches_found:
#     print(match)

# # Mostrar la imagen final con los matches encontrados
# cv2.imshow("Reconocido", img_res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
