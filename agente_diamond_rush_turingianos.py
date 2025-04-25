from itertools import count
import cv2
import numpy as np
from collections import deque
import pyautogui

from skimage.metrics import structural_similarity as ssim

# --- CONFIG ---
# Diccionario con objetos y su imagen base
templates_raw = {
	"door": cv2.imread("Diamond-Rush-Bot\FDoor.png", cv2.IMREAD_COLOR ),
	"spike": cv2.imread("Diamond-Rush-Bot\FSpikeBGColor.png", cv2.IMREAD_COLOR),
    "diamondBG": cv2.imread("Diamond-Rush-Bot\FDiamondBGColor.png", cv2.IMREAD_COLOR),
    "terrain" : cv2.imread("Diamond-Rush-Bot\FTerrain.png", cv2.IMREAD_COLOR),
    "key" : cv2.imread("Diamond-Rush-Bot\FKeyBGColor.png", cv2.IMREAD_COLOR),
    "ladder" : cv2.imread("Diamond-Rush-Bot\FLadderBG.png", cv2.IMREAD_COLOR),
}

#template_ladder = cv2.imread("Diamond-Rush-Bot\FLadderBG.png", cv2.IMREAD_COLOR)

# Imagen principal
#img = cv2.imread("fullscreenSS.png", cv2.IMREAD_UNCHANGED)
img = cv2.imread("Diamond-Rush-Bot\SSexample.png", cv2.IMREAD_COLOR)

# imagen editada
img_res = cv2.imread("Diamond-Rush-Bot\SSexample.png", cv2.IMREAD_COLOR)

resXOriginal = 720
resYOriginal = 1280
cellHNumber = 16
cellWNumber = 10
cellHOriginal = 1280/16
cellWOriginal = 720/10
#while True:

#img = pyautogui.screenshot()
#img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
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
            img_res,
            (x, int(firstGrid[1])),
            (x, int(firstGrid[1] + rows * cell_height)),
            (255, 0, 0),
            1,
        )

    for j in range(rows + 1):  # Dibujar líneas horizontales
        y = int(firstGrid[1] + j * cell_height)  # Calcular posición exacta
        cv2.line(
            img_res,
            (int(firstGrid[0]), y),
            (int(firstGrid[0] + cols * cell_width), y),
            (255, 0, 0),
            1,
        )
else:
    print("No se encontraron suficientes contornos para calcular el área.")

# Mostrar la imagen con el área resaltada
#cv2.imshow("Area entre contornos", img_contours)
#cv2.waitKey(1)
#cv2.destroyAllWindows()


# Resize match templates to the size of the cells
for name, template in templates_raw.items():
    templates_raw[name] = cv2.resize(template, (int(cell_width), int(cell_height)), interpolation=cv2.INTER_NEAREST)




# Hallar numero de contornos de spikes
# Convertir el color hexadecimal #160d07 a BGR
target_color = (7, 13, 22)  # Color en formato BGR (Blue, Green, Red)

# Definir un rango de color para el template spike
lower_bound = np.array([0, 0, 0])  # Límite inferior del color (ajusta según sea necesario)
upper_bound = np.array([20, 20, 20])  # Límite superior del color (ajusta según sea necesario)

# Cargar el template spike
template_spike = templates_raw["spike"]

# Verificar si el template se cargó correctamente
if template_spike is None:
    print("El template 'spike' no se cargó correctamente.")
else:
    # Convertir el template a espacio de color HSV
    template_hsv = cv2.cvtColor(template_spike, cv2.COLOR_BGR2HSV)

    # Crear una máscara binaria para el rango de color
    mask = cv2.inRange(template_spike, lower_bound, upper_bound)

    # Encontrar los contornos en la máscara
    contoursSpike, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en el template original
    template_with_contours = template_spike.copy()
    cv2.drawContours(template_with_contours, contours, -1, (0, 255, 0), 2)  # Color verde para los contornos

    # Mostrar la máscara y el template con contornos
    # cv2.imshow("Máscara binaria", mask)
    # cv2.imshow("Contornos en el template spike", template_with_contours)
    # cv2.waitKey(0)
# Contornos totales = 9, con tamaños:


# Traverse the grid and check for matches with templates
for i in range(rows):
    for j in range(cols):
        # Calcular la posición de la celda actual
        cell_x = int(firstGrid[0] + j * cell_width)
        cell_y = int(firstGrid[1] + i * cell_height)
        cell_w = int(cell_width)
        cell_h = int(cell_height)
        cell_roi = img[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w]
        # Verificar el color promedio del ROI
        roi_mean_color = cv2.mean(cell_roi)[:3]
        # Verificar si el ROI coincide con el terreno basado en el color promedio
        terrain_mean_color = cv2.mean(templates_raw["terrain"])[:3]
        color_diff = np.linalg.norm(np.array(roi_mean_color) - np.array(terrain_mean_color))
        color_threshold = 10  # Ajusta este valor según sea necesario

        if color_diff < color_threshold:
            print(f"terreno encontrado en celda ({i}, {j}) con confianza {max_val:.2f}")

            #  Dibujar un rectángulo alrededor del match en la imagen editada
            bottom_right = (cell_x + cell_w, cell_y + cell_h)
            cv2.rectangle(img_res, (cell_x, cell_y), bottom_right, (0, 120, 120), 2)
            cv2.putText(img_res, "Terreno", (cell_x, cell_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        
        # 
        target_color = (7, 13, 22)  # Color en formato BGR (Blue, Green, Red)
        # Definir un rango de color para el template spike
        lower_bound = np.array([0, 0, 0])  # Límite inferior del color (ajusta según sea necesario)
        upper_bound = np.array([20, 20, 20])  # Límite superior del color (ajusta según sea necesario)
        # Convertir el template a espacio de color HSV
        template_hsv = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV)

        # Crear una máscara binaria para el rango de color
        mask = cv2.inRange(cell_roi, lower_bound, upper_bound)

        # Encontrar los contornos en la máscarai
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Verificar si los contornos de spike son iguales a los de la imagen
        if len(contours) == len(contoursSpike):
            match = True
            if match:
                print(f"Spike encontrado en celda ({i}, {j})")
                # Dibujar un rectángulo alrededor del match en la imagen editada
                bottom_right = (cell_x + cell_w, cell_y + cell_h)
                cv2.rectangle(img_res, (cell_x, cell_y), bottom_right, (0, 0, 255), 2)
                cv2.putText(img_res, "Spike", (cell_x, cell_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
        for name, template in templates_raw.items():
            # Realizar el match template
            result = cv2.matchTemplate(cell_roi, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Si el resultado supera el umbral y es el mejor hasta ahora, registrar el match
            if max_val >= 0.9:
                print(f"{name.capitalize()} encontrado en celda ({i}, {j}) con confianza {max_val:.2f}")
                # Dibujar un rectángulo alrededor del match en la imagen editada
                bottom_right = (cell_x + cell_w, cell_y + cell_h)
                cv2.rectangle(img_res, (cell_x, cell_y), bottom_right, (0, 255, 0), 2)
                cv2.putText(img_res, name.capitalize(), (cell_x, cell_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                continue
        # cv2.imshow("cell", cell_roi)
        # cv2.waitKey(0)
        
cv2.imshow("cell", img_res)
cv2.waitKey(0)
        
