import cv2
import numpy as np

# Crear objeto para la detección de movimiento
detector_movimiento = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30, detectShadows=True)

# Función para combinar las bounding boxes que se superponen o están muy cerca
def combinar_cajas(cajas, umbral=10):
    combinadas = []
    while cajas:
        a = cajas.pop(0)
        nueva_caja = a
        superpuestas = [b for b in cajas if (a[0] - umbral < b[2] and a[2] + umbral > b[0] and a[1] - umbral < b[3] and a[3] + umbral > b[1])]
        while superpuestas:
            b = superpuestas.pop(0)
            cajas.remove(b)
            nueva_caja = (min(nueva_caja[0], b[0]), min(nueva_caja[1], b[1]), max(nueva_caja[2], b[2]), max(nueva_caja[3], b[3]))
            superpuestas = [b for b in cajas if (nueva_caja[0] - umbral < b[2] and nueva_caja[2] + umbral > b[0] and nueva_caja[1] - umbral < b[3] and nueva_caja[3] + umbral > b[1])]
        combinadas.append(nueva_caja)
    return combinadas

# Función para procesar un video y mostrar los resultados en tiempo real
def procesar_video(video):
    cap = cv2.VideoCapture(video)
    total_frames, tiempo_dibujo_total = 0, 0
    conteo_pixeles_movimiento, posiciones_previas = [], []
    tasa_frames = cap.get(cv2.CAP_PROP_FPS)
    tiempo_frame = 1.0 / tasa_frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        roi = cv2.GaussianBlur(frame[10:800, 530:800], (5, 5), 0)
        mascara = detector_movimiento.apply(roi)
        _, mascara = cv2.threshold(mascara, 120, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cajas_delimitadoras = [(x, y, x + w, y + h) for contorno in contornos if cv2.contourArea(contorno) >= 250 for x, y, w, h in [cv2.boundingRect(contorno)]]
        cajas_combinadas = combinar_cajas(cajas_delimitadoras)

        pixeles_movimiento, posiciones_actuales, velocidades = 0, [], []
        tiempo_inicio_dibujo = cv2.getTickCount()

        for caja in cajas_combinadas:
            x, y, x2, y2 = caja
            centro_x, centro_y = x + (x2 - x) // 2, y + (y2 - y) // 2
            radio = int(max(x2 - x, y2 - y) // 3)
            cv2.circle(roi, (centro_x, centro_y), radio, (255, 255, 0), -1)
            cv2.rectangle(frame, (x + 530, y + 10), (x2 + 530, y2 + 10), (0, 0, 255), 2)  # Dibujar rectángulo en el frame original
            pixeles_movimiento += (x2 - x) * (y2 - y)
            posiciones_actuales.append((centro_x, centro_y))

        if posiciones_previas:
            for actual, previa in zip(posiciones_actuales, posiciones_previas):
                dx, dy = actual[0] - previa[0], actual[1] - previa[1]
                velocidad = np.sqrt(dx ** 2 + dy ** 2) / tiempo_frame
                velocidades.append(velocidad)
                cv2.putText(roi, f'{velocidad:.2f} px/s', (actual[0], actual[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        posiciones_previas = posiciones_actuales
        tiempo_dibujo = (cv2.getTickCount() - tiempo_inicio_dibujo) / cv2.getTickFrequency() * 1000
        total_frames += 1
        tiempo_dibujo_total += tiempo_dibujo
        conteo_pixeles_movimiento.append(pixeles_movimiento)

        cv2.imshow('Mascara de Movimiento en ROI', mascara)
        cv2.imshow('Frame Procesado', frame)
        print(f'Información del Frame: Pixeles ocupados: {pixeles_movimiento}, Tiempo de dibujo: {tiempo_dibujo:.4f} ms')
        if velocidades:
            print(f'Velocidades de contornos: {[f"{v:.2f} px/s" for v in velocidades]}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if total_frames > 0:
        tiempo_dibujo_promedio = tiempo_dibujo_total / total_frames
        promedio_pixeles_movimiento = sum(conteo_pixeles_movimiento) / total_frames
        print(f'\nEstadísticas finales:')
        print(f'Total de frames procesados: {total_frames}')
        print(f'Tiempo de dibujo promedio por frame: {tiempo_dibujo_promedio:.4f} ms')
        print(f'Promedio de píxeles con movimiento por frame: {promedio_pixeles_movimiento:.2f}')

# Ruta al archivo de video
ruta_video = 'a.mp4'

# Llamar a la función para procesar el video
procesar_video(ruta_video)
