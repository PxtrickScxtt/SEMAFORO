import cv2
import numpy as np

# Crear objeto para la detección de movimiento
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30, detectShadows=True)


# Función para procesar un video y mostrar los resultados en tiempo real
def process_video(video):
    # Capturar video desde un archivo
    cap = cv2.VideoCapture(video)
    total_frames = 0
    total_drawing_time = 0
    motion_pixel_counts = []
    previous_positions = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / frame_rate

    while cap.isOpened():
        # Capturar frame del video
        ret, frame = cap.read()

        # Verificar que el frame se haya capturado correctamente
        if not ret:
            break

        # Redimensionar el frame a una resolución más alta para capturar más detalles
        frame = cv2.resize(frame, (1280, 720))

        # Aplicar un filtro gaussiano para reducir el ruido
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # Ajustar la región de interés (ROI)
        roi = frame[10: 800, 530: 800]  # Ajustar los valores según sea necesario

        # 1. Detección de objetos en el ROI
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Crear una lista para almacenar las bounding boxes de los contornos combinados
        bounding_boxes = []

        for contour in contours:
            if cv2.contourArea(contour) < 250:  # Ignorar áreas pequeñas
                continue
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, x + w, y + h))

        # Combinar las bounding boxes que se superponen o están muy cerca
        def combine_boxes(boxes, threshold=10):
            combined = []
            while boxes:
                a = boxes.pop(0)
                new_box = a
                overlapping = [b for b in boxes if (
                            a[0] - threshold < b[2] and a[2] + threshold > b[0] and a[1] - threshold < b[3] and a[
                        3] + threshold > b[1])]
                while overlapping:
                    b = overlapping.pop(0)
                    boxes.remove(b)
                    new_box = (
                    min(new_box[0], b[0]), min(new_box[1], b[1]), max(new_box[2], b[2]), max(new_box[3], b[3]))
                    overlapping = [b for b in boxes if (
                                new_box[0] - threshold < b[2] and new_box[2] + threshold > b[0] and new_box[
                            1] - threshold < b[3] and new_box[3] + threshold > b[1])]
                combined.append(new_box)
            return combined

        combined_boxes = combine_boxes(bounding_boxes)

        motion_pixels = 0
        drawing_time = 0
        current_positions = []
        velocities = []

        if combined_boxes:
            drawing_start_time = cv2.getTickCount()
            for box in combined_boxes:
                x, y, x2, y2 = box
                w, h = x2 - x, y2 - y
                center_x, center_y = x + w // 2, y + h // 2
                radius = int(max(w, h) // 3)
                cv2.circle(roi, (center_x, center_y), radius, (255, 255, 0), -1)
                motion_pixels += w * h
                current_positions.append((center_x, center_y))

            if previous_positions:
                for current, previous in zip(current_positions, previous_positions):
                    dx = current[0] - previous[0]
                    dy = current[1] - previous[1]
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    velocity = distance / frame_time
                    velocities.append(velocity)
                    cv2.putText(roi, f'{velocity:.2f} px/s', (current[0], current[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

            previous_positions = current_positions

            drawing_end_time = cv2.getTickCount()
            drawing_time = (
                                       drawing_end_time - drawing_start_time) / cv2.getTickFrequency() * 1000  # Convertir a milisegundos

        # Acumular el tiempo de dibujo y el número de píxeles con movimiento
        total_frames += 1
        total_drawing_time += drawing_time
        motion_pixel_counts.append(motion_pixels)

        # Mostrar la máscara de movimiento en la región de interés
        cv2.imshow('Motion Mask in ROI', mask)

        # Mostrar el frame procesado en la ventana
        cv2.imshow('Processed Frame', frame)

        # Imprimir en consola la información de cada frame
        print(f'Frame Info: Pixeles ocupados: {motion_pixels}, Tiempo de dibujo: {drawing_time:.4f} ms')
        if velocities:
            print(f'Velocidades de contornos: {[f"{v:.2f} px/s" for v in velocities]}')

        # Esperar 1 milisegundo (se puede ajustar si es necesario)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

    # Calcular y mostrar estadísticas finales
    if total_frames > 0:
        avg_drawing_time = total_drawing_time / total_frames
        avg_motion_pixels = sum(motion_pixel_counts) / total_frames
        print(f'\nEstadísticas finales:')
        print(f'Total de frames procesados: {total_frames}')
        print(f'Tiempo de dibujo promedio por frame: {avg_drawing_time:.4f} ms')
        print(f'Promedio de píxeles con movimiento por frame: {avg_motion_pixels:.2f}')


# Ruta al archivo de video
video_path = 'a.mp4'

# Llamar a la función para procesar el video
process_video(video_path)
