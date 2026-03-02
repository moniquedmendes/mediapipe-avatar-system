import cv2
import mediapipe as mp
import pygame
import sys
import numpy as np
import time

# =====================================================
# FUNÇÃO PARA CALCULAR EAR (Eye Aspect Ratio)
# =====================================================
def calcular_ear(landmarks, indices):
    """
    Calcula o Eye Aspect Ratio.
    Quando o olho fecha, o valor diminui.
    """

    p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
    p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
    p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
    p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
    p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
    p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])

    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


# =====================================================
# INICIALIZA MEDIAPIPE (API NOVA)
# =====================================================
mp_face_landmarker = mp.tasks.vision.FaceLandmarker
mp_base_options = mp.tasks.BaseOptions
mp_running_mode = mp.tasks.vision.RunningMode

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp_base_options(model_asset_path="face_landmarker.task"),
    running_mode=mp_running_mode.VIDEO,
    num_faces=1
)

face_landmarker = mp_face_landmarker.create_from_options(options)

# =====================================================
# INICIALIZA PYGAME
# =====================================================
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Avatar Prototype")

clock = pygame.time.Clock()

# =====================================================
# CARREGA ASSETS
# =====================================================

# Corpo / rosto base
rosto_frente = pygame.image.load("assets/rosto_frente.png").convert_alpha()
perfil_esq = pygame.image.load("assets/perfil_esq.png").convert_alpha()
perfil_dir = pygame.image.load("assets/perfil_dir.png").convert_alpha()

# Olhos
olho_aberto_esq = pygame.image.load("assets/olho_aberto_esq.png").convert_alpha()
olho_aberto_dir = pygame.image.load("assets/olho_aberto_dir.png").convert_alpha()

olho_fechado_esq = pygame.image.load("assets/olho_fechado_esq.png").convert_alpha()
olho_fechado_dir = pygame.image.load("assets/olho_fechado_dir.png").convert_alpha()

# Estado atual
sprite_atual = rosto_frente
olho_esq_atual = olho_aberto_esq
olho_dir_atual = olho_aberto_dir

# =====================================================
# INICIALIZA CÂMERA
# =====================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# =====================================================
# LOOP PRINCIPAL
# =====================================================
running = True
start_time = time.time()

while running:
    clock.tick(60)

    # Eventos Pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Captura frame da câmera
    ret, frame = cap.read()
    if not ret:
        print("Camera falhou")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp_ms = int((time.time() - start_time) * 1000)

    results = face_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # =====================================================
    # PROCESSAMENTO DE LANDMARKS
    # =====================================================
    if results.face_landmarks:

        landmarks = results.face_landmarks[0]

        # --------- ROTACAO DA CABEÇA ---------
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose = landmarks[1]

        center_x = (left_eye.x + right_eye.x) / 2
        rotation = nose.x - center_x

        threshold_rot = 0.015

        if rotation > threshold_rot:
            sprite_atual = perfil_dir
        elif rotation < -threshold_rot:
            sprite_atual = perfil_esq
        else:
            sprite_atual = rosto_frente

        # --------- DETECCAO DE PISCAR (SOMENTE FRENTE) ---------
        if sprite_atual == rosto_frente:

            left_eye_indices = [33,160,158,133,153,144]
            right_eye_indices = [362,385,387,263,373,380]

            ear_left = calcular_ear(landmarks, left_eye_indices)
            ear_right = calcular_ear(landmarks, right_eye_indices)

            ear = (ear_left + ear_right) / 2

            blink_threshold = 0.20

            if ear < blink_threshold:
                olho_esq_atual = olho_fechado_esq
                olho_dir_atual = olho_fechado_dir
            else:
                olho_esq_atual = olho_aberto_esq
                olho_dir_atual = olho_aberto_dir

        # --------- DESENHA LANDMARKS NA CAMERA ---------
        h, w, _ = frame.shape
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Mostra janela da câmera
    cv2.imshow("Camera Debug", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False

    # =====================================================
    # RENDERIZAÇÃO NO PYGAME
    # =====================================================
    screen.fill((30, 30, 30))

    center = rosto_frente.get_rect(center=(400, 300))

    if sprite_atual == rosto_frente:
        screen.blit(rosto_frente, center)
        screen.blit(olho_esq_atual, center)
        screen.blit(olho_dir_atual, center)
    else:
        screen.blit(sprite_atual, center)

    pygame.display.flip()

# =====================================================
# FINALIZAÇÃO
# =====================================================
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()