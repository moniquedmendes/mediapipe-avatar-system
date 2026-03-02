import cv2
import mediapipe as mp
import pygame
import sys
import numpy as np
import time

# =========================
# INICIALIZA MEDIAPIPE (API NOVA)
# =========================

mp_face_landmarker = mp.tasks.vision.FaceLandmarker
mp_base_options = mp.tasks.BaseOptions
mp_running_mode = mp.tasks.vision.RunningMode

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp_base_options(model_asset_path="face_landmarker.task"),
    running_mode=mp_running_mode.VIDEO,
    num_faces=1
)

face_landmarker = mp_face_landmarker.create_from_options(options)

# =========================
# INICIALIZA PYGAME
# =========================
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Avatar Prototype")

# Carrega sprites
sprite_frente = pygame.image.load("assets/frente.png").convert_alpha()
sprite_esq = pygame.image.load("assets/perfil_esq.png").convert_alpha()
sprite_dir = pygame.image.load("assets/perfil_dir.png").convert_alpha()

sprite_atual = sprite_frente

clock = pygame.time.Clock()

# =========================
# INICIALIZA CÂMERA
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# =========================
# LOOP PRINCIPAL
# =========================
running = True
start_time = time.time()

while running:
    clock.tick(60)

    # Eventos Pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

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

    # =========================
    # PROCESSA LANDMARKS
    # =========================
    if results.face_landmarks:

        landmarks = results.face_landmarks[0]

        # Pontos importantes
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose = landmarks[1]

        # Centro entre olhos
        center_x = (left_eye.x + right_eye.x) / 2

        # Diferença horizontal (simula yaw)
        rotation = nose.x - center_x

        threshold = 0.015  # ajuste de sensibilidade

        if rotation > threshold:
            sprite_atual = sprite_dir
        elif rotation < -threshold:
            sprite_atual = sprite_esq
        else:
            sprite_atual = sprite_frente

        # =========================
        # DESENHA LANDMARKS NA CAMERA
        # =========================
        h, w, _ = frame.shape
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Mostra janela da câmera
    cv2.imshow("Camera Debug", frame)

    # Tecla Q fecha tudo
    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False

    # =========================
    # DESENHA AVATAR (PYGAME)
    # =========================
    screen.fill((30, 30, 30))

    sprite_rect = sprite_atual.get_rect(center=(400, 300))
    screen.blit(sprite_atual, sprite_rect)

    pygame.display.flip()

# =========================
# FINALIZA
# =========================
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()