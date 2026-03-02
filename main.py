import cv2
import mediapipe as mp
import pygame
import sys
import numpy as np
import time

# =====================================================
# FUNÇÃO EAR
# =====================================================
def calcular_ear(landmarks, indices):
    p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
    p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
    p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
    p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
    p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
    p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])

    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    return (vertical1 + vertical2) / (2.0 * horizontal)

# =====================================================
# MEDIAPIPE
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
# PYGAME
# =====================================================
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# =====================================================
# ASSETS
# =====================================================

# PERFIS
perfil_esq = pygame.image.load("assets/perfil_esq.png").convert_alpha()
perfil_dir = pygame.image.load("assets/perfil_dir.png").convert_alpha()

# ROSTO FRENTE (idle animation)
rosto_base = pygame.image.load("assets/rosto_frente_base.png").convert_alpha()
rosto_1 = pygame.image.load("assets/rosto_frente_1.png").convert_alpha()
rosto_2 = pygame.image.load("assets/rosto_frente_2.png").convert_alpha()

idle_frames = [rosto_base, rosto_1, rosto_2]
idle_index = 0
idle_timer = 0

# OLHOS
olho_aberto_esq = pygame.image.load("assets/olho_aberto_esq.png").convert_alpha()
olho_aberto_dir = pygame.image.load("assets/olho_aberto_dir.png").convert_alpha()
olho_fechado_esq = pygame.image.load("assets/olho_fechado_esq.png").convert_alpha()
olho_fechado_dir = pygame.image.load("assets/olho_fechado_dir.png").convert_alpha()

# SOBRANCELHAS
sob_neutra_esq = pygame.image.load("assets/sobrancelha_neutra_esq.png").convert_alpha()
sob_neutra_dir = pygame.image.load("assets/sobrancelha_neutra_dir.png").convert_alpha()
sob_up_esq = pygame.image.load("assets/sobrancelha_suspresa_esq.png").convert_alpha()
sob_up_dir = pygame.image.load("assets/sobrancelha_suspresa_dir.png").convert_alpha()

# BOCA
boca_aberta_sprite = pygame.image.load("assets/boca_aberta.png").convert_alpha()

# =====================================================
# ESTADOS
# =====================================================
sprite_atual = rosto_base
olho_esq_atual = olho_aberto_esq
olho_dir_atual = olho_aberto_dir
sob_esq_atual = sob_neutra_esq
sob_dir_atual = sob_neutra_dir
boca_aberta = False

# =====================================================
# CAMERA
# =====================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
running = True
start_time = time.time()

# =====================================================
# LOOP
# =====================================================
while running:
    dt = clock.tick(60)
    idle_timer += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp_ms = int((time.time() - start_time) * 1000)
    results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

    if results.face_landmarks:
        landmarks = results.face_landmarks[0]

        # =================================================
        # ROTAÇÃO
        # =================================================
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
            sprite_atual = None  # usando idle animado

        # =================================================
        # PISCAR
        # =================================================
        left_eye_indices = [33,160,158,133,153,144]
        right_eye_indices = [362,385,387,263,373,380]

        ear_left = calcular_ear(landmarks, left_eye_indices)
        ear_right = calcular_ear(landmarks, right_eye_indices)

        blink_threshold = 0.20

        olho_esq_atual = olho_fechado_esq if ear_left < blink_threshold else olho_aberto_esq
        olho_dir_atual = olho_fechado_dir if ear_right < blink_threshold else olho_aberto_dir

        # =================================================
        # SOBRANCELHA
        # =================================================
        sobrancelha_esq = landmarks[70]
        olho_esq_top = landmarks[159]
        sobrancelha_dir = landmarks[300]
        olho_dir_top = landmarks[386]

        dist_esq = abs(sobrancelha_esq.y - olho_esq_top.y)
        dist_dir = abs(sobrancelha_dir.y - olho_dir_top.y)

        sob_threshold = 0.035

        sob_esq_atual = sob_up_esq if dist_esq > sob_threshold else sob_neutra_esq
        sob_dir_atual = sob_up_dir if dist_dir > sob_threshold else sob_neutra_dir

        # =================================================
        # BOCA
        # =================================================
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]

        mouth_open = abs(top_lip.y - bottom_lip.y)
        boca_aberta = mouth_open > 0.05

        # =================================================
        # LANDMARKS NA CAMERA
        # =================================================
        h, w, _ = frame.shape
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # =====================================================
    # ANIMAÇÃO IDLE (ROSTO SEMPRE MEXENDO)
    # =====================================================
    if sprite_atual is None:
        if idle_timer > 150:
            idle_index = (idle_index + 1) % len(idle_frames)
            idle_timer = 0

        sprite_atual = idle_frames[idle_index]

    # =====================================================
    # RENDER
    # =====================================================
    screen.fill((30, 30, 30))
    center = rosto_base.get_rect(center=(400, 300))

    screen.blit(sprite_atual, center)

    if sprite_atual not in [perfil_esq, perfil_dir]:
        screen.blit(sob_esq_atual, center)
        screen.blit(sob_dir_atual, center)
        screen.blit(olho_esq_atual, center)
        screen.blit(olho_dir_atual, center)

        if boca_aberta:
            screen.blit(boca_aberta_sprite, center)

    pygame.display.flip()

    cv2.imshow("Camera Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False

cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()