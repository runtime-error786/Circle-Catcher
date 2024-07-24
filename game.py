import cv2
import numpy as np
import random
import time
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def generate_circle(frame_height, frame_width):
    x = random.randint(50, frame_width - 50)
    y = random.randint(50, frame_height - 50)
    radius = 30
    return (x, y, radius)

def is_hand_near_circle(hand_landmarks, circle_center, radius, threshold=50):
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * frame_width), int(lm.y * frame_height)
        distance = np.sqrt((x - circle_center[0])**2 + (y - circle_center[1])**2)
        if distance < (radius + threshold):
            return True
    return False
