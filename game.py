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

