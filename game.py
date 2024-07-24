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

def play_game():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    score = 0
    start_time = time.time()
    game_duration = 30  

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        return
    
    global frame_height, frame_width
    frame_height, frame_width, _ = frame.shape
    circle_center = generate_circle(frame_height, frame_width)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if is_hand_near_circle(hand_landmarks, circle_center, circle_center[2]):
                    score += 1
                    circle_center = generate_circle(frame_height, frame_width)
        
        cv2.circle(frame, (circle_center[0], circle_center[1]), circle_center[2], (0, 255, 0), -1)
        
        elapsed_time = int(time.time() - start_time)
        remaining_time = max(0, game_duration - elapsed_time)
        cv2.putText(frame, f'Score: {score}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.putText(frame, f'Time: {remaining_time}s', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        
        cv2.imshow('Hand Game', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or remaining_time <= 0:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f'Final Score: {score}')
    while True:
        final_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(final_frame, f'Final Score: {score}', (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        cv2.putText(final_frame, 'Press "r" to Restart or "q" to Quit', (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Hand Game', final_frame)
        
        key = cv2.waitKey(0)
        if key & 0xFF == ord('r'):
            play_game()
            break
        elif key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    play_game()
