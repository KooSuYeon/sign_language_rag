import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import numpy as np
import time

# MediaPipe 손 추적 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 학습된 모델 로드 (모델 입력: (MAX_FRAMES, 126))
model = load_model("sign_language_lstm_model.h5")

# 비디오 캡처
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 설정값
FPS = 30  # 초당 프레임 수
MAX_FRAMES = 90  # 3초(30fps 기준)
NUM_FEATURES = 126  # 손 랜드마크 특징 수

# 클래스 이름 목록
titles = [
    '반복,거듭,수시로,자꾸,자주,잦다,여러 번,연거푸', 
    '똑같다,같다,동일하다', 
    '걷어차다', 
    '느끼다,느낌,뉘앙스', 
    '문화재', 
    '방심,부주의', 
    '가난,곤궁,궁핍,빈곤', 
    '가치', 
    '장애인 기능 경진 대회', 
    '의사소통'
]

def extract_keypoints_from_webcam(frame, hands):
    """웹캠 프레임에서 손 랜드마크 좌표를 추출 (양손 데이터 처리 포함)"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    keypoints_list = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:2]:  # 최대 2개 손 인식
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            keypoints_list.append(keypoints.flatten())  # (63,)

    # 손이 한 개만 감지되었을 경우, 더미 데이터 추가
    while len(keypoints_list) < 2:
        keypoints_list.append(np.zeros(63, dtype=np.float32))  # (63,)

    # 두 손 데이터를 하나로 합치기 → (126,)
    if keypoints_list:
        return np.concatenate(keypoints_list, axis=0)  # (126,)
    else:
        return None  # 손이 없을 경우 None 반환

def prepare_input(landmark_array):
    """ 모델 입력 크기 (90, 126)로 맞추기 위해 패딩 적용 """
    padded = np.zeros((MAX_FRAMES, NUM_FEATURES), dtype=np.float32)
    seq_len = min(len(landmark_array), MAX_FRAMES)

    padded[:seq_len, :] = landmark_array[:seq_len, :]
    return np.expand_dims(padded, axis=0)  # (1, 90, 126) 형태로 변경 (배치 차원 추가)


# 예측 결과 및 타이머 변수
predicted_word = None
last_prediction_time = time.time()
frame_buffer = []  # 90 프레임을 저장하는 버퍼

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 손 랜드마크 추출
        landmark_array = extract_keypoints_from_webcam(frame, hands)

        if landmark_array is not None:
            # 90 프레임 유지
            if len(frame_buffer) >= MAX_FRAMES:
                frame_buffer.pop(0)
            frame_buffer.append(landmark_array)
            
            # 3초마다 예측 수행
            current_time = time.time()
            if len(frame_buffer) == MAX_FRAMES and (current_time - last_prediction_time) >= 3:  # 3초 이상 차이
                X_input = prepare_input(np.array(frame_buffer))
                prediction = model.predict(X_input)

                predicted_label = np.argmax(prediction, axis=1)[0]
                predicted_word = titles[predicted_label]
                print("예측된 수화 단어:", predicted_word)

                # 마지막 예측 시간 업데이트
                last_prediction_time = current_time
        else:
            # 손이 감지되지 않으면 프레임 버퍼 초기화 & 예측 단어 삭제
            frame_buffer.clear()
            predicted_word = None

        # 손 랜드마크 시각화
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 예측된 단어 화면에 표시
        if predicted_word:
            cv2.putText(frame, f"Predicted: {predicted_word}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 결과 화면 출력
        cv2.imshow('Sign Language Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
