import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import numpy as np

# MediaPipe 손 추적 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 학습된 모델 로드 (모델 입력: (407, 126, 1))
model = load_model("sign_language_model.h5")

# 비디오 캡처 (웹캠 번호 조정)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 최대 프레임 수 (훈련 시 사용한 값)
MAX_FRAMES = 407

# 클래스 이름 목록 (원핫 인코딩된 레이블에 해당하는 클래스 이름)
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

# 초당 프레임 수 (FPS)
FPS = 30  # 예시로 30fps로 가정
# 3초 동안의 동작을 한 클래스에서 처리할 수 있도록 MAX_FRAMES를 3초에 맞추기
max_frames_per_class = FPS * 3  # 3초 동안의 동작

def extract_keypoints_from_webcam(frame, hands, holistic=False):
    """웹캠 프레임에서 손 랜드마크 좌표를 추출하여 (21,3) 배열 또는 없으면 빈 배열 반환"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if holistic:
        results = hands.process(frame_rgb)
    else:
        results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # 첫 번째 손만 사용
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        return keypoints  # (21, 3) 형태의 키포인트
    else:
        return None


def prepare_input(landmark_array):
    """
    landmark_array: (N, 3) 배열. 만약 N < 42, 42개(왼손21 + 오른손21)를 채우도록 0-padding.
    평탄화한 후, 현재 프레임의 데이터를 max_frames_per_class 만큼 복제하여 (max_frames_per_class, 126) 배열을 생성하고
    마지막에 채널 차원을 추가하여 (1, max_frames_per_class, 126, 1)로 반환.
    """
    # 모델 학습 시, 아마도 양손 데이터를 사용하여 42개 좌표를 기대했을 가능성이 있으므로,
    # 여기서는 한 손만 있다면 나머지 21개를 0으로 채워 42개로 만듭니다.
    N = landmark_array.shape[0]
    if N < 42:
        padded = np.zeros((42, 3), dtype=np.float32)
        padded[:N, :] = landmark_array
    else:
        padded = landmark_array[:42, :]  # 만약 42개 이상이면 처음 42개 사용
    
    # 평탄화: (42, 3) -> (126,)
    flattened = padded.flatten()  # shape (126,)
    
    # 현재 프레임의 데이터를 max_frames_per_class만큼 복제하여 시퀀스 생성: (max_frames_per_class, 126)
    sequence = np.tile(flattened, (max_frames_per_class, 1))
    
    # 배치 차원과 채널 차원 추가: (1, max_frames_per_class, 126)
    X_input = np.expand_dims(sequence, axis=0)        # (1, max_frames_per_class, 126)
    X_input = np.expand_dims(X_input, axis=-1)          # (1, max_frames_per_class, 126, 1)
    return X_input

# 3초 동안의 데이터를 저장할 버퍼
frame_buffer = []
predicted_labels = []

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 손 랜드마크 추출 (첫 번째 손 사용)
        landmark_array = extract_keypoints_from_webcam(frame, hands)
        
        # 예측 수행: 손이 감지되었을 때만 진행
        if landmark_array is not None:
            # 3초 동안의 동작을 저장
            frame_buffer.append(landmark_array)
            
            # 버퍼에 충분한 프레임이 모였을 때 예측
            if len(frame_buffer) >= max_frames_per_class:
                # 3초 동안의 데이터 준비
                X_input = prepare_input(np.array(frame_buffer))
                
                # 모델 예측
                prediction = model.predict(X_input)
                
                # 예측된 클래스를 여러 개 얻을 수 있도록 변경
                predicted_label = np.argmax(prediction, axis=1)
                
                # 예측된 단어 찾기
                predicted_words = [titles[label] for label in predicted_label]
                
                # 예측된 단어를 화면에 표시
                cv2.putText(frame, f"Predicted Class: {', '.join(predicted_words)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # 모든 예측된 단어를 모아서 출력
                predicted_labels.append(predicted_words)

                # 예측된 단어 출력 (연속적인 예측을 위해)
                print("예측된 단어들:", predicted_labels)

                # 버퍼 초기화 (새로운 예측을 위해)
                frame_buffer = []

        # 손 랜드마크 시각화 (모든 손에 대해)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:  # None 체크
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 결과 화면 출력
        cv2.imshow('Real-time Sign Language Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
