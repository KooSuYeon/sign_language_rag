import os
import time
import pickle
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, BatchNormalization, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.neighbors import KNeighborsClassifier
import ast
import csv

# AIhub에서 가져온 한국수어 데이터 셋 (한국수어와 해당하는 수어동영상 url이 존재함)
raw_data = pd.read_csv("dataset.csv")
# raw_data.head()
"""
1. 학습을 위한 매핑 : 데이터 셋으로부터 한국 수어를 딕셔너리로 매핑해줌

{0: '느끼다,느낌,뉘앙스', 1: '장애인 기능 경진 대회', 2: '똑같다,같다,동일하다', 3: '가난,곤궁,궁핍,빈곤', 4: '반복,거듭,수시로,자꾸,자주,잦다,여러 번,연거푸', 5: '가치', 6: '의사소통', 7: '걷어차다', 8: '문화재', 9: '방심,부주의'}
"""
gesture = {i: title for i, title in enumerate(raw_data["Title"][:10])}
# print(gesture)

"""
2. Mediapipe Hands Model 설정

- 인식할수 있는 손의 갯수 선언 : 양손
- 사용할 손의 유틸리티 선언 : hands 유틸리티
- 회소 탐지 신뢰도, 최소 추적 신뢰도 (기본값 사용) : 0.5

"""
MAX_NUM_HANDS = 2
mp_hands=mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands=mp_hands.Hands(
    max_num_hands=MAX_NUM_HANDS, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5 
)


"""
npy 형태로 저장되도록
"""

# 10개의 테스트 데이터들
actions = raw_data["Title"][:10].tolist() 

# # 미정
# seq_length = 30

# # 저장할 디렉토리 저장
# os.makedirs('dataset', exist_ok=True)

# # URL에서 비디오 데이터 추출 후 시각적으로 확인
# def extract_from_url():
#     urls = raw_data['SubDescription'][:10]
    
#     for idx, url in enumerate(urls):
#         video = cv2.VideoCapture(url)
#         if not video.isOpened():
#             print(f"URL에서 비디오를 열 수 없습니다: {url}")
#             continue
        
#         print(f"비디오 시작: {url}")
#         data = []
#         frame_count = 0
        
#         while video.isOpened():
#             ret, img = video.read()
#             if not ret:
#                 break
            
#             img = cv2.flip(img, 1)  # 영상 좌우 반전
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             result = hands.process(img_rgb)
            
#             if result.multi_hand_landmarks:
#                 for res in result.multi_hand_landmarks:
#                     joint = np.zeros((21, 3))
#                     for j, lm in enumerate(res.landmark):
#                         joint[j] = [lm.x, lm.y, lm.z]
                    
#                     v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
#                     v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
#                     v = v2 - v1  
#                     v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  
                    
#                     angle = np.arccos(np.einsum('nt,nt->n',
#                                                 v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
#                                                 v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
#                     angle = np.degrees(angle)  
                    
#                     angle_label = np.array([angle], dtype=np.float32)
#                     angle_label = np.append(angle_label, idx)
                    
#                     d = np.concatenate([joint.flatten(), angle_label])
#                     data.append(d)
                    
#                     # 화면에 손 관절 및 각도 표시
#                     mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                    
#                     # 각도를 텍스트로 표시
#                     for i, (x, y, z) in enumerate(joint):
#                         text_pos = (int(x * img.shape[1]), int(y * img.shape[0]))
#                         cv2.putText(img, str(int(angle[i] if i < len(angle) else 0)), text_pos,
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            
#             frame_count += 1
#             cv2.imshow('Hand Tracking', img)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        
#         data = np.array(data)
#         print(actions[idx], data.shape)
#         np.save(os.path.join('dataset', f'raw_{actions[idx]}'), data)
        
#         full_seq_data = []
#         for seq in range(len(data) - seq_length):
#             full_seq_data.append(data[seq:seq + seq_length])
#         full_seq_data = np.array(full_seq_data)
#         print(actions[idx], full_seq_data.shape)
#         np.save(os.path.join('dataset', f'seq_{actions[idx]}'), full_seq_data)
        
#         video.release()
#     cv2.destroyAllWindows()

# # 실행
# extract_from_url()

"""
dataset에 있는 모든 npy 가져오도록
"""

# dataset 폴더 내의 모든 seq 파일을 actions 순으로 가져오기
data_list = []

for action in actions:
    # seq 파일 목록 불러오기 (action에 해당하는 파일만)
    seq_files = [f for f in os.listdir('dataset') if f.startswith(f'seq_{action}')]
    
    for seq_file in seq_files:
        # .npy 파일 로드
        seq_data = np.load(os.path.join('dataset', seq_file))
        data_list.append(seq_data)

# 모든 데이터를 결합
data = np.concatenate(data_list, axis=0)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]

# print(x_data)
# print(x_data.shape)
# print(labels.shape)

y_data = to_categorical(labels, num_classes=len(actions))
y_data.shape

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]), # node 개수 64개
    Dense(32, activation='relu'), # node 개수 32개
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) # loss='categorical_crossentropy -> 3개의 action 중 어떤 건지 모델에게 추론하게 함
model.summary()

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)

# 학습이 완료되면 그래프를 그리도록
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()

model.save('models/model.h5') 


'''
리소스 문제로 멀티쓰레드를 통해 병렬 작업 진행
위함수에서 npy 파일 형태로 저장하게 함
'''
# def working_threads():
#     executor = ThreadPoolExecutor(max_workers=8)
#     futures = []

#     # 비디오 URL을 직접 처리하도록 수정
#     video_urls = raw_data["SubDescription"].to_list()[:10]

#     for video_url in tqdm(video_urls, colour='green'):
#         future = executor.submit(extract_hand_keypoints_from_url, video_url, output_path)
#         futures.append(future)

#     print("file listup finish.. threads start..")

#     for future in tqdm(futures, colour='blue'):
#         future.result()

#     executor.shutdown()

# working_threads()




"""
아래부터는 이미지 > 동영상으로 확장하기 위해 사용한 메서드 (실제 사용 X)
예시로 가위, 바위, 보 학습

"""
# video = cv2.VideoCapture(1)
# gestures = []
# labels = []
# # print("손 모양을 만든 후, 0~9 또는 s 키를 눌러 데이터를 저장하세요. (q 키를 누르면 종료)")

# while video.isOpened():
#     ret, img = video.read()
#     img = cv2.flip(img, 1)

#     # BGR → RGB로 변경 (파이썬이 인식할 수 있도록)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # 손 탐지하기
#     result = hands.process(img_rgb)

#     if not ret:
#         break

#     # 찾은 손 표시하기
#     if result.multi_hand_landmarks:
#         for res in result.multi_hand_landmarks:
#             joint = np.zeros((21, 3))  # 21개 관절, xyz값 저장할 배열 생성
#             for j, lm in enumerate(res.landmark):
#                 joint[j] = [lm.x, lm.y, lm.z]

#             # 연결할 관절 번호 가져오기
#             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
#             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
#             v = v2 - v1  # 뼈의 값(x, y, z 좌표값 → 벡터값)
#             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # 유클리디안 길이로 변환

#             # 뼈의 값으로 뼈 사이의 각도 구하기, 변화값이 큰 15개
#             angle = np.arccos(np.einsum('nt,nt->n',
#                                         v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
#                                         v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
#             # radian 각도를 degree 각도로 변경
#             angle = np.degrees(angle)

#             X = angle.astype(np.float32)
#             mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
#             # 화면에 출력
#             cv2.imshow('hand', img)
            
#             key = cv2.waitKey(1) & 0xFF
            
#             # 0~9 또는 's' 키로 저장
#             if key in [ord(str(i)) for i in range(10)] or key == ord('s'):
#                 label = chr(key)
#                 gestures.append(X)
#                 labels.append(label)
#                 print(f"저장됨: {label}")
            
#             # 'q' 키로 종료
#             elif key == ord('q'):
#                 break
    
#     k = cv2.waitKey(30)
#     if k == ord('q'):
#         break

#     cv2.imshow('hand', img)

# # 카메라 종료
# video.release()
# cv2.destroyAllWindows()

# if gestures:
#     gestures_array = np.vstack(gestures)  # 리스트를 2D numpy 배열로 변환
#     df = pd.DataFrame(gestures_array)  # numpy 배열을 DataFrame으로 변환
#     df['label'] = labels  # 레이블 추가
    
#     # CSV 저장 (쉼표 없이 정렬된 형태로)
#     df.to_csv('./data/gesture_train.csv', index=False, header=False)
    
#     print("데이터 저장 완료: gesture_train.csv")
# else:
#     print("저장된 데이터가 없습니다.")

"""
웹캠에서 불러와서 학습
"""
# gesture = {
#     0:'Scissor', 1:'Rock', 2:'Paper'
# }

# # # CSV 파일을 읽기
# file = pd.read_csv('./data/gesture_train.csv', header=None)
# # X = np.array(file.iloc[:, :-1], dtype=np.float32)
# X = file.iloc[:, :-1].astype(dtype=np.float32)
# y = file.iloc[:, -1].astype(dtype=np.float32)

# # print(X)
# # print(y)
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X, y)

# video = cv2.VideoCapture(1)

# while video.isOpened():
#     ret, img = video.read()
#     img = cv2.flip(img, 1)

#     # BGR → RGB로 변경 (파이썬이 인식할 수 있도록)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # 손 탐지하기
#     result = hands.process(img_rgb)

#     if not ret:
#         break

#     # 찾은 손 표시하기
#     if result.multi_hand_landmarks:
#         for res in result.multi_hand_landmarks:
#             joint = np.zeros((21, 3))  # 21개 관절, xyz값 저장할 배열 생성
#             for j, lm in enumerate(res.landmark):
#                 joint[j] = [lm.x, lm.y, lm.z]

#             # 연결할 관절 번호 가져오기
#             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
#             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
#             v = v2 - v1  # 뼈의 값(x, y, z 좌표값 → 벡터값)
#             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # 유클리디안 길이로 변환

#             # 뼈의 값으로 뼈 사이의 각도 구하기, 변화값이 큰 15개
#             angle = np.arccos(np.einsum('nt,nt->n',
#                                         v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
#                                         v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
#             # radian 각도를 degree 각도로 변경
#             angle = np.degrees(angle)

#             # 구한 각도를 knn 모델에 예측시키기
#             X_pred = np.array([angle], dtype = np.float32)
#             results = knn.predict(X_pred)

#             print(results)
#             idx = int(results)
#             # print(idx)

#             # 인식된 제스쳐 표현하기
#             img_x = img.shape[1]
#             img_y = img.shape[0]
#             hand_x = res.landmark[0].x
#             hand_y = res.landmark[0].y

#             cv2.putText(img, text = gesture[idx].upper(),
#                        org = (int(hand_x * img_x), int(hand_y * img_y)+20),
#                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2
#                        )
            
#             mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

#     # 키 입력 대기
#     k = cv2.waitKey(30)
#     if k == 49:  # '1' 키를 누르면 종료
#         break

#     # 손 이미지 보여주기
#     cv2.imshow('hand', img)

# # 비디오 캡처 객체 해제
# video.release()
# cv2.destroyAllWindows()