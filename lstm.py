import os
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

raw_data = pd.read_csv("dataset.csv")
raw_data.head()
output_path = './keypoint'
mp4_list = []

'''
영상 데이터 셋으로부터 
유의미한 왼손, 오른손의
x,y,z 축 정보를 받아옴
'''
# def extract_keypoints_from_url(video_url, dest_path):
#     # Mediapipe 솔루션 로드
#     mp_holistic = mp.solutions.holistic
#     cap = cv2.VideoCapture(video_url)  # OpenCV로 비디오 스트림 열기

#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         # 랜드마크 데이터를 저장할 리스트
#         holistic_keypoints_list = []

#         while True:
#             opened, image = cap.read()
#             if not opened:
#                 break

#             # Mediapipe는 RGB 이미지를 필요로 하므로 변환
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             # Mediapipe를 사용해 포즈, 얼굴, 손 랜드마크 추출
#             results = holistic.process(image_rgb)

#             frame_keypoints = {}

#             # # 포즈 랜드마크 추출
#             # if results.pose_landmarks:
#             #     pose_keypoints = []
#             #     for landmark in results.pose_landmarks.landmark:
#             #         pose_keypoints.append([landmark.x, landmark.y, landmark.z])
#             #     frame_keypoints['pose'] = pose_keypoints

#             # # 얼굴 랜드마크 추출
#             # if results.face_landmarks:
#             #     face_keypoints = []
#             #     for landmark in results.face_landmarks.landmark:
#             #         face_keypoints.append([landmark.x, landmark.y, landmark.z])
#             #     frame_keypoints['face'] = face_keypoints

#             # 왼손 랜드마크 추출
#             if results.left_hand_landmarks:
#                 left_hand_keypoints = []
#                 for landmark in results.left_hand_landmarks.landmark:
#                     left_hand_keypoints.append([landmark.x, landmark.y, landmark.z])
#                 frame_keypoints['left_hand'] = left_hand_keypoints

#             # 오른손 랜드마크 추출
#             if results.right_hand_landmarks:
#                 right_hand_keypoints = []
#                 for landmark in results.right_hand_landmarks.landmark:
#                     right_hand_keypoints.append([landmark.x, landmark.y, landmark.z])
#                 frame_keypoints['right_hand'] = right_hand_keypoints

#             # 현재 프레임의 키포인트를 전체 리스트에 추가
#             holistic_keypoints_list.append(frame_keypoints)

#         # 키포인트 데이터를 numpy 배열로 변환하고 저장
#         np.save(f'{dest_path}/{os.path.basename(video_url).split(".")[0]}_holistic.npy', holistic_keypoints_list)

#     # 비디오 캡처 해제
#     cap.release()

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
#         future = executor.submit(extract_keypoints_from_url, video_url, output_path)
#         futures.append(future)

#     print("file listup finish.. threads start..")

#     for future in tqdm(futures, colour='blue'):
#         future.result()

#     executor.shutdown()

# working_threads()



'''
저장한 npy 파일을 로드함
nparray형태로 저장됨
! 로드할 때에는 반드시 allow_pickle=True 값을 줘야 함 !
형태 : { 'left_hand' : [[x축, y축, z축], [...]]}
'''
# def inspect_npy_file(file_path):
#     # .npy 파일 로드
#     keypoints_array = np.load(file_path, allow_pickle=True)
    
#     # 데이터의 타입과 구조 확인
#     print(f"Data type: {type(keypoints_array)}")
#     print(f"Data shape: {keypoints_array.shape if hasattr(keypoints_array, 'shape') else 'No shape (likely a list of dicts)'}")

#     # 데이터 샘플 확인 (앞부분 출력)
#     print("Sample data:")
#     if isinstance(keypoints_array, list):
#         for i, frame in enumerate(keypoints_array[:3]):  # 첫 3 프레임만 출력
#             print(f"Frame {i}: {frame}")
#     else:
#         print(keypoints_array)

# # 파일 경로 지정
# file_path = './keypoint/MOV000246496_700X466_holistic.npy'
# inspect_npy_file(file_path)

mp4_list = []
filename_list = []
keypoints_list = []
# title_list = []

# # 가장 긴 frames 길이를 찾기 위한 변수
# max_frames = 0
    
# 디렉토리 내 모든 npy 파일 순회
# for filename in os.listdir(output_path):
#     if filename.endswith(".npy"):
#         file_path = os.path.join(output_path, filename)
#         keypoints_array = np.load(file_path, allow_pickle=True)
#         mp4_list.append((filename, keypoints_array))

# with open('test_mp4_list.pkl', 'wb') as f:
#     pickle.dump(mp4_list, f)




with open('test_mp4_list.pkl', 'rb') as f:
    mp4_list = pickle.load(f)

for filename, _ in mp4_list:
    filename_list.append(filename)
    keypoints_list.append(_)

def match_and_get_titles_and_one_hot(npy_files_name, raw_data):
    y = []
    matched_rows = []

    # 고유 Title 정수 인코딩 매핑 생성 (Title을 기준으로)
    unique_titles = raw_data["Title"].unique()
    title_to_index = {title: idx for idx, title in enumerate(unique_titles)}
    raw_data["Title_Encoded"] = raw_data["Title"].map(title_to_index)

    print("\n🔍 디버깅: name_list와 raw_data 비교")
    
    for name in npy_files_name:
        # '_holistic.npy' 앞부분만 추출
        name_base = name.split("_holistic.npy")[0]
        print(f"\n🔹 현재 파일: {name} -> 비교할 부분: {name_base}")
        
        for _, row in raw_data.iterrows():
            d = row["SubDescription"]
            # 마지막 '/' 뒤의 파일명만 추출하고, 확장자(.mp4 등)를 제거
            d_base_name = os.path.basename(d).strip()[:-4]
            print(f"  - 비교 대상: {d_base_name}")
            
            if name_base == d_base_name:
                print("  ✅ 매칭됨!")
                y.append(row["Title"])  # 텍스트 레이블 저장
                matched_rows.append(row.to_dict())
                break

    matched_data = pd.DataFrame(matched_rows)

    if matched_data.empty:
        print("⚠ 매칭된 데이터가 없습니다.")
        return [], np.array([])

    if "Title_Encoded" not in matched_data.columns:
        print("⚠ 'Title_Encoded' 컬럼이 없습니다.")
        return [], np.array([])

    encoded_labels = matched_data["Title_Encoded"].values
    one_hot_labels = to_categorical(encoded_labels, num_classes=len(unique_titles))

    return y, one_hot_labels

# 사용 예:
titles, one_hot_labels = match_and_get_titles_and_one_hot(filename_list, raw_data)

print("Titles:", titles)
print("One-Hot Encoded Labels:", one_hot_labels)

from tensorflow.keras.preprocessing.sequence import pad_sequences


y = one_hot_labels
X = keypoints_list

def extract_keypoints(frame_keypoints):
    """각 프레임의 랜드마크(dict)를 (126,) 형태의 numpy 배열로 변환"""
    keypoints = np.zeros((42, 3))  # 왼손(21) + 오른손(21) = 42개
    if "left_hand" in frame_keypoints:
        keypoints[:21] = frame_keypoints["left_hand"]  # 왼손 21개 점
    if "right_hand" in frame_keypoints:
        keypoints[21:] = frame_keypoints["right_hand"]  # 오른손 21개 점
    
    return keypoints.flatten()  # (42, 3) → (126,)


# 모든 프레임 변환
X_array = [np.array([extract_keypoints(frame) for frame in video], dtype="float32") for video in keypoints_list]


MAX_FRAMES = 407  # 비디오의 최대 프레임 길이 (예제 값)
# 시퀀스 패딩 적용
X_padded = [pad_sequences(video, maxlen=MAX_FRAMES, dtype="float32", padding="post", truncating="post") for video in X_array]

# shape이 다를 경우 디버깅
# for i, video in enumerate(X_padded):
#     print(f"Video {i} shape after padding: {video.shape}")

# 2️⃣ **각 비디오를 동일한 길이(MAX_FRAMES)로 패딩**
X_padded = pad_sequences(X_array, maxlen=MAX_FRAMES, dtype="float32", padding="post", truncating="post")
X_padded = X_padded / np.max(X_padded)  # 최댓값 기준 정규화

# # 3️⃣ **배치 형태를 CNN 입력 형식에 맞추기 (batch, frames, features, 1)**
# X_padded = np.expand_dims(X_padded, axis=-1)  # (10, 407, 126, 1)

print("Final X_padded.shape:", X_padded.shape)  # (10, 407, 126, 1)

X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape) # (8, 407, 126, 1)
print(X_test.shape, y_test.shape) # (2, 407, 126, 1)

# 레이블 데이터를 one-hot 인코딩 (이미 되어 있으면 생략 가능)
if y_train.ndim == 1:  # y_train이 정수형 레이블인 경우
    num_classes = 3612  # 클래스 개수
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

# CNN 모델 정의
model = Sequential([
    Input(shape=(MAX_FRAMES, 126)),  # (407, 126) 형태 입력
    Bidirectional(LSTM(128, return_sequences=True)),  
    Bidirectional(LSTM(64)),  
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3577, activation='softmax')
])


model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 모델 훈련
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300)

# 모델 저장
model.save("sign_language_lstm_model.h5")