# 방법 2) .h5 모델사용
#   일반적으로 TensorFlow나 Keras로 학습된 원본 모델로 .tflite 모델보다 정확도가 높음
# 단점 : 모델 파일 크기가 큼, 저사양 기기에는 느릴 수 있음
# 파일 확장자만 .tflite로 바꾸면 경량화 파일 사용 가능
import cv2
import numpy as np
# keras.models하면서 오류 날 수 있음_ notion에 정리 해 둠
from keras.models import load_model
import time # 생략 가능

# 파일 경로
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model_path = 'fer_model.hdf5' 

# 모델 로드
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure your model file is in the correct directory.")
    exit()

# 표정 라벨링
expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 감정 분석 함수: 감정 값만 반환
def get_emotion(frame):
    """
    주어진 프레임에서 얼굴을 감지하고 감정을 예측합니다.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # 첫 번째 얼굴만 분석
        face_roi = gray[y:y+h, x:x+w]
        
        # 표정 dataset과 똑같은 사이즈 변환
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0

        # 모델을 통해 표정 분석
        output = model.predict(face_roi)
        expression_index = np.argmax(output)
        expression_label = expression_labels[expression_index]
        
        return expression_label
    
    return "No face detected"


#위 모델 경량화 시킨 .tflite 사용
import cv2
import numpy as np
import tensorflow as tf

# --- 모델 및 파일 경로 설정 ---
size = 64

# OpenCV Haar Cascade 얼굴 감지기 로드 (OpenCV 내장 파일)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 경량화된 .tflite 모델 파일 경로
model_path = 'fer_model.tflite'

try:
    # TFLite Interpreter를 사용하여 모델 로드
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    print("Please make sure your model file is in the correct directory.")
    exit()

# 모델 입력 및 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 모델이 학습한 표정 라벨링 (순서가 중요)
expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 감정 분석 함수: 감정 값만 반환
def get_emotion(frame):
    """
    주어진 프레임에서 얼굴을 감지하고 TFLite 모델로 감정을 예측합니다.
    """
    # 얼굴 감지를 위한 흑백 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # 첫 번째 얼굴만 분석
        face_roi = gray[y:y+h, x:x+w]
        
        # TODO: 모델 입력 크기(64x64)에 맞게 변환
        face_roi_resized = cv2.resize(face_roi, (size, size), interpolation=cv2.INTER_AREA)
        # TFLite 모델이 1채널(흑백)을 요구한다고 가정하고 수정합니다.
        face_roi_gray = np.expand_dims(face_roi_resized, axis=-1)
        
        # 모델 입력 형태로 전처리
        input_data = np.expand_dims(face_roi_gray, axis=0).astype(np.float32)
        input_data = input_data / 255.0

        # TFLite 모델을 통해 표정 분석
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        expression_index = np.argmax(output_data)
        expression_label = expression_labels[expression_index]
        
        return expression_label
    
    return "No face detected"