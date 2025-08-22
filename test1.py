# 방식 1) 
#   OpenCV의 Haar Cascade를 사용하여 얼굴의 위치만 파악한 후, 
#   .tflite 모델을 이용하여 이미지 자체를 분석하여 감정 예측
# 장점:
# - 얼굴 특징 점 사용 X  
#  -> 간단, 가벼움 / 속도가 더 빠를 수 있음
# 단점:
# - 정확도가 좀 (많이) 떨어지는 듯.... 다른 방법도 찾고 올리겠습니다 
import cv2
import numpy as np
import tensorflow as tf
# 실시간 처리(컴이 처리 가능한 최대속도)를 할 경우 time 생략 가능
import time

# 전역 변수로 모델과 레이블을 미리 로드
size = 224
model_path = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 감정 분석 함수
def get_emotion(frame):
    """
    주어진 프레임에서 얼굴을 감지하고 감정을 예측합니다.
    """
    gray_for_detection = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_for_detection, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # 첫 번째 얼굴만 분석
        
        face_roi = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (size, size), interpolation=cv2.INTER_AREA)
        normalized_face = np.float32(resized_face / 255.0)
        input_data = normalized_face.reshape(1, size, size, 3)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        emotion_index = np.argmax(output_data)
        emotion = emotion_labels[emotion_index]
        
        return emotion, (x, y, w, h)
    
    return None, None

# 메인 : 함수의 결과를 받아 화면에 그리는 역할
def main():
    cap = cv2.VideoCapture(0)
    
    last_emotion = ""
    last_face_coords = None
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        current_time = time.time()
        
        # 1초마다 감정 분석 함수 호출
        if current_time - last_time >= 1:
            # 함수에서 감정과 얼굴 좌표를 받아옴
            detected_emotion, coords = get_emotion(frame)
            if coords:
                last_emotion = detected_emotion
                last_face_coords = coords
            
            # 콘솔에 감정 출력
            print(last_emotion)
            last_time = current_time

        # 결과 시각화
        if last_face_coords:
            (x, y, w, h) = last_face_coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, last_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            # 얼굴이 감지되지 않으면 텍스트만 표시
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()