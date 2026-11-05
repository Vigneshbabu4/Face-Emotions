import cv2
from fer import FER

emotion_detector = FER()
cap = cv2.VideoCapture(1)

emotion_colors = {
    'angry': (0, 0, 255),      
    'sad': (255, 255, 255),     
    'happy': (0, 255, 0),       
    'unknown': (0, 0, 0),       
    'surprised': (0, 255, 255), 
    'neutral': (255, 0, 0)      
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    for (x, y, w, h) in faces:
        
        face = frame[y:y + h, x:x + w]
        emotion, _ = emotion_detector.top_emotion(face)
        if emotion is None:
            emotion = "unknown"

        print(f"Detected Emotion: {emotion}")

        
        color = emotion_colors.get(emotion, (0, 0, 0))  

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        
        cv2.putText(frame, f"{emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()