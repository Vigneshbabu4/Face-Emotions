import cv2
import numpy as np

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
video_capture = cv2.VideoCapture(1)

# Function to simulate shooting
def simulate_shooting():
    print("Target Locked! Shooting...")

# Main loop for face detection and tracking
while True:
    ret, frame = video_capture.read()  # Capture frame from the webcam
    if not ret:
        print("Failed to capture image from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    target_count = len(faces)  # Count the number of detected faces

    for (x, y, w, h) in faces: 
        # Draw a red rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle

        # Display "LOCKED" if the face is close enough for shooting
        if w > 100:  # Assuming the face width > 100 pixels is close enough for shooting
            cv2.putText(frame, "LOCKED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            simulate_shooting()  # Simulate shooting when locked onto a target

    # Display the number of detected targets on the frame
    cv2.putText(frame, f'Target: {target_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the video frame with detections
    cv2.imshow('Face Detection with Target Lock', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('h'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()