import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from datetime import datetime

# Load YOLOv8 model
yolo_model = YOLO('yolov8_trained.pt')

# Load your custom VGG16-based gender classification model
gender_model = load_model('gender_classification_vgg16.h5')

# Function to preprocess image for gender classification
def preprocess_for_gender_classification(image):
    image = cv2.resize(image, (224, 224))  # Resize to 224x224 as expected by VGG16
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Function to check if it's night time
def is_night():
    current_hour = datetime.now().hour
    # Define night hours (e.g., 8 PM to 6 AM)
    return current_hour >= 20 or current_hour <= 6

# Function to filter out small bounding boxes
def is_valid_person_bbox(bbox, min_area=500):
    x_min, y_min, x_max, y_max = bbox
    area = (x_max - x_min) * (y_max - y_min)
    return area > min_area

# Open a connection to the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    men_count = 0
    women_count = 0
    people = []
    women_bboxes = []

    # Detect humans with YOLOv8
    results = yolo_model(frame)

    for result in results:
        for detection in result.boxes:
            confidence = detection.conf[0]  # Confidence score
            if detection.cls == 0 and confidence > 0.5:  # class ID 0 for 'person', confidence threshold
                bbox = detection.xyxy[0].tolist()
                x_min, y_min, x_max, y_max = map(int, bbox)

                # Filter out small bounding boxes
                if is_valid_person_bbox([x_min, y_min, x_max, y_max]):
                    # Extract the human region
                    human_img = frame[y_min:y_max, x_min:x_max]

                    # Prepare and classify with the gender model
                    preprocessed_img = preprocess_for_gender_classification(human_img)
                    predictions = gender_model.predict(preprocessed_img)

                    # Post-process the predictions to get gender
                    gender = 'Female' if predictions[0] > 0.5 else 'Male'

                    # Track counts and store positions of women
                    if gender == 'Male':
                        men_count += 1
                        people.append((x_min, y_min, x_max, y_max, 'Male'))
                    else:
                        women_count += 1
                        women_bboxes.append((x_min, y_min, x_max, y_max))
                        people.append((x_min, y_min, x_max, y_max, 'Female'))

                    # Draw the bounding box and label
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f'Gender: {gender}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Check if it's night and detect lone women
    if is_night() and women_count == 1:  # Only if there's exactly one woman
        wx_min, wy_min, wx_max, wy_max = women_bboxes[0]
        # Highlight the lone woman with a red border and a label
        cv2.rectangle(frame, (wx_min, wy_min), (wx_max, wy_max), (0, 0, 255), 4)
        cv2.putText(frame, 'Lone Woman', (wx_min, wy_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display the number of men and women
    cv2.putText(frame, f'Men: {men_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Women: {women_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv8 + Gender Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()