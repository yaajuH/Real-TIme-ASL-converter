import cv2
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp

# Initialize MediaPipe with multiple configurations
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Different hand detectors for different scenarios
hands_standard = mp_hands.Hands(
    static_image_mode=False,  # False for video
    max_num_hands=1,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)

hands_complex = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)

# Load the trained model
model = tf.keras.models.load_model('asl_model.keras')

# Load both mapping files
with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

with open('label_to_int.json', 'r') as f:
    label_to_int = json.load(f)

# Create reverse mapping from integer to label
int_to_label = {int(v): k for k, v in label_to_int.items()}

def preprocess_frame(frame):
    """Enhanced frame preprocessing pipeline."""
    frame = cv2.resize(frame, (320, 320))
    
    versions = []
    versions.append(frame)
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    versions.append(enhanced)
    
    bright = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
    versions.append(bright)
    
    contrast = cv2.convertScaleAbs(frame, alpha=1.8, beta=0)
    versions.append(contrast)
    
    return versions

def extract_landmarks(image, hands_detector):
    """Extract landmarks with normalization."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = []
        hand_landmarks = results.multi_hand_landmarks[0]
        
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        cx = np.mean(x_coords)
        cy = np.mean(y_coords)
        scale = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
        
        for point in hand_landmarks.landmark:
            landmarks.extend([
                (point.x - cx) / scale,
                (point.y - cy) / scale,
                point.z
            ])
        
        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return np.array(landmarks)
    return None

def predict_sign(frame):
    """Predict sign with multiple detection attempts."""
    frame_versions = preprocess_frame(frame)
    
    for frame_version in frame_versions:
        # Try standard detector
        landmarks = extract_landmarks(frame_version, hands_standard)
        if landmarks is not None:
            landmarks = np.expand_dims(landmarks, axis=0)
            prediction = model.predict(landmarks, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class] * 100
            
            # Use the correct mapping to get the label
            predicted_label = int_to_label[predicted_class]
            return predicted_label, confidence
        
        # Try complex detector
        landmarks = extract_landmarks(frame_version, hands_complex)
        if landmarks is not None:
            landmarks = np.expand_dims(landmarks, axis=0)
            prediction = model.predict(landmarks, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class] * 100
            
            # Use the correct mapping to get the label
            predicted_label = int_to_label[predicted_class]
            return predicted_label, confidence
    
    return None, None

def test_video_stream():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit")
    print("Available classes:", sorted(label_mapping.keys()))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        predicted_sign, confidence = predict_sign(frame)
        
        if predicted_sign is not None:
            text = f"Sign: {predicted_sign} ({confidence:.1f}%)"
            cv2.putText(frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            bar_length = int(confidence * 2)
            cv2.rectangle(frame, (10, 50), (10 + bar_length, 70),
                         (0, 255, 0), -1)
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("ASL Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting video stream...")
    test_video_stream()
