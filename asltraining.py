import cv2
import numpy as np
import os
import json
import tensorflow as tf
from keras import layers, models, callbacks
import mediapipe as mp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initialize MediaPipe with multiple configurations
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Different hand detectors for different scenarios
hands_standard = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.1,  # Very low threshold to catch more cases
    min_tracking_confidence=0.1
)

hands_complex = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,  # Allow detection of multiple hands to improve chances
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)

# Global Constants
DATASET_PATH = "/Users/user/Desktop/asl_dataset"
LANDMARKS_SIZE = 63 # 21 * 3 = 63

def preprocess_image(image):
    """Enhanced image preprocessing pipeline."""
    # Resize to standard size
    image = cv2.resize(image, (320, 320))
    
    # Create multiple versions with different preprocessing
    versions = []
    
    # Version 1: Original resized
    versions.append(image)
    
    # Version 2: Enhanced contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    versions.append(enhanced)
    
    # Version 3: Increased brightness
    bright = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
    versions.append(bright)
    
    # Version 4: Increased contrast
    contrast = cv2.convertScaleAbs(image, alpha=1.8, beta=0)
    versions.append(contrast)
    
    return versions

def extract_landmarks(image, hands_detector):
    """Extract landmarks from detected hand."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = []
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Calculate hand center and scale for normalization
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        cx = np.mean(x_coords)
        cy = np.mean(y_coords)
        scale = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
        
        for point in hand_landmarks.landmark:
            # Normalize coordinates relative to hand center and scale
            landmarks.extend([
                (point.x - cx) / scale,
                (point.y - cy) / scale,
                point.z
            ])
        return np.array(landmarks)
    return None

def process_image_for_training(image_path):
    """Process a single image and extract landmarks using the successful detection method."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Try different image versions
    image_versions = preprocess_image(image)
    
    for img_version in image_versions:
        # Try normal orientation with standard detector
        landmarks = extract_landmarks(img_version, hands_standard)
        if landmarks is not None:
            return landmarks
            
        # Try flipped version
        flipped = cv2.flip(img_version, 1)
        landmarks = extract_landmarks(flipped, hands_standard)
        if landmarks is not None:
            return landmarks
        
        # Try with complex detector
        landmarks = extract_landmarks(img_version, hands_complex)
        if landmarks is not None:
            return landmarks
        
        # Try rotated versions
        for angle in [15, -15, 30, -30]:
            rows, cols = img_version.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(img_version, M, (cols, rows))
            landmarks = extract_landmarks(rotated, hands_complex)
            if landmarks is not None:
                return landmarks
    
    return None

def process_dataset():
    """Process dataset and extract landmarks."""
    features = []
    labels = []
    
    # Create label mapping
    folders = sorted(os.listdir(DATASET_PATH))
    folders = [f for f in folders if os.path.isdir(os.path.join(DATASET_PATH, f))]
    label_mapping = {folder: folder for folder in folders}
    
    total_processed = 0
    total_images = 0
    
    for folder in folders:
        folder_path = os.path.join(DATASET_PATH, folder)
        print(f"\nProcessing class: {folder}")
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        folder_total = len(image_files)
        folder_successful = 0
        total_images += folder_total
        
        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            
            try:
                landmarks = process_image_for_training(img_path)
                if landmarks is not None:
                    features.append(landmarks)
                    labels.append(folder)
                    folder_successful += 1
                    total_processed += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        success_rate = (folder_successful/folder_total)*100
        print(f"Class {folder}: {folder_successful}/{folder_total} processed ({success_rate:.1f}%)")
    
    print(f"\nData Sumaary:")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {total_processed}")
    print(f"Success rate: {(total_processed/total_images)*100:.2f}%")
    
    # Save label mapping
    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
        
    return np.array(features), np.array(labels)

def create_model(num_classes):
    """Create model architecture."""
    model = models.Sequential([
        layers.Input(shape=(LANDMARKS_SIZE,)),
        layers.BatchNormalization(),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ... [previous code remains the same until callbacks definition] ...

def train_model(features, labels):
    """Train the model."""
    # Convert labels to integers
    unique_labels = sorted(list(set(labels)))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    
    # Save the integer mapping
    with open('label_to_int.json', 'w') as f:
        json.dump(label_to_int, f, indent=2)
    
    # Convert labels to integers
    int_labels = np.array([label_to_int[label] for label in labels])
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        features, int_labels, test_size=0.2, stratify=int_labels, random_state=42
    )
    
    # Create and compile model
    model = create_model(len(unique_labels))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks with updated file extension
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        callbacks.ModelCheckpoint(
            'best_model.keras',  # Changed from .h5 to .keras
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return model, history

def main():
    print("Processing dataset...")
    features, labels = process_dataset()
    
    print("\nTraining model...")
    model, history = train_model(features, labels)
    
    # Save final model with updated extension
    model.save('asl_model.keras')  # Changed from .h5 to .keras
    print("\nModel saved as 'asl_model.keras'")
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()
