# pain_detector.py
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

class PainPredictor:
    """
    Classe pour prédire l'intensité de la douleur sur de nouvelles images
    """
    def __init__(self, model_path='best_pain_model.pkl'):
        # Chargement du modèle
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_name = model_data['model_name']
        self.accuracy = model_data['accuracy']

        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Labels
        self.pain_labels = {0: 'Neutral', 1: 'Posed Pain', 2: 'Algometer Pain', 3: 'Laser Pain'}
        self.pain_colors = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'}

    def extract_landmarks(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None, image
        face_landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append([landmark.x * w, landmark.y * h, landmark.z * w])
        return np.array(landmarks), image

    def calculate_features(self, landmarks):
        if landmarks is None:
            return None
        features = {}
        # 10 features principales
        left_brow, right_brow = landmarks[70], landmarks[300]
        features['eyebrow_distance'] = np.linalg.norm(left_brow - right_brow)

        left_eye_top, left_eye_bottom = landmarks[159], landmarks[145]
        features['left_eye_opening'] = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_top, right_eye_bottom = landmarks[386], landmarks[374]
        features['right_eye_opening'] = np.linalg.norm(right_eye_top - right_eye_bottom)

        upper_lip, lower_lip = landmarks[13], landmarks[14]
        features['mouth_opening'] = np.linalg.norm(upper_lip - lower_lip)
        left_mouth, right_mouth = landmarks[61], landmarks[291]
        features['mouth_width'] = np.linalg.norm(left_mouth - right_mouth)

        left_cheek, right_cheek, nose_bridge = landmarks[205], landmarks[425], landmarks[6]
        features['left_cheek_elevation'] = nose_bridge[1] - left_cheek[1]
        features['right_cheek_elevation'] = nose_bridge[1] - right_cheek[1]

        # Angles sourcils
        for side, points in zip(['left', 'right'], [(70,63,105),(300,293,334)]):
            p1, p2, p3 = landmarks[points[0]], landmarks[points[1]], landmarks[points[2]]
            v1, v2 = p1 - p2, p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1,1)
            features[f'{side}_eyebrow_angle'] = np.degrees(np.arccos(cos_angle))

        # Ratio visage
        face_width = np.linalg.norm(landmarks[234] - landmarks[454])
        face_height = np.linalg.norm(landmarks[10] - landmarks[152])
        features['face_aspect_ratio'] = face_width / (face_height + 1e-6)

        return features

    def predict_single_image(self, image_path, visualize=True):
        landmarks, image = self.extract_landmarks(image_path)
        if landmarks is None:
            return None
        features = self.calculate_features(landmarks)
        features_array = np.array(list(features.values())).reshape(1, -1)
        features_normalized = self.scaler.transform(features_array)
        prediction = self.model.predict(features_normalized)[0]
        probabilities = self.model.predict_proba(features_normalized)[0] if hasattr(self.model, 'predict_proba') else None
        pain_type = self.pain_labels[prediction]

        # Visualisation matplotlib
        vis_image = None
        if visualize:
            fig, ax = plt.subplots(figsize=(6,6))
            for x,y,z in landmarks:
                cv2.circle(image, (int(x), int(y)), 1, (0,255,0), -1)
            vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(vis_image)
            ax.axis('off')
            ax.set_title(f'Prediction: {pain_type}')
            plt.close(fig)

        return {
            'prediction': prediction,
            'pain_type': pain_type,
            'probabilities': probabilities,
            'features': features,
            'visualization': vis_image
        }

    def predict_batch(self, image_folder, pattern='*.jpg'):
        image_folder = Path(image_folder)
        image_paths = list(image_folder.glob(pattern))
        results = []
        for img_path in image_paths:
            res = self.predict_single_image(img_path, visualize=False)
            if res:
                results.append({'image': img_path.name, 'prediction': res['pain_type'], 'label': res['prediction'], **res['features']})
        return pd.DataFrame(results)

    def test_on_test_set(self, csv_path='pain_features_complete.csv', n_samples=5):
        df = pd.read_csv(csv_path)
        sample = df.sample(n=min(n_samples,len(df)), random_state=42)
        correct = 0
        for idx, row in sample.iterrows():
            res = self.predict_single_image(row['image_path'])
            if res and res['prediction'] == row['pain_label']:
                correct += 1
        accuracy = correct / len(sample) * 100
        print(f"Accuracy sur l'échantillon : {accuracy:.2f}% ({correct}/{len(sample)})")
