"""
Système Complet - Détection Intensité de Douleur
1. Annotation automatique du dataset
2. Entraînement du modèle
3. Visualisation temps réel avec tracking
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ============================================================
# PARTIE 1 : ANNOTATION AUTOMATIQUE
# ============================================================

def calculate_intensity_label(row):
    """
    Formule pour générer les labels d'intensité (0-10)
    Basé sur FACS (Facial Action Coding System)
    """
    intensity = 0.0
    
    # Bouche (40%)
    mouth = row['mouth_opening']
    if mouth > 100:
        intensity += 4.0
    elif mouth > 60:
        intensity += 3.0
    elif mouth > 30:
        intensity += 2.0
    elif mouth > 10:
        intensity += 1.0
    
    # Yeux (30%)
    eye_avg = (row['left_eye_opening'] + row['right_eye_opening']) / 2
    if eye_avg < 12:
        intensity += 3.0
    elif eye_avg < 18:
        intensity += 2.0
    elif eye_avg < 25:
        intensity += 1.0
    
    # Sourcils (20%)
    if row['eyebrow_distance'] < 355:
        intensity += 2.0
    elif row['eyebrow_distance'] < 370:
        intensity += 1.0
    
    # Largeur bouche (10%)
    if row['mouth_width'] > 200:
        intensity += 1.0
    
    return min(intensity, 10.0)


def annotate_dataset(csv_path='pain_features_complete.csv'):
    """
    Annote le dataset avec les labels d'intensité
    """
    print("=" * 60)
    print("📝 ÉTAPE 1 : ANNOTATION DU DATASET")
    print("=" * 60)
    
    df = pd.read_csv(csv_path)
    print(f"✅ Chargé: {len(df)} frames")
    
    # Calculer intensité pour chaque frame
    df['intensity'] = df.apply(calculate_intensity_label, axis=1)
    
    # Sauvegarder
    output = 'pain_dataset_annotated.csv'
    df.to_csv(output, index=False)
    
    print(f"✅ Sauvegardé: {output}")
    print(f"\n📊 Distribution Intensité:")
    print(f"   Moyenne: {df['intensity'].mean():.2f}")
    print(f"   Min: {df['intensity'].min():.1f}")
    print(f"   Max: {df['intensity'].max():.1f}")
    
    return df


# ============================================================
# PARTIE 2 : ENTRAÎNEMENT DU MODÈLE
# ============================================================

def train_intensity_model(df):
    """
    Entraîne un modèle de régression pour prédire l'intensité
    """
    print("\n" + "=" * 60)
    print("🤖 ÉTAPE 2 : ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60)
    
    # Features
    feature_cols = ['eyebrow_distance', 'left_eye_opening', 'right_eye_opening',
                   'mouth_opening', 'mouth_width', 'left_cheek_elevation',
                   'right_cheek_elevation', 'left_eyebrow_angle', 
                   'right_eyebrow_angle', 'face_aspect_ratio']
    
    # Garder seulement les colonnes qui existent
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].values
    y = df['intensity'].values
    
    print(f"📊 Features: {len(feature_cols)}")
    print(f"📊 Samples: {len(X)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entraînement
    print("\n🔄 Entraînement Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n✅ RÉSULTATS:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {np.sqrt(mse):.4f}")
    
    # Sauvegarder
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'mse': mse,
        'mae': mae
    }
    joblib.dump(model_data, 'intensity_model.pkl')
    print(f"\n💾 Modèle sauvegardé: intensity_model.pkl")
    
    # Feature importance
    print(f"\n📈 Importance des Features:")
    importance = sorted(zip(feature_cols, model.feature_importances_), 
                       key=lambda x: x[1], reverse=True)
    for feat, imp in importance:
        print(f"   {feat:25s}: {imp:.4f}")
    
    return model, feature_cols


# ============================================================
# PARTIE 3 : VISUALISATION TEMPS RÉEL
# ============================================================

class RealTimePainDetector:
    """
    Détection d'intensité en temps réel avec tracking des points
    """
    
    def __init__(self, model_path='intensity_model.pkl'):
        # Charger modèle
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        
        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Historique pour courbe
        self.intensity_history = []
        self.max_history = 100
        
        print("✅ Modèle et MediaPipe chargés")
    
    def extract_features(self, landmarks, h, w):
        """Extrait les features"""
        lm = landmarks
        
        features = {
            'eyebrow_distance': np.linalg.norm(
                np.array([lm[70].x*w, lm[70].y*h]) - 
                np.array([lm[300].x*w, lm[300].y*h])
            ),
            'left_eye_opening': np.linalg.norm(
                np.array([lm[159].x*w, lm[159].y*h]) - 
                np.array([lm[145].x*w, lm[145].y*h])
            ),
            'right_eye_opening': np.linalg.norm(
                np.array([lm[386].x*w, lm[386].y*h]) - 
                np.array([lm[374].x*w, lm[374].y*h])
            ),
            'mouth_opening': np.linalg.norm(
                np.array([lm[13].x*w, lm[13].y*h]) - 
                np.array([lm[14].x*w, lm[14].y*h])
            ),
            'mouth_width': np.linalg.norm(
                np.array([lm[61].x*w, lm[61].y*h]) - 
                np.array([lm[291].x*w, lm[291].y*h])
            ),
            'left_cheek_elevation': lm[6].y*h - lm[205].y*h,
            'right_cheek_elevation': lm[6].y*h - lm[425].y*h,
            'left_eyebrow_angle': 157.0,  # Approximation
            'right_eyebrow_angle': 157.0,
            'face_aspect_ratio': 0.79
        }
        
        return features
    
    def predict_intensity(self, features):
        """Prédit l'intensité avec le modèle"""
        X = np.array([[features.get(col, 0) for col in self.feature_cols]])
        intensity = self.model.predict(X)[0]
        return np.clip(intensity, 0, 10)
    
    def get_color_for_intensity(self, intensity):
        """Couleur selon l'intensité"""
        if intensity < 3:
            return (0, 255, 0)    # Vert
        elif intensity < 6:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)    # Rouge
    
    def draw_intensity_bar(self, frame, intensity):
        """Dessine la barre d'intensité"""
        h, w = frame.shape[:2]
        
        # Barre de fond
        bar_x, bar_y = 30, 50
        bar_width, bar_height = 30, 300
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Barre d'intensité
        fill_height = int((intensity / 10) * bar_height)
        color = self.get_color_for_intensity(intensity)
        cv2.rectangle(frame, 
                     (bar_x, bar_y + bar_height - fill_height),
                     (bar_x + bar_width, bar_y + bar_height),
                     color, -1)
        
        # Texte
        cv2.putText(frame, f"{intensity:.1f}/10", (bar_x - 10, bar_y + bar_height + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "INTENSITE", (bar_x - 20, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Labels
        cv2.putText(frame, "10", (bar_x + 35, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "5", (bar_x + 35, bar_y + bar_height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "0", (bar_x + 35, bar_y + bar_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_mini_graph(self, frame):
        """Dessine mini graphique d'historique"""
        if len(self.intensity_history) < 2:
            return
        
        h, w = frame.shape[:2]
        graph_x, graph_y = w - 220, 50
        graph_w, graph_h = 200, 100
        
        # Fond
        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x, graph_y), 
                     (graph_x + graph_w, graph_y + graph_h),
                     (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Dessiner courbe
        points = []
        history = self.intensity_history[-50:]  # Derniers 50 points
        for i, val in enumerate(history):
            x = graph_x + int((i / max(len(history)-1, 1)) * graph_w)
            y = graph_y + graph_h - int((val / 10) * graph_h)
            points.append((x, y))
        
        for i in range(1, len(points)):
            color = self.get_color_for_intensity(history[i])
            cv2.line(frame, points[i-1], points[i], color, 2)
        
        cv2.putText(frame, "Historique", (graph_x, graph_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_frame(self, frame):
        """Traite une frame et retourne le résultat"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb)
        
        intensity = 0.0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Dessiner les points
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Points clés en couleur
            key_points = {
                'mouth': [13, 14, 61, 291],
                'eyes': [159, 145, 386, 374],
                'eyebrows': [70, 300]
            }
            
            colors = {'mouth': (0, 0, 255), 'eyes': (255, 255, 0), 'eyebrows': (255, 0, 0)}
            
            for region, indices in key_points.items():
                for idx in indices:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, colors[region], -1)
            
            # Extraire features et prédire
            features = self.extract_features(face_landmarks.landmark, h, w)
            intensity = self.predict_intensity(features)
            
            # Historique
            self.intensity_history.append(intensity)
            if len(self.intensity_history) > self.max_history:
                self.intensity_history.pop(0)
        
        # Dessiner UI
        self.draw_intensity_bar(frame, intensity)
        self.draw_mini_graph(frame)
        
        # Status
        color = self.get_color_for_intensity(intensity)
        status = "Douleur Legere" if intensity < 3 else "Douleur Moderee" if intensity < 6 else "Douleur Severe"
        cv2.putText(frame, status, (w//2 - 100, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame, intensity
    
    def run_on_video(self, video_path):
        """Lance sur une vidéo"""
        print(f"\n🎬 Lecture: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la vidéo")
            return
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame, intensity = self.process_frame(frame)
            
            cv2.imshow('Detection Intensite Douleur', frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_on_webcam(self):
        """Lance sur la webcam"""
        print("\n📹 Webcam activée (Appuyez 'q' pour quitter)")
        
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Miroir
            frame, intensity = self.process_frame(frame)
            
            cv2.imshow('Detection Intensite Douleur - Webcam', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_on_frames_folder(self, folder_path):
        """Lance sur un dossier de frames"""
        folder = Path(folder_path)
        frames = sorted(folder.glob('*.jpg')) + sorted(folder.glob('*.png'))
        
        if not frames:
            print(f"❌ Aucune image dans {folder}")
            return
        
        print(f"\n🖼️ {len(frames)} frames trouvées")
        
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            
            frame, intensity = self.process_frame(frame)
            
            cv2.imshow('Detection Intensite Douleur', frame)
            
            key = cv2.waitKey(100)  # 100ms entre frames
            if key & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("🎯 SYSTÈME DE DÉTECTION D'INTENSITÉ DE DOULEUR")
    print("=" * 60)
    
    # Étape 1: Annotation
    df = annotate_dataset('pain_features_complete.csv')
    
    # Étape 2: Entraînement
    model, feature_cols = train_intensity_model(df)
    
    # Étape 3: Test temps réel
    print("\n" + "=" * 60)
    print("🎬 ÉTAPE 3 : VISUALISATION TEMPS RÉEL")
    print("=" * 60)
    
    detector = RealTimePainDetector('intensity_model.pkl')
    
    print("\nChoisissez une option:")
    print("1. Tester sur dossier de frames")
    print("2. Tester sur webcam")
    print("3. Tester sur vidéo")
    
    choice = input("\nVotre choix (1/2/3): ").strip()
    
    if choice == '1':
        folder = input("Chemin du dossier: ").strip().strip('"')
        if not folder:
            folder = r"C:\Users\MSI\Desktop\pain_dataset\Pictures\Modified\S001\Algometer Pain\Colour frames"
        detector.run_on_frames_folder(folder)
    
    elif choice == '2':
        detector.run_on_webcam()
    
    elif choice == '3':
        video = input("Chemin de la vidéo: ").strip().strip('"')
        detector.run_on_video(video)
    
    print("\n✅ Terminé!")


if __name__ == "__main__":
    main()