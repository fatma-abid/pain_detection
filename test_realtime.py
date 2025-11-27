"""
Test Temps Réel - Utilise le modèle déjà entraîné
Pas de réentraînement, juste la visualisation
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path

class RealTimePainDetector:
    def __init__(self, model_path='intensity_model.pkl'):
        # Charger modèle
        print("🔄 Chargement du modèle...")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        print(f"✅ Modèle chargé: {model_path}")
        
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
        
        self.intensity_history = []
        self.max_history = 100
    
    def extract_features(self, landmarks, h, w):
        lm = landmarks
        features = {
            'eyebrow_distance': np.linalg.norm(
                np.array([lm[70].x*w, lm[70].y*h]) - np.array([lm[300].x*w, lm[300].y*h])),
            'left_eye_opening': np.linalg.norm(
                np.array([lm[159].x*w, lm[159].y*h]) - np.array([lm[145].x*w, lm[145].y*h])),
            'right_eye_opening': np.linalg.norm(
                np.array([lm[386].x*w, lm[386].y*h]) - np.array([lm[374].x*w, lm[374].y*h])),
            'mouth_opening': np.linalg.norm(
                np.array([lm[13].x*w, lm[13].y*h]) - np.array([lm[14].x*w, lm[14].y*h])),
            'mouth_width': np.linalg.norm(
                np.array([lm[61].x*w, lm[61].y*h]) - np.array([lm[291].x*w, lm[291].y*h])),
            'left_cheek_elevation': lm[6].y*h - lm[205].y*h,
            'right_cheek_elevation': lm[6].y*h - lm[425].y*h,
            'left_eyebrow_angle': 157.0,
            'right_eyebrow_angle': 157.0,
            'face_aspect_ratio': 0.79
        }
        return features
    
    def predict_intensity(self, features):
        X = np.array([[features.get(col, 0) for col in self.feature_cols]])
        intensity = self.model.predict(X)[0]
        return np.clip(intensity, 0, 10)
    
    def get_color(self, intensity):
        if intensity < 3: return (0, 255, 0)      # Vert
        elif intensity < 6: return (0, 165, 255)  # Orange
        else: return (0, 0, 255)                  # Rouge
    
    def draw_ui(self, frame, intensity):
        h, w = frame.shape[:2]
        
        # Barre d'intensité
        bar_x, bar_y, bar_w, bar_h = 30, 50, 30, 300
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
        fill = int((intensity/10) * bar_h)
        color = self.get_color(intensity)
        cv2.rectangle(frame, (bar_x, bar_y+bar_h-fill), (bar_x+bar_w, bar_y+bar_h), color, -1)
        cv2.putText(frame, f"{intensity:.1f}/10", (bar_x-10, bar_y+bar_h+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, "INTENSITE", (bar_x-20, bar_y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Mini graphique
        if len(self.intensity_history) > 1:
            gx, gy, gw, gh = w-220, 50, 200, 100
            cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (30,30,30), -1)
            hist = self.intensity_history[-50:]
            for i in range(1, len(hist)):
                x1 = gx + int(((i-1)/max(len(hist)-1,1))*gw)
                y1 = gy + gh - int((hist[i-1]/10)*gh)
                x2 = gx + int((i/max(len(hist)-1,1))*gw)
                y2 = gy + gh - int((hist[i]/10)*gh)
                cv2.line(frame, (x1,y1), (x2,y2), self.get_color(hist[i]), 2)
        
        # Status
        status = "LEGERE" if intensity < 3 else "MODEREE" if intensity < 6 else "SEVERE"
        cv2.putText(frame, f"Douleur {status}", (w//2-100, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        intensity = 0.0
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            
            # Dessiner mesh
            self.mp_drawing.draw_landmarks(frame, face, self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Points clés colorés
            for idx, color in [(13,(0,0,255)), (14,(0,0,255)), (61,(0,0,255)), (291,(0,0,255)),
                               (159,(255,255,0)), (145,(255,255,0)), (386,(255,255,0)), (374,(255,255,0)),
                               (70,(255,0,0)), (300,(255,0,0))]:
                lm = face.landmark[idx]
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, color, -1)
            
            features = self.extract_features(face.landmark, h, w)
            intensity = self.predict_intensity(features)
            self.intensity_history.append(intensity)
            if len(self.intensity_history) > self.max_history:
                self.intensity_history.pop(0)
        
        self.draw_ui(frame, intensity)
        return frame, intensity
    
    def run_webcam(self):
        print("\n📹 Webcam (Appuyez 'q' pour quitter)")
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame, _ = self.process_frame(frame)
            cv2.imshow('Detection Intensite Douleur', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video(self, path):
        print(f"\n🎬 Vidéo: {path}")
        cap = cv2.VideoCapture(str(path))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame, _ = self.process_frame(frame)
            cv2.imshow('Detection Intensite Douleur', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
    
    def run_frames(self, folder):
        folder = Path(folder)
        frames = sorted(folder.glob('*.jpg')) + sorted(folder.glob('*.png'))
        print(f"\n🖼️ {len(frames)} frames")
        for f in frames:
            img = cv2.imread(str(f))
            if img is None: continue
            img, _ = self.process_frame(img)
            cv2.imshow('Detection Intensite Douleur', img)
            if cv2.waitKey(100) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 50)
    print("🎯 TEST TEMPS RÉEL - INTENSITÉ DOULEUR")
    print("=" * 50)
    
    detector = RealTimePainDetector('intensity_model.pkl')
    
    print("\n1. Webcam")
    print("2. Vidéo")
    print("3. Dossier frames")
    
    choice = input("\nChoix (1/2/3): ").strip()
    
    if choice == '1':
        detector.run_webcam()
    elif choice == '2':
        path = input("Chemin vidéo: ").strip().strip('"')
        detector.run_video(path)
    elif choice == '3':
        path = input("Chemin dossier: ").strip().strip('"')
        detector.run_frames(path)