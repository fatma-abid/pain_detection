"""
Real-Time Detection with Calibrated PSPI Model
Should now show ~0 for neutral faces!
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
from pathlib import Path


class CalibratedPSPIDetector:
    """
    Real-time pain intensity detection using calibrated PSPI thresholds
    """

    def __init__(self, model_path='pspi_calibrated_model.pkl'):
        print("🔄 Chargement du modèle calibré...")

        # Load model
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.thresholds = model_data.get('thresholds', {})

        print(f"✅ Modèle: {model_data.get('annotation_method', 'PSPI_Calibrated')}")
        print(f"✅ R² Score: {model_data['metrics']['r2']:.4f}")

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

        # History
        self.intensity_history = []
        self.max_history = 100

        print("✅ Prêt pour la détection!\n")

    def extract_features(self, landmarks, h, w):
        """Extract geometric features from landmarks"""
        lm = landmarks

        features = {
            'eyebrow_distance': np.linalg.norm(
                np.array([lm[70].x * w, lm[70].y * h]) -
                np.array([lm[300].x * w, lm[300].y * h])
            ),
            'left_eye_opening': np.linalg.norm(
                np.array([lm[159].x * w, lm[159].y * h]) -
                np.array([lm[145].x * w, lm[145].y * h])
            ),
            'right_eye_opening': np.linalg.norm(
                np.array([lm[386].x * w, lm[386].y * h]) -
                np.array([lm[374].x * w, lm[374].y * h])
            ),
            'mouth_opening': np.linalg.norm(
                np.array([lm[13].x * w, lm[13].y * h]) -
                np.array([lm[14].x * w, lm[14].y * h])
            ),
            'mouth_width': np.linalg.norm(
                np.array([lm[61].x * w, lm[61].y * h]) -
                np.array([lm[291].x * w, lm[291].y * h])
            ),
            'left_cheek_elevation': lm[6].y * h - lm[205].y * h,
            'right_cheek_elevation': lm[6].y * h - lm[425].y * h,
            'left_eyebrow_angle': 157.0,
            'right_eyebrow_angle': 157.0,
            'face_aspect_ratio': 0.79
        }

        return features

    def predict_intensity(self, features):
        """Predict pain intensity"""
        X = np.array([[features.get(col, 0) for col in self.feature_cols]])
        X_scaled = self.scaler.transform(X)
        intensity = self.model.predict(X_scaled)[0]
        return np.clip(intensity, 0, 10)

    def get_color_for_intensity(self, intensity):
        """Color coding based on intensity"""
        if intensity < 2:
            return (0, 255, 0)  # Green - No/minimal pain
        elif intensity < 4:
            return (0, 255, 255)  # Yellow - Mild
        elif intensity < 6:
            return (0, 165, 255)  # Orange - Moderate
        else:
            return (0, 0, 255)  # Red - Severe

    def get_pain_level(self, intensity):
        """Text description of pain level"""
        if intensity < 1:
            return "Pas de Douleur"
        elif intensity < 3:
            return "Douleur Légère"
        elif intensity < 5:
            return "Douleur Modérée"
        elif intensity < 7:
            return "Douleur Forte"
        else:
            return "Douleur Sévère"

    def draw_ui(self, frame, intensity, features=None):
        """Draw comprehensive UI"""
        h, w = frame.shape[:2]
        color = self.get_color_for_intensity(intensity)

        # === INTENSITY BAR ===
        bar_x, bar_y = 30, 50
        bar_width, bar_height = 35, 300

        # Background
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      (40, 40, 40), -1)

        # Fill based on intensity
        fill_height = int((intensity / 10) * bar_height)
        cv2.rectangle(frame,
                      (bar_x, bar_y + bar_height - fill_height),
                      (bar_x + bar_width, bar_y + bar_height),
                      color, -1)

        # Border
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      (255, 255, 255), 2)

        # Scale markers
        for i in [0, 2, 4, 6, 8, 10]:
            y_pos = bar_y + bar_height - int((i / 10) * bar_height)
            cv2.line(frame, (bar_x - 5, y_pos), (bar_x, y_pos), (255, 255, 255), 2)
            cv2.putText(frame, str(i), (bar_x - 25, y_pos + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Labels
        cv2.putText(frame, "INTENSITE", (bar_x - 20, bar_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{intensity:.2f}", (bar_x - 10, bar_y + bar_height + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # === PAIN LEVEL TEXT ===
        pain_level = self.get_pain_level(intensity)
        text_size = cv2.getTextSize(pain_level, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2

        # Background for text
        cv2.rectangle(frame, (text_x - 10, 10),
                      (text_x + text_size[0] + 10, 50),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (text_x - 10, 10),
                      (text_x + text_size[0] + 10, 50),
                      color, 2)

        cv2.putText(frame, pain_level, (text_x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # === HISTORY GRAPH ===
        if len(self.intensity_history) > 1:
            graph_x, graph_y = w - 230, 50
            graph_w, graph_h = 210, 120

            # Background
            overlay = frame.copy()
            cv2.rectangle(overlay, (graph_x, graph_y),
                          (graph_x + graph_w, graph_y + graph_h),
                          (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Border
            cv2.rectangle(frame, (graph_x, graph_y),
                          (graph_x + graph_w, graph_y + graph_h),
                          (255, 255, 255), 1)

            # Plot line
            history = self.intensity_history[-60:]  # Last 60 frames
            points = []
            for i, val in enumerate(history):
                x = graph_x + int((i / max(len(history) - 1, 1)) * graph_w)
                y = graph_y + graph_h - int((val / 10) * graph_h)
                points.append((x, y))

            # Draw line segments with colors
            for i in range(1, len(points)):
                color_seg = self.get_color_for_intensity(history[i])
                cv2.line(frame, points[i - 1], points[i], color_seg, 2)

            # Grid lines
            for val in [2.5, 5.0, 7.5]:
                y_line = graph_y + graph_h - int((val / 10) * graph_h)
                cv2.line(frame, (graph_x, y_line), (graph_x + graph_w, y_line),
                         (100, 100, 100), 1, cv2.LINE_AA)

            cv2.putText(frame, "Historique", (graph_x + 5, graph_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # === FEATURE VALUES (Debug info) ===
        if features:
            info_x, info_y = 80, 50
            cv2.putText(frame, "Features:", (info_x, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            key_features = {
                'Sourcils': features.get('eyebrow_distance', 0),
                'Yeux': (features.get('left_eye_opening', 0) +
                         features.get('right_eye_opening', 0)) / 2,
                'Bouche': features.get('mouth_opening', 0)
            }

            for i, (name, val) in enumerate(key_features.items(), 1):
                cv2.putText(frame, f"{name}: {val:.1f}",
                            (info_x, info_y + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # === CALIBRATION BADGE ===
        cv2.putText(frame, "PSPI Calibre", (w - 130, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

        # === INSTRUCTIONS ===
        cv2.putText(frame, "Q: Quitter | R: Reset", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def process_frame(self, frame):
        """Process a single frame"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        intensity = 0.0
        features = None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Draw mesh
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Highlight key points
            key_colors = {
                'mouth': (0, 0, 255),  # Red
                'eyes': (255, 255, 0),  # Yellow
                'eyebrows': (255, 0, 255)  # Magenta
            }

            key_points = {
                'mouth': [13, 14, 61, 291],
                'eyes': [159, 145, 386, 374],
                'eyebrows': [70, 300]
            }

            for region, indices in key_points.items():
                for idx in indices:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, key_colors[region], -1)
                    cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)

            # Extract features and predict
            features = self.extract_features(face_landmarks.landmark, h, w)
            intensity = self.predict_intensity(features)

            # Update history
            self.intensity_history.append(intensity)
            if len(self.intensity_history) > self.max_history:
                self.intensity_history.pop(0)
        else:
            # No face detected
            cv2.putText(frame, "Aucun visage detecte", (w // 2 - 150, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw UI
        self.draw_ui(frame, intensity, features)

        return frame, intensity

    def run_webcam(self):
        """Run on webcam"""
        print("📹 Webcam activée")
        print("   Q: Quitter")
        print("   R: Reset historique")
        print("-" * 70)

        cap = cv2.VideoCapture(0)

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror
            frame, intensity = self.process_frame(frame)

            cv2.imshow('Detection Intensite PSPI Calibre', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.intensity_history = []
                print("✅ Historique réinitialisé")

        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Terminé!")

    def run_video(self, path):
        """Run on video file"""
        print(f"🎬 Vidéo: {path}")

        cap = cv2.VideoCapture(str(path))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, intensity = self.process_frame(frame)

            cv2.imshow('Detection Intensite PSPI Calibre', frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_folder(self, folder_path):
        """Run on folder of images"""
        folder = Path(folder_path)
        frames = sorted(list(folder.glob('*.jpg')) + list(folder.glob('*.png')))

        if not frames:
            print(f"❌ Aucune image dans {folder}")
            return

        print(f"🖼️  {len(frames)} frames trouvées")

        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            frame, intensity = self.process_frame(frame)

            cv2.imshow('Detection Intensite PSPI Calibre', frame)

            key = cv2.waitKey(100)
            if key & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


def main():
    print("=" * 70)
    print("🎯 TEST TEMPS RÉEL - PSPI CALIBRÉ")
    print("=" * 70)

    # Check if model exists
    if not Path('pspi_calibrated_model.pkl').exists():
        print("\n❌ Modèle calibré non trouvé!")
        print("   → Exécutez: python train_calibrated_pspi_model.py")
        return

    detector = CalibratedPSPIDetector('pspi_calibrated_model.pkl')

    print("\nChoisissez une option:")
    print("1. Webcam (Recommandé pour tester visage neutre)")
    print("2. Vidéo")
    print("3. Dossier d'images")

    choice = input("\nVotre choix (1/2/3): ").strip()

    if choice == '1':
        detector.run_webcam()
    elif choice == '2':
        path = input("Chemin vidéo: ").strip().strip('"')
        detector.run_video(path)
    elif choice == '3':
        path = input("Chemin dossier: ").strip().strip('"')
        detector.run_folder(path)
    else:
        print("❌ Choix invalide")


if __name__ == "__main__":
    main()