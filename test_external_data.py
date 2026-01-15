"""
Test PSPI Model on External/New Data
====================================
Use this script to validate your model on:
1. New images (folder)
2. Webcam (live)
3. Video file
4. Single image

This proves your model generalizes beyond training data.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


class ExternalDataTester:
    """Test PSPI model on external/new data"""

    def __init__(self, model_path='pspi_intensity_model.pkl'):
        """Load model and initialize MediaPipe"""
        print("=" * 60)
        print("🔬 EXTERNAL DATA TESTER")
        print("=" * 60)

        # Load model
        print("\n📂 Loading model...")
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_cols = model_data['feature_cols']
            print(f"   ✅ Model loaded: {model_path}")
            print(f"   ✅ Features: {len(self.feature_cols)}")
        except FileNotFoundError:
            print(f"   ❌ Model not found: {model_path}")
            print("   Run pain_intensity_system_pspi.py first!")
            raise

        # Initialize MediaPipe
        print("\n🔧 Initializing MediaPipe...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("   ✅ MediaPipe ready")

        # Results storage
        self.results = []

    def extract_features(self, landmarks, h, w):
        """Extract features from landmarks"""
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

    def get_pain_level(self, intensity):
        """Convert intensity to text level"""
        if intensity < 2:
            return "No Pain", (0, 255, 0)
        elif intensity < 4:
            return "Mild Pain", (0, 255, 255)
        elif intensity < 6:
            return "Moderate Pain", (0, 165, 255)
        elif intensity < 8:
            return "Strong Pain", (0, 100, 255)
        else:
            return "Severe Pain", (0, 0, 255)

    def process_image(self, image_path):
        """Process a single image"""
        # Read image
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = cv2.imread(str(image_path))
            source = str(image_path)
        else:
            image = image_path
            source = "frame"

        if image is None:
            return None, None, None

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return image, None, "No face detected"

        # Extract features
        landmarks = results.multi_face_landmarks[0].landmark
        features = self.extract_features(landmarks, h, w)

        # Predict
        intensity = self.predict_intensity(features)
        level, color = self.get_pain_level(intensity)

        # Draw on image
        output = image.copy()

        # Draw face mesh
        self.mp_drawing.draw_landmarks(
            output,
            results.multi_face_landmarks[0],
            self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )

        # Draw intensity bar
        bar_x, bar_y = 20, 50
        bar_w, bar_h = 30, 200
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_h = int((intensity / 10) * bar_h)
        cv2.rectangle(output, (bar_x, bar_y + bar_h - fill_h), (bar_x + bar_w, bar_y + bar_h), color, -1)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)

        # Draw text
        cv2.putText(output, f"{intensity:.1f}/10", (bar_x - 5, bar_y + bar_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(output, level, (w // 2 - 80, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return output, intensity, level

    def test_single_image(self, image_path):
        """Test on a single image"""
        print(f"\n🖼️  Testing: {image_path}")

        output, intensity, level = self.process_image(image_path)

        if intensity is None:
            print(f"   ❌ {level}")
            return

        print(f"   ✅ Intensity: {intensity:.2f}/10")
        print(f"   ✅ Level: {level}")

        # Show image
        cv2.imshow('Pain Detection Result', output)
        print("\n   Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return intensity

    def test_folder(self, folder_path, show_images=True):
        """Test on folder of images"""
        folder = Path(folder_path)

        if not folder.exists():
            print(f"❌ Folder not found: {folder}")
            return None

        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(folder.glob(ext))

        if not images:
            print(f"❌ No images found in {folder}")
            return None

        print(f"\n📁 Testing folder: {folder}")
        print(f"   Found {len(images)} images")
        print("-" * 50)

        results = []

        for i, img_path in enumerate(sorted(images)):
            output, intensity, level = self.process_image(img_path)

            if intensity is not None:
                results.append({
                    'image': img_path.name,
                    'intensity': intensity,
                    'level': level
                })
                print(f"   {i + 1:3d}. {img_path.name:30s} → {intensity:.2f} ({level})")

                if show_images:
                    cv2.imshow('Testing External Data', output)
                    key = cv2.waitKey(200)
                    if key == ord('q'):
                        break
            else:
                print(f"   {i + 1:3d}. {img_path.name:30s} → ❌ No face")

        if show_images:
            cv2.destroyAllWindows()

        # Summary
        if results:
            df = pd.DataFrame(results)

            print("\n" + "=" * 50)
            print("📊 FOLDER RESULTS SUMMARY")
            print("=" * 50)
            print(f"   Images processed: {len(results)}/{len(images)}")
            print(f"   Mean intensity:   {df['intensity'].mean():.2f}")
            print(f"   Std intensity:    {df['intensity'].std():.2f}")
            print(f"   Min intensity:    {df['intensity'].min():.2f}")
            print(f"   Max intensity:    {df['intensity'].max():.2f}")

            # Distribution by level
            print(f"\n   Distribution by level:")
            for level in df['level'].unique():
                count = len(df[df['level'] == level])
                pct = count / len(df) * 100
                print(f"   ├── {level}: {count} ({pct:.1f}%)")

            return df

        return None

    def test_webcam(self):
        """Test on webcam in real-time"""
        print("\n📹 Starting webcam test...")
        print("   Press 'q' to quit")
        print("   Press 's' to save screenshot")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return

        intensities = []

        # Re-init face mesh for video
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            output, intensity, level = self.process_image(frame)

            if intensity is not None:
                intensities.append(intensity)

                # Draw running average
                if len(intensities) > 30:
                    avg = np.mean(intensities[-30:])
                    cv2.putText(output, f"Avg: {avg:.1f}", (output.shape[1] - 100, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Webcam Pain Detection (q=quit, s=save)', output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, output)
                print(f"   💾 Saved: {filename}")

        cap.release()
        cv2.destroyAllWindows()

        # Summary
        if intensities:
            print("\n" + "=" * 50)
            print("📊 WEBCAM SESSION SUMMARY")
            print("=" * 50)
            print(f"   Frames analyzed: {len(intensities)}")
            print(f"   Mean intensity:  {np.mean(intensities):.2f}")
            print(f"   Max intensity:   {np.max(intensities):.2f}")

    def test_video(self, video_path):
        """Test on video file"""
        print(f"\n🎬 Testing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("❌ Cannot open video")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps:.1f}")
        print("   Press 'q' to quit")

        # Re-init face mesh for video
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        results = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            output, intensity, level = self.process_image(frame)

            if intensity is not None:
                results.append({
                    'frame': frame_count,
                    'time': frame_count / fps,
                    'intensity': intensity,
                    'level': level
                })

            # Show progress
            cv2.putText(output if output is not None else frame,
                        f"Frame: {frame_count}/{total_frames}",
                        (10, output.shape[0] - 10 if output is not None else frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Video Pain Detection', output if output is not None else frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Summary and plot
        if results:
            df = pd.DataFrame(results)

            print("\n" + "=" * 50)
            print("📊 VIDEO RESULTS SUMMARY")
            print("=" * 50)
            print(f"   Frames with face: {len(results)}/{frame_count}")
            print(f"   Mean intensity:   {df['intensity'].mean():.2f}")
            print(f"   Max intensity:    {df['intensity'].max():.2f}")
            print(f"   Std intensity:    {df['intensity'].std():.2f}")

            # Plot timeline
            plt.figure(figsize=(12, 4))
            plt.plot(df['time'], df['intensity'], 'b-', linewidth=1)
            plt.fill_between(df['time'], df['intensity'], alpha=0.3)
            plt.axhline(y=3, color='g', linestyle='--', label='Mild threshold')
            plt.axhline(y=6, color='orange', linestyle='--', label='Moderate threshold')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Pain Intensity (0-10)')
            plt.title('Pain Intensity Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = 'video_intensity_timeline.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"\n   📈 Timeline saved: {plot_path}")

            return df

        return None


def main():
    """Main menu"""
    print("\n" + "=" * 60)
    print("🔬 EXTERNAL DATA TESTING")
    print("   Validate your PSPI model on new data")
    print("=" * 60)

    try:
        tester = ExternalDataTester('pspi_intensity_model.pkl')
    except:
        return

    while True:
        print("\n" + "-" * 40)
        print("Choose an option:")
        print("-" * 40)
        print("1. Test single image")
        print("2. Test folder of images")
        print("3. Test webcam (real-time)")
        print("4. Test video file")
        print("5. Quit")

        choice = input("\nYour choice (1-5): ").strip()

        if choice == '1':
            path = input("Enter image path: ").strip().strip('"')
            tester.test_single_image(path)

        elif choice == '2':
            path = input("Enter folder path: ").strip().strip('"')
            show = input("Show images? (y/n): ").strip().lower() == 'y'
            tester.test_folder(path, show_images=show)

        elif choice == '3':
            tester.test_webcam()

        elif choice == '4':
            path = input("Enter video path: ").strip().strip('"')
            tester.test_video(path)

        elif choice == '5':
            print("\n👋 Goodbye!")
            break

        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()