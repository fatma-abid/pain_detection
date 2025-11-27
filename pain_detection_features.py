"""
Extraction des Caractéristiques Faciales pour la Détection de Douleur
Utilise MediaPipe pour extraire 468 points de repère faciaux
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class FacialFeatureExtractor:
    """
    Extracteur de caractéristiques faciales pour la détection de douleur
    Utilise MediaPipe pour extraire 468 points de repère faciaux
    """
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Indices des régions importantes pour la douleur
        self.pain_regions = {
            'front': list(range(70, 109)),
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'left_eye': [33, 160, 158, 133, 153, 144, 163, 7],
            'right_eye': [362, 385, 387, 263, 373, 380, 374, 249],
            'nose': [1, 2, 98, 327],
            'mouth': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
            'cheeks': [50, 205, 425, 280]
        }
    
    def extract_landmarks(self, image_path):
        """
        Extrait les points de repère faciaux d'une image
        """
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
            landmarks.append([
                landmark.x * w,
                landmark.y * h,
                landmark.z * w
            ])
        
        return np.array(landmarks), image
    
    def calculate_distances(self, landmarks):
        """
        Calcule les distances entre points clés
        """
        if landmarks is None:
            return None
        
        features = {}
        
        # Distance entre les sourcils
        left_brow = landmarks[70]
        right_brow = landmarks[300]
        features['eyebrow_distance'] = np.linalg.norm(left_brow - right_brow)
        
        # Ouverture des yeux
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        features['left_eye_opening'] = np.linalg.norm(left_eye_top - left_eye_bottom)
        
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        features['right_eye_opening'] = np.linalg.norm(right_eye_top - right_eye_bottom)
        
        # Ouverture de la bouche
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        features['mouth_opening'] = np.linalg.norm(upper_lip - lower_lip)
        
        # Largeur de la bouche
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        features['mouth_width'] = np.linalg.norm(left_mouth - right_mouth)
        
        # Élévation des joues
        left_cheek = landmarks[205]
        right_cheek = landmarks[425]
        nose_bridge = landmarks[6]
        features['left_cheek_elevation'] = nose_bridge[1] - left_cheek[1]
        features['right_cheek_elevation'] = nose_bridge[1] - right_cheek[1]
        
        return features
    
    def extract_geometric_features(self, landmarks):
        """
        Extrait des caractéristiques géométriques avancées
        """
        if landmarks is None:
            return None
        
        features = {}
        
        # Angle du sourcil gauche
        p1, p2, p3 = landmarks[70], landmarks[63], landmarks[105]
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        features['left_eyebrow_angle'] = np.degrees(angle)
        
        # Angle du sourcil droit
        p1, p2, p3 = landmarks[300], landmarks[293], landmarks[334]
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        features['right_eyebrow_angle'] = np.degrees(angle)
        
        # Ratio d'aspect du visage
        face_width = np.linalg.norm(landmarks[234] - landmarks[454])
        face_height = np.linalg.norm(landmarks[10] - landmarks[152])
        features['face_aspect_ratio'] = face_width / (face_height + 1e-6)
        
        return features
    
    def process_dataset(self, dataset_path, output_csv='pain_features_complete.csv', 
                       include_raw_landmarks=False, frame_type='Colour frames'):
        """
        Traite toutes les images du dataset
        """
        dataset_path = Path(dataset_path)
        all_features = []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Mapping des labels
        pain_labels = {
            'Neutral': 0,
            'Posed Pain': 1,
            'Algometer Pain': 2,
            'Laser Pain': 3
        }
        
        print("=" * 70)
        print("🚀 EXTRACTION DES CARACTÉRISTIQUES FACIALES")
        print("=" * 70)
        print(f"📁 Dataset: {dataset_path}")
        print(f"🎨 Type de frames: {frame_type}")
        print(f"💾 Fichier de sortie: {output_csv}")
        print(f"📊 Coordonnées brutes: {'Oui' if include_raw_landmarks else 'Non'}")
        print(f"⏰ Début: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 70)
        
        processed = 0
        failed = 0
        stats_by_pain = {label: 0 for label in pain_labels.keys()}
        
        subject_folders = sorted([d for d in dataset_path.iterdir() 
                                 if d.is_dir() and d.name.startswith('S')])
        
        total_subjects = len(subject_folders)
        
        for idx, subject_folder in enumerate(subject_folders, 1):
            subject_id = subject_folder.name
            subject_count = 0
            
            print(f"\n[{idx}/{total_subjects}] 📂 {subject_id}...", end=' ', flush=True)
            
            for pain_type_folder in subject_folder.iterdir():
                if not pain_type_folder.is_dir():
                    continue
                
                pain_type = pain_type_folder.name
                
                # Ignorer les dossiers anormaux
                if pain_type not in pain_labels:
                    continue
                
                frame_folder = pain_type_folder / frame_type
                
                if not frame_folder.exists():
                    continue
                
                for image_path in frame_folder.iterdir():
                    if image_path.suffix.lower() not in image_extensions:
                        continue
                    
                    landmarks, image = self.extract_landmarks(image_path)
                    
                    if landmarks is None:
                        failed += 1
                        continue
                    
                    distance_features = self.calculate_distances(landmarks)
                    geometric_features = self.extract_geometric_features(landmarks)
                    
                    features = {
                        'subject_id': subject_id,
                        'pain_type': pain_type,
                        'pain_label': pain_labels[pain_type],
                        'frame_type': frame_type,
                        'image_name': image_path.name,
                        'image_path': str(image_path),
                        **distance_features,
                        **geometric_features
                    }
                    
                    if include_raw_landmarks:
                        for i, (x, y, z) in enumerate(landmarks):
                            features[f'landmark_{i}_x'] = x
                            features[f'landmark_{i}_y'] = y
                            features[f'landmark_{i}_z'] = z
                    
                    all_features.append(features)
                    processed += 1
                    subject_count += 1
                    stats_by_pain[pain_type] += 1
            
            print(f"✅ {subject_count} images")
        
        if not all_features:
            print("\n❌ ERREUR: Aucune image traitée!")
            return None
        
        df = pd.DataFrame(all_features)
        df.to_csv(output_csv, index=False)
        
        print("\n" + "=" * 70)
        print("✅ EXTRACTION TERMINÉE!")
        print("=" * 70)
        print(f"⏰ Fin: {datetime.now().strftime('%H:%M:%S')}")
        print(f"\n📊 STATISTIQUES:")
        print(f"   • Images traitées: {processed}")
        print(f"   • Images échouées: {failed}")
        print(f"   • Taux de réussite: {processed/(processed+failed)*100:.1f}%")
        print(f"   • Fichier: {output_csv}")
        print(f"   • Nombre de features: {len(df.columns)}")
        print(f"   • Dimensions: {df.shape}")
        
        print(f"\n📈 DISTRIBUTION PAR TYPE DE DOULEUR:")
        for pain_type, count in sorted(stats_by_pain.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"   • {pain_type:20s}: {count:4d} images")
        
        print("=" * 70)
        
        return df
    
    def visualize_sample(self, df, output_dir='visualizations'):
        """
        Visualise un échantillon d'images avec leurs features
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\n🎨 Création de visualisations...")
        
        for pain_type in df['pain_type'].unique():
            sample = df[df['pain_type'] == pain_type].iloc[0]
            image_path = sample['image_path']
            
            landmarks, image = self.extract_landmarks(image_path)
            
            if landmarks is not None:
                for i, (x, y, z) in enumerate(landmarks):
                    cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
                
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(f"{pain_type} - {sample['subject_id']}")
                plt.axis('off')
                
                output_path = Path(output_dir) / f"sample_{pain_type.replace(' ', '_')}.png"
                plt.savefig(output_path, bbox_inches='tight', dpi=100)
                plt.close()
        
        print(f"   ✅ Visualisations sauvegardées dans: {output_dir}/")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎯 DÉTECTION AUTOMATIQUE DE LA DOULEUR")
    print("   Étape 1: Extraction des Caractéristiques Faciales")
    print("=" * 70)
    
    # Configuration
    dataset_path = r"C:\Users\MSI\Desktop\pain_dataset\Pictures\Modified"
    frame_type_choice = 'Colour frames'  # Le plus recommandé
    
    # Initialisation
    extractor = FacialFeatureExtractor()
    
    # Extraction
    df = extractor.process_dataset(
        dataset_path=dataset_path,
        output_csv='pain_features_complete.csv',
        include_raw_landmarks=False,  # False = ~20 features, True = ~1400
        frame_type=frame_type_choice
    )
    
    if df is not None:
        print("\n" + "=" * 70)
        print("📊 APERÇU DES DONNÉES")
        print("=" * 70)
        
        print("\n🔍 Premières lignes:")
        print(df[['subject_id', 'pain_type', 'pain_label', 
                  'eyebrow_distance', 'mouth_opening']].head(10))
        
        print("\n📈 Statistiques descriptives:")
        numeric_cols = ['eyebrow_distance', 'left_eye_opening', 
                       'right_eye_opening', 'mouth_opening', 'mouth_width']
        print(df[numeric_cols].describe())
        
        print("\n🏷️  Distribution des labels:")
        print(df['pain_type'].value_counts())
        
        # Visualisations
        extractor.visualize_sample(df)
        
        print("\n" + "=" * 70)
        print("✅ PROCESSUS TERMINÉ AVEC SUCCÈS!")
        print("=" * 70)
        print("\n📁 Fichiers générés:")
        print("   • pain_features_complete.csv")
        print("   • visualizations/ (exemples d'images)")
        print("\n🚀 Prochaine étape: Entraînement du modèle ML")
        print("=" * 70)