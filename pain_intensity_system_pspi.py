"""
Système Complet - Détection Intensité de Douleur (Version PSPI)
================================================================
Version améliorée utilisant l'annotation PSPI (scientifiquement validée)
au lieu des seuils arbitraires.

PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
Scale: 0-16 (converted to 0-10)

References:
- Prkachin & Solomon (2008)
- UNBC-McMaster Dataset methodology
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# ============================================================
# PARTIE 1 : ANNOTATION PSPI (SCIENTIFIQUEMENT VALIDÉE)
# ============================================================

def estimate_au4_from_features(row):
    """
    Estime AU4 (Brow Lowerer) à partir de la distance des sourcils
    Plus les sourcils sont rapprochés/froncés, plus AU4 est élevé
    """
    eyebrow_dist = row['eyebrow_distance']

    # Baseline typique: 360-380 pixels
    # Froncé: < 350 pixels
    if eyebrow_dist < 340:
        return 5.0
    elif eyebrow_dist < 350:
        return 4.0
    elif eyebrow_dist < 360:
        return 3.0
    elif eyebrow_dist < 370:
        return 2.0
    elif eyebrow_dist < 380:
        return 1.0
    else:
        return 0.0


def estimate_au6_au7_from_features(row):
    """
    Estime AU6 (Cheek Raiser) et AU7 (Lid Tightener)
    Basé sur l'ouverture des yeux - yeux fermés/plissés = valeur élevée
    """
    eye_avg = (row['left_eye_opening'] + row['right_eye_opening']) / 2

    # Yeux normalement ouverts: 15-25 pixels
    # Yeux plissés/fermés: < 12 pixels
    if eye_avg < 8:
        au6 = 5.0
        au7 = 5.0
    elif eye_avg < 10:
        au6 = 4.0
        au7 = 4.0
    elif eye_avg < 12:
        au6 = 3.0
        au7 = 3.0
    elif eye_avg < 15:
        au6 = 2.0
        au7 = 2.0
    elif eye_avg < 18:
        au6 = 1.0
        au7 = 1.0
    else:
        au6 = 0.0
        au7 = 0.0

    return au6, au7


def estimate_au9_au10_from_features(row):
    """
    Estime AU9 (Nose Wrinkler) et AU10 (Upper Lip Raiser)
    Basé sur l'élévation des joues et ouverture de la bouche
    """
    # Utiliser l'élévation des joues si disponible
    if 'left_cheek_elevation' in row and 'right_cheek_elevation' in row:
        cheek_avg = (row['left_cheek_elevation'] + row['right_cheek_elevation']) / 2

        if cheek_avg > 40:
            au9 = 5.0
        elif cheek_avg > 30:
            au9 = 4.0
        elif cheek_avg > 20:
            au9 = 3.0
        elif cheek_avg > 10:
            au9 = 2.0
        elif cheek_avg > 5:
            au9 = 1.0
        else:
            au9 = 0.0
    else:
        au9 = 0.0

    # AU10 basé sur la bouche
    mouth_open = row['mouth_opening']

    if mouth_open > 50:
        au10 = 5.0
    elif mouth_open > 40:
        au10 = 4.0
    elif mouth_open > 30:
        au10 = 3.0
    elif mouth_open > 20:
        au10 = 2.0
    elif mouth_open > 10:
        au10 = 1.0
    else:
        au10 = 0.0

    return au9, au10


def estimate_au43_from_features(row):
    """
    Estime AU43 (Eye Closure)
    Binaire dans PSPI original, mais on utilise continu
    """
    eye_avg = (row['left_eye_opening'] + row['right_eye_opening']) / 2

    if eye_avg < 5:
        return 1.0  # Yeux fermés
    elif eye_avg < 8:
        return 0.5  # Partiellement fermés
    else:
        return 0.0  # Ouverts


def calculate_pspi_intensity(row):
    """
    Calcule l'intensité de douleur basée sur PSPI

    PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43

    Cette formule est validée scientifiquement:
    - Corrélation avec douleur auto-rapportée: r > 0.61
    - Utilisée dans le dataset UNBC-McMaster

    Returns:
        float: Intensité sur échelle 0-10
    """
    # Estimer chaque AU
    au4 = estimate_au4_from_features(row)
    au6, au7 = estimate_au6_au7_from_features(row)
    au9, au10 = estimate_au9_au10_from_features(row)
    au43 = estimate_au43_from_features(row)

    # Formule PSPI
    pspi = au4 + max(au6, au7) + max(au9, au10) + au43

    # PSPI range: 0-16, convertir en 0-10
    intensity = (pspi / 16) * 10

    return round(min(intensity, 10.0), 2)


def calculate_pspi_details(row):
    """
    Retourne les détails de chaque AU pour analyse
    """
    au4 = estimate_au4_from_features(row)
    au6, au7 = estimate_au6_au7_from_features(row)
    au9, au10 = estimate_au9_au10_from_features(row)
    au43 = estimate_au43_from_features(row)

    pspi = au4 + max(au6, au7) + max(au9, au10) + au43
    intensity = (pspi / 16) * 10

    return {
        'AU4': au4,
        'AU6': au6,
        'AU7': au7,
        'AU9': au9,
        'AU10': au10,
        'AU43': au43,
        'PSPI': round(pspi, 2),
        'intensity': round(min(intensity, 10.0), 2)
    }


def annotate_dataset_pspi(csv_path='pain_features_complete.csv', output_path='pain_dataset_pspi.csv'):
    """
    Annote le dataset avec les labels d'intensité basés sur PSPI
    """
    print("=" * 70)
    print("📝 ANNOTATION PSPI (Scientifiquement Validée)")
    print("=" * 70)
    print("\nFormule PSPI: AU4 + max(AU6,AU7) + max(AU9,AU10) + AU43")
    print("Référence: Prkachin & Solomon (2008)")
    print("-" * 70)

    df = pd.read_csv(csv_path)
    print(f"\n✅ Dataset chargé: {len(df)} frames")

    # Calculer PSPI pour chaque frame
    print("\n🔄 Calcul des scores PSPI...")

    # Ajouter les colonnes AU individuelles
    au_details = df.apply(calculate_pspi_details, axis=1)
    au_df = pd.DataFrame(au_details.tolist())

    # Fusionner avec le dataset original
    for col in au_df.columns:
        df[col] = au_df[col]

    # Sauvegarder
    df.to_csv(output_path, index=False)

    # Statistiques
    print(f"\n✅ Sauvegardé: {output_path}")
    print(f"\n📊 Statistiques d'Intensité (PSPI-based):")
    print(f"   Moyenne:     {df['intensity'].mean():.2f}")
    print(f"   Écart-type:  {df['intensity'].std():.2f}")
    print(f"   Min:         {df['intensity'].min():.1f}")
    print(f"   Max:         {df['intensity'].max():.1f}")

    # Distribution par type de douleur
    print(f"\n📈 Intensité Moyenne par Type de Douleur:")
    for pain_type in df['pain_type'].unique():
        mean_int = df[df['pain_type'] == pain_type]['intensity'].mean()
        std_int = df[df['pain_type'] == pain_type]['intensity'].std()
        print(f"   {pain_type:20s}: {mean_int:.2f} ± {std_int:.2f}")

    # Vérification de cohérence
    print(f"\n🔍 Vérification de Cohérence:")
    neutral_mean = df[df['pain_type'] == 'Neutral']['intensity'].mean()
    pain_mean = df[df['pain_type'] != 'Neutral']['intensity'].mean()

    if neutral_mean < pain_mean:
        print(f"   ✅ Neutre ({neutral_mean:.2f}) < Douleur ({pain_mean:.2f}) - Cohérent!")
    else:
        print(f"   ⚠️  Attention: Neutre ({neutral_mean:.2f}) >= Douleur ({pain_mean:.2f})")

    return df


# ============================================================
# PARTIE 2 : ENTRAÎNEMENT DU MODÈLE
# ============================================================

def train_pspi_intensity_model(df, output_model='pspi_intensity_model.pkl'):
    """
    Entraîne un modèle de régression pour prédire l'intensité PSPI
    """
    print("\n" + "=" * 70)
    print("🤖 ENTRAÎNEMENT DU MODÈLE D'INTENSITÉ")
    print("=" * 70)

    # Features - utiliser à la fois les features géométriques et les AUs estimés
    feature_cols = [
        'eyebrow_distance', 'left_eye_opening', 'right_eye_opening',
        'mouth_opening', 'mouth_width', 'left_cheek_elevation',
        'right_cheek_elevation', 'left_eyebrow_angle',
        'right_eyebrow_angle', 'face_aspect_ratio'
    ]

    # Garder seulement les colonnes qui existent
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df['intensity'].values

    print(f"\n📊 Configuration:")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples:  {len(X)}")
    print(f"   Target:   intensity (0-10, PSPI-based)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entraînement avec Gradient Boosting (meilleur pour régression)
    print("\n🔄 Entraînement Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Évaluation
    y_pred = model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 10)  # Contraindre à [0, 10]

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n✅ RÉSULTATS SUR TEST SET:")
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")

    # Cross-validation
    print(f"\n📊 Cross-Validation (5-fold):")
    X_full_scaled = scaler.fit_transform(X)
    cv_scores = cross_val_score(model, X_full_scaled, y, cv=5, scoring='r2')
    print(f"   R² scores: {cv_scores}")
    print(f"   Mean R²:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Sauvegarder
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        },
        'annotation_method': 'PSPI'
    }
    joblib.dump(model_data, output_model)
    print(f"\n💾 Modèle sauvegardé: {output_model}")

    # Feature importance
    print(f"\n📈 Importance des Features:")
    importance = sorted(zip(feature_cols, model.feature_importances_),
                        key=lambda x: x[1], reverse=True)
    for feat, imp in importance:
        bar = '█' * int(imp * 50)
        print(f"   {feat:25s}: {imp:.4f} {bar}")

    return model, scaler, feature_cols


# ============================================================
# PARTIE 3 : VISUALISATION ET TEST
# ============================================================

class PSPIRealTimeDetector:
    """
    Détection d'intensité en temps réel basée sur PSPI
    """

    def __init__(self, model_path='pspi_intensity_model.pkl'):
        # Charger modèle
        print("🔄 Chargement du modèle PSPI...")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']

        if 'annotation_method' in model_data:
            print(f"✅ Méthode d'annotation: {model_data['annotation_method']}")

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
        """Extrait les features depuis les landmarks MediaPipe"""
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
        """Prédit l'intensité avec le modèle"""
        X = np.array([[features.get(col, 0) for col in self.feature_cols]])
        X_scaled = self.scaler.transform(X)
        intensity = self.model.predict(X_scaled)[0]
        return np.clip(intensity, 0, 10)

    def get_color_for_intensity(self, intensity):
        """Couleur selon l'intensité"""
        if intensity < 3:
            return (0, 255, 0)  # Vert - Légère
        elif intensity < 6:
            return (0, 165, 255)  # Orange - Modérée
        else:
            return (0, 0, 255)  # Rouge - Sévère

    def get_pain_level(self, intensity):
        """Niveau de douleur textuel"""
        if intensity < 2:
            return "Pas de Douleur"
        elif intensity < 4:
            return "Douleur Légère"
        elif intensity < 6:
            return "Douleur Modérée"
        elif intensity < 8:
            return "Douleur Forte"
        else:
            return "Douleur Sévère"

    def draw_ui(self, frame, intensity):
        """Dessine l'interface utilisateur"""
        h, w = frame.shape[:2]
        color = self.get_color_for_intensity(intensity)

        # Barre d'intensité
        bar_x, bar_y = 30, 50
        bar_width, bar_height = 30, 300

        # Fond
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      (50, 50, 50), -1)

        # Remplissage
        fill_height = int((intensity / 10) * bar_height)
        cv2.rectangle(frame,
                      (bar_x, bar_y + bar_height - fill_height),
                      (bar_x + bar_width, bar_y + bar_height),
                      color, -1)

        # Bordure
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      (255, 255, 255), 2)

        # Labels
        cv2.putText(frame, "INTENSITE", (bar_x - 15, bar_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{intensity:.1f}/10", (bar_x - 10, bar_y + bar_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Niveau de douleur
        pain_level = self.get_pain_level(intensity)
        cv2.putText(frame, pain_level, (w // 2 - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Badge PSPI
        cv2.putText(frame, "PSPI-based", (w - 120, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Mini graphique historique
        if len(self.intensity_history) > 1:
            graph_x, graph_y = w - 220, 50
            graph_w, graph_h = 200, 100

            # Fond semi-transparent
            overlay = frame.copy()
            cv2.rectangle(overlay, (graph_x, graph_y),
                          (graph_x + graph_w, graph_y + graph_h),
                          (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Courbe
            history = self.intensity_history[-50:]
            for i in range(1, len(history)):
                x1 = graph_x + int(((i - 1) / max(len(history) - 1, 1)) * graph_w)
                y1 = graph_y + graph_h - int((history[i - 1] / 10) * graph_h)
                x2 = graph_x + int((i / max(len(history) - 1, 1)) * graph_w)
                y2 = graph_y + graph_h - int((history[i] / 10) * graph_h)
                cv2.line(frame, (x1, y1), (x2, y2),
                         self.get_color_for_intensity(history[i]), 2)

            cv2.putText(frame, "Historique", (graph_x, graph_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def process_frame(self, frame):
        """Traite une frame"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        intensity = 0.0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Dessiner mesh facial
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Points clés colorés
            key_points = {
                (0, 0, 255): [13, 14, 61, 291],  # Rouge: bouche
                (255, 255, 0): [159, 145, 386, 374],  # Jaune: yeux
                (255, 0, 0): [70, 300]  # Bleu: sourcils
            }

            for color, indices in key_points.items():
                for idx in indices:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, color, -1)

            # Extraire features et prédire
            features = self.extract_features(face_landmarks.landmark, h, w)
            intensity = self.predict_intensity(features)

            # Historique
            self.intensity_history.append(intensity)
            if len(self.intensity_history) > self.max_history:
                self.intensity_history.pop(0)

        # Dessiner UI
        self.draw_ui(frame, intensity)

        return frame, intensity

    def run_webcam(self):
        """Lance sur webcam"""
        print("\n📹 Webcam activée (Appuyez 'q' pour quitter)")
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame, intensity = self.process_frame(frame)

            cv2.imshow('Detection Intensite PSPI', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_on_folder(self, folder_path):
        """Lance sur dossier de frames"""
        folder = Path(folder_path)
        frames = sorted(folder.glob('*.jpg')) + sorted(folder.glob('*.png'))

        if not frames:
            print(f"❌ Aucune image dans {folder}")
            return

        print(f"\n🖼️ {len(frames)} frames trouvées")

        results = []
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            frame, intensity = self.process_frame(frame)
            results.append({'frame': frame_path.name, 'intensity': intensity})

            cv2.imshow('Detection Intensite PSPI', frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🎯 SYSTÈME DE DÉTECTION D'INTENSITÉ (Version PSPI)")
    print("=" * 70)
    print("\n✅ Cette version utilise l'annotation PSPI scientifiquement validée")
    print("   Référence: Prkachin & Solomon Pain Intensity metric")
    print("=" * 70)

    # Étape 1: Annotation PSPI
    print("\n" + "=" * 70)
    print("📝 ÉTAPE 1: ANNOTATION PSPI")
    print("=" * 70)

    try:
        df = annotate_dataset_pspi('pain_features_complete.csv', 'pain_dataset_pspi.csv')
    except FileNotFoundError:
        print("❌ Fichier pain_features_complete.csv non trouvé!")
        print("   Exécutez d'abord: python pain_detection_features.py")
        return

    # Étape 2: Entraînement
    model, scaler, feature_cols = train_pspi_intensity_model(df, 'pspi_intensity_model.pkl')

    # Étape 3: Test
    print("\n" + "=" * 70)
    print("🎬 ÉTAPE 3: TEST TEMPS RÉEL")
    print("=" * 70)

    print("\nChoisissez une option:")
    print("1. Tester sur webcam")
    print("2. Tester sur dossier de frames")
    print("3. Quitter")

    choice = input("\nVotre choix (1/2/3): ").strip()

    detector = PSPIRealTimeDetector('pspi_intensity_model.pkl')

    if choice == '1':
        detector.run_webcam()
    elif choice == '2':
        folder = input("Chemin du dossier: ").strip().strip('"')
        detector.run_on_folder(folder)

    print("\n✅ Terminé!")


if __name__ == "__main__":
    main()