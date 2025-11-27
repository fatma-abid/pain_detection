"""
Ré-calibration du Modèle avec Seuils Ajustés
Neutre = ~0-1, Douleur sévère = 8-10
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_intensity_calibrated(row):
    """
    Formule CALIBRÉE - Neutre ≈ 0-1
    """
    intensity = 0.0
    
    # 1. BOUCHE (max 4 points)
    mouth = row['mouth_opening']
    if mouth > 120:
        intensity += 4.0
    elif mouth > 80:
        intensity += 3.0
    elif mouth > 50:
        intensity += 2.0
    elif mouth > 30:
        intensity += 1.0
    
    # 2. YEUX (max 3 points)
    eye_avg = (row['left_eye_opening'] + row['right_eye_opening']) / 2
    if eye_avg < 8:
        intensity += 3.0
    elif eye_avg < 12:
        intensity += 2.0
    elif eye_avg < 16:
        intensity += 1.0
    
    # 3. SOURCILS (max 2 points)
    eyebrow = row['eyebrow_distance']
    if eyebrow < 340:
        intensity += 2.0
    elif eyebrow < 355:
        intensity += 1.0
    
    # 4. LARGEUR BOUCHE (max 1 point)
    if row['mouth_width'] > 220:
        intensity += 1.0
    
    return min(intensity, 10.0)


def recalibrate():
    print("=" * 60)
    print("🔧 RE-CALIBRATION DU MODÈLE")
    print("=" * 60)
    
    # Charger dataset original
    df = pd.read_csv('pain_features_complete.csv')
    print(f"✅ Dataset chargé: {len(df)} frames")
    
    # Re-calculer intensités avec nouvelle formule
    print("\n🔄 Re-calcul des intensités...")
    df['intensity'] = df.apply(calculate_intensity_calibrated, axis=1)
    
    # Stats
    print(f"\n📊 Nouvelles statistiques:")
    print(f"   Moyenne: {df['intensity'].mean():.2f}")
    print(f"   Min:     {df['intensity'].min():.1f}")
    print(f"   Max:     {df['intensity'].max():.1f}")
    
    # Stats par type de douleur
    print(f"\n📈 Par type de douleur:")
    for pain_type in df['pain_type'].unique():
        mean_int = df[df['pain_type'] == pain_type]['intensity'].mean()
        print(f"   {pain_type:20s}: {mean_int:.2f}")
    
    # Sauvegarder dataset annoté
    df.to_csv('pain_dataset_calibrated.csv', index=False)
    print(f"\n✅ Sauvegardé: pain_dataset_calibrated.csv")
    
    # Ré-entraîner modèle
    print("\n" + "=" * 60)
    print("🤖 RÉ-ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60)
    
    feature_cols = ['eyebrow_distance', 'left_eye_opening', 'right_eye_opening',
                   'mouth_opening', 'mouth_width', 'left_cheek_elevation',
                   'right_cheek_elevation', 'left_eyebrow_angle', 
                   'right_eyebrow_angle', 'face_aspect_ratio']
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].values
    y = df['intensity'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
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
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 MÉTRIQUES:")
    print(f"   MSE:  {mse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")
    
    # Sauvegarder modèle
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }
    joblib.dump(model_data, 'intensity_model.pkl')
    
    print(f"\n✅ Modèle sauvegardé: intensity_model.pkl")
    
    print("\n" + "=" * 60)
    print("✅ RE-CALIBRATION TERMINÉE!")
    print("=" * 60)
    print("""
    Maintenant quand vous testez avec webcam:
    
    😐 Visage neutre  → 0-1
    😣 Douleur légère → 2-4  
    😖 Douleur modérée → 4-6
    😫 Douleur sévère → 6-10
    """)


if __name__ == "__main__":
    recalibrate()