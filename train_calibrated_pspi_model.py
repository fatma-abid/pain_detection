"""
Train Model with Calibrated PSPI Thresholds
Uses pain_dataset_pspi_calibrated.csv
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def train_calibrated_model():
    """
    Train model using calibrated PSPI annotations
    """
    print("=" * 70)
    print("🤖 ENTRAÎNEMENT AVEC SEUILS CALIBRÉS")
    print("=" * 70)

    # Load calibrated dataset
    df = pd.read_csv('pain_dataset_pspi_amplified.csv')
    print(f"\n✅ Dataset calibré chargé: {len(df)} frames")

    # Load calibrated thresholds
    with open('pspi_calibrated_thresholds.json', 'r') as f:
        thresholds = json.load(f)

    print(f"✅ Seuils calibrés chargés")

    # Verify neutral intensity is low
    neutral_mean = df[df['pain_type'] == 'Neutral']['intensity'].mean()
    pain_mean = df[df['pain_type'] != 'Neutral']['intensity'].mean()

    print(f"\n🔍 Vérification Pré-entraînement:")
    print(f"   Intensité Neutre:  {neutral_mean:.2f}")
    print(f"   Intensité Douleur: {pain_mean:.2f}")

    if neutral_mean > 1.5:
        print(f"\n   ⚠️  ATTENTION: Intensité neutre encore élevée ({neutral_mean:.2f})")
        print(f"   → Considérez d'ajuster les seuils manuellement")
    else:
        print(f"   ✅ Calibration valide!")

    # Features
    feature_cols = [
        'eyebrow_distance', 'left_eye_opening', 'right_eye_opening',
        'mouth_opening', 'mouth_width', 'left_cheek_elevation',
        'right_cheek_elevation', 'left_eyebrow_angle',
        'right_eyebrow_angle', 'face_aspect_ratio'
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df['intensity'].values

    print(f"\n📊 Configuration:")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples:  {len(X)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['pain_type']
    )

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n🔄 Entraînement Gradient Boosting...")

    # Train
    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 10)

    # Metrics
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

    # Test on each pain type
    print(f"\n📈 Prédictions Moyennes par Type de Douleur:")
    print("-" * 70)

    df_test = df.iloc[X_test.shape[0]:].copy()  # Get test indices

    for pain_type in df['pain_type'].unique():
        mask = df['pain_type'] == pain_type
        if mask.sum() == 0:
            continue

        X_pain = df[mask][feature_cols].values
        X_pain_scaled = scaler.transform(X_pain)
        y_pred_pain = model.predict(X_pain_scaled)
        y_pred_pain = np.clip(y_pred_pain, 0, 10)

        actual_mean = df[mask]['intensity'].mean()
        pred_mean = y_pred_pain.mean()

        print(f"   {pain_type:20s}: Réel={actual_mean:.2f}, Prédit={pred_mean:.2f}")

    # Feature importance
    print(f"\n📊 Importance des Features:")
    importance = sorted(zip(feature_cols, model.feature_importances_),
                        key=lambda x: x[1], reverse=True)
    for feat, imp in importance:
        bar = '█' * int(imp * 50)
        print(f"   {feat:25s}: {imp:.4f} {bar}")

    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'thresholds': thresholds,
        'metrics': {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        },
        'annotation_method': 'PSPI_Calibrated'
    }

    joblib.dump(model_data, 'pspi_calibrated_model.pkl')
    print(f"\n💾 Modèle sauvegardé: pspi_calibrated_model.pkl")

    # Visualizations
    create_evaluation_plots(y_test, y_pred, df, feature_cols, model.feature_importances_)

    return model, scaler, thresholds


def create_evaluation_plots(y_test, y_pred, df, feature_cols, importances):
    """
    Create comprehensive evaluation plots
    """
    print(f"\n📊 Création des visualisations...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Prediction vs Actual
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, y_pred, alpha=0.5, c='blue', s=20)
    ax1.plot([0, 10], [0, 10], 'r--', linewidth=2, label='Parfait')
    ax1.set_xlabel('Intensité Réelle')
    ax1.set_ylabel('Intensité Prédite')
    ax1.set_title(f'Prédiction vs Réel')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_pred - y_test
    ax2.scatter(y_pred, residuals, alpha=0.5, c='coral', s=20)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Intensité Prédite')
    ax2.set_ylabel('Résidus (Prédit - Réel)')
    ax2.set_title('Analyse des Résidus')
    ax2.grid(True, alpha=0.3)

    # 3. Error distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Erreur (Prédit - Réel)')
    ax3.set_ylabel('Fréquence')
    ax3.set_title(f'Distribution des Erreurs\nMAE={mean_absolute_error(y_test, y_pred):.3f}')
    ax3.grid(True, alpha=0.3)

    # 4. Feature importance
    ax4 = fig.add_subplot(gs[1, :])
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=True)

    colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())
    ax4.barh(importance_df['feature'], importance_df['importance'], color=colors)
    ax4.set_xlabel('Importance')
    ax4.set_title('Importance des Features')
    ax4.grid(True, alpha=0.3, axis='x')

    # 5. Intensity distribution by pain type
    ax5 = fig.add_subplot(gs[2, 0])
    pain_order = ['Neutral', 'Laser Pain', 'Algometer Pain', 'Posed Pain']
    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity']
            ax5.hist(subset, alpha=0.6, label=pain_type, bins=20)
    ax5.set_xlabel('Intensité')
    ax5.set_ylabel('Fréquence')
    ax5.set_title('Distribution par Type de Douleur')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Box plot by pain type
    ax6 = fig.add_subplot(gs[2, 1])
    pain_data = [df[df['pain_type'] == pt]['intensity'].values
                 for pt in pain_order if pt in df['pain_type'].values]
    pain_labels = [pt for pt in pain_order if pt in df['pain_type'].values]
    bp = ax6.boxplot(pain_data, labels=pain_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['green', 'yellow', 'orange', 'red']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax6.set_ylabel('Intensité')
    ax6.set_title('Box Plot par Type de Douleur')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 7. Mean intensity comparison
    ax7 = fig.add_subplot(gs[2, 2])
    means = df.groupby('pain_type')['intensity'].mean().sort_values()
    stds = df.groupby('pain_type')['intensity'].std()
    colors_map = {'Neutral': 'green', 'Laser Pain': 'yellow',
                  'Algometer Pain': 'orange', 'Posed Pain': 'red'}
    bar_colors = [colors_map.get(pt, 'gray') for pt in means.index]
    means.plot(kind='barh', ax=ax7, xerr=stds[means.index],
               capsize=5, color=bar_colors, alpha=0.7)
    ax7.set_xlabel('Intensité Moyenne')
    ax7.set_title('Intensité Moyenne ± SD')
    ax7.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Évaluation du Modèle PSPI Calibré', fontsize=16, fontweight='bold')

    plt.savefig('calibrated_model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✅ calibrated_model_evaluation.png")


def main():
    print("\n" + "=" * 70)
    print("🎯 ENTRAÎNEMENT AVEC PSPI CALIBRÉ")
    print("=" * 70)

    # Check if calibrated dataset exists
    if not Path('pain_dataset_pspi_calibrated.csv').exists():
        print("\n❌ ERREUR: Dataset calibré non trouvé!")
        print("   → Exécutez d'abord: python calibrate_pspi_thresholds.py")
        return

    # Train model
    model, scaler, thresholds = train_calibrated_model()

    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("=" * 70)
    print("\n📁 Fichiers générés:")
    print("   • pspi_calibrated_model.pkl (modèle entraîné)")
    print("   • calibrated_model_evaluation.png (visualisations)")
    print("\n🚀 Prochaine étape: Tester en temps réel!")
    print("   → python test_calibrated_realtime.py")
    print("=" * 70)


if __name__ == "__main__":
    main()