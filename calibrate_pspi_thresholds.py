"""
Calibrate PSPI Thresholds Based on Your Dataset
Ensures Neutral ≈ 0 and proper pain scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_feature_distributions():
    """
    Analyze feature distributions by pain type to find proper thresholds
    """
    print("=" * 70)
    print("📊 CALIBRATION DES SEUILS PSPI")
    print("=" * 70)

    df = pd.read_csv('pain_features_complete.csv')

    # Define pain severity order
    pain_order = ['Neutral', 'Laser Pain', 'Algometer Pain', 'Posed Pain']

    # Key features for each AU
    features_analysis = {
        'AU4 (Brow Lowerer)': 'eyebrow_distance',
        'AU6/7 (Eye Squeeze)': ['left_eye_opening', 'right_eye_opening'],
        'AU9/10 (Nose/Lip)': ['mouth_opening', 'left_cheek_elevation', 'right_cheek_elevation'],
        'AU43 (Eye Closure)': ['left_eye_opening', 'right_eye_opening']
    }

    print("\n🔍 Analyse des Distributions par Type de Douleur:\n")

    # Analyze each feature
    calibrated_thresholds = {}

    # 1. Eyebrow Distance (AU4)
    print("1️⃣  AU4 - Eyebrow Distance (Brow Lowerer)")
    print("-" * 70)

    for pain_type in pain_order:
        subset = df[df['pain_type'] == pain_type]['eyebrow_distance']
        mean = subset.mean()
        std = subset.std()
        median = subset.median()
        print(f"   {pain_type:20s}: mean={mean:.1f}, median={median:.1f}, std={std:.1f}")

    # Use percentiles from neutral as baseline
    neutral_eyebrow = df[df['pain_type'] == 'Neutral']['eyebrow_distance']
    neutral_mean = neutral_eyebrow.mean()
    neutral_std = neutral_eyebrow.std()

    calibrated_thresholds['eyebrow_distance'] = {
        'au4_0': neutral_mean + 0.5 * neutral_std,  # Very relaxed
        'au4_1': neutral_mean,  # Neutral
        'au4_2': neutral_mean - 0.5 * neutral_std,  # Slight furrow
        'au4_3': neutral_mean - 1.0 * neutral_std,  # Moderate
        'au4_4': neutral_mean - 1.5 * neutral_std,  # Strong
        'au4_5': neutral_mean - 2.0 * neutral_std  # Extreme
    }

    print(f"\n   ✅ Seuils Calibrés (Eyebrow Distance):")
    for key, val in calibrated_thresholds['eyebrow_distance'].items():
        print(f"      {key}: < {val:.1f} pixels")

    # 2. Eye Opening (AU6/7 + AU43)
    print("\n2️⃣  AU6/7 + AU43 - Eye Opening (Eye Squeeze/Closure)")
    print("-" * 70)

    df['eye_opening_avg'] = (df['left_eye_opening'] + df['right_eye_opening']) / 2

    for pain_type in pain_order:
        subset = df[df['pain_type'] == pain_type]['eye_opening_avg']
        mean = subset.mean()
        median = subset.median()
        std = subset.std()
        print(f"   {pain_type:20s}: mean={mean:.1f}, median={median:.1f}, std={std:.1f}")

    neutral_eye = df[df['pain_type'] == 'Neutral']['eye_opening_avg']
    neutral_mean_eye = neutral_eye.mean()
    neutral_std_eye = neutral_eye.std()

    calibrated_thresholds['eye_opening_avg'] = {
        'au67_0': neutral_mean_eye + 0.5 * neutral_std_eye,  # Wide open
        'au67_1': neutral_mean_eye,  # Normal
        'au67_2': neutral_mean_eye - 0.5 * neutral_std_eye,  # Slightly squeezed
        'au67_3': neutral_mean_eye - 1.0 * neutral_std_eye,  # Moderate squeeze
        'au67_4': neutral_mean_eye - 1.5 * neutral_std_eye,  # Strong squeeze
        'au67_5': neutral_mean_eye - 2.0 * neutral_std_eye  # Eyes closed
    }

    print(f"\n   ✅ Seuils Calibrés (Eye Opening):")
    for key, val in calibrated_thresholds['eye_opening_avg'].items():
        print(f"      {key}: < {val:.1f} pixels")

    # 3. Mouth Opening (AU9/10)
    print("\n3️⃣  AU9/10 - Mouth Opening (Nose Wrinkle/Upper Lip Raiser)")
    print("-" * 70)

    for pain_type in pain_order:
        subset = df[df['pain_type'] == pain_type]['mouth_opening']
        mean = subset.mean()
        median = subset.median()
        std = subset.std()
        print(f"   {pain_type:20s}: mean={mean:.1f}, median={median:.1f}, std={std:.1f}")

    neutral_mouth = df[df['pain_type'] == 'Neutral']['mouth_opening']
    neutral_mean_mouth = neutral_mouth.mean()
    neutral_std_mouth = neutral_mouth.std()

    calibrated_thresholds['mouth_opening'] = {
        'au910_0': neutral_mean_mouth + 0.5 * neutral_std_mouth,
        'au910_1': neutral_mean_mouth + 1.0 * neutral_std_mouth,
        'au910_2': neutral_mean_mouth + 1.5 * neutral_std_mouth,
        'au910_3': neutral_mean_mouth + 2.0 * neutral_std_mouth,
        'au910_4': neutral_mean_mouth + 2.5 * neutral_std_mouth,
        'au910_5': neutral_mean_mouth + 3.0 * neutral_std_mouth
    }

    print(f"\n   ✅ Seuils Calibrés (Mouth Opening):")
    for key, val in calibrated_thresholds['mouth_opening'].items():
        print(f"      {key}: > {val:.1f} pixels")

    return calibrated_thresholds, df


def calculate_pspi_calibrated(row, thresholds):
    """
    Calculate PSPI with calibrated thresholds
    """
    # AU4 - Brow Lowerer
    eyebrow = row['eyebrow_distance']
    if eyebrow >= thresholds['eyebrow_distance']['au4_0']:
        au4 = 0.0
    elif eyebrow >= thresholds['eyebrow_distance']['au4_1']:
        au4 = 1.0
    elif eyebrow >= thresholds['eyebrow_distance']['au4_2']:
        au4 = 2.0
    elif eyebrow >= thresholds['eyebrow_distance']['au4_3']:
        au4 = 3.0
    elif eyebrow >= thresholds['eyebrow_distance']['au4_4']:
        au4 = 4.0
    else:
        au4 = 5.0

    # AU6/7 - Eye Squeeze
    eye_avg = (row['left_eye_opening'] + row['right_eye_opening']) / 2
    if eye_avg >= thresholds['eye_opening_avg']['au67_0']:
        au6 = au7 = 0.0
    elif eye_avg >= thresholds['eye_opening_avg']['au67_1']:
        au6 = au7 = 1.0
    elif eye_avg >= thresholds['eye_opening_avg']['au67_2']:
        au6 = au7 = 2.0
    elif eye_avg >= thresholds['eye_opening_avg']['au67_3']:
        au6 = au7 = 3.0
    elif eye_avg >= thresholds['eye_opening_avg']['au67_4']:
        au6 = au7 = 4.0
    else:
        au6 = au7 = 5.0

    # AU9/10 - Nose/Lip
    mouth = row['mouth_opening']
    if mouth <= thresholds['mouth_opening']['au910_0']:
        au9 = au10 = 0.0
    elif mouth <= thresholds['mouth_opening']['au910_1']:
        au9 = au10 = 1.0
    elif mouth <= thresholds['mouth_opening']['au910_2']:
        au9 = au10 = 2.0
    elif mouth <= thresholds['mouth_opening']['au910_3']:
        au9 = au10 = 3.0
    elif mouth <= thresholds['mouth_opening']['au910_4']:
        au9 = au10 = 4.0
    else:
        au9 = au10 = 5.0

    # AU43 - Eye Closure (binary but scaled)
    if eye_avg < thresholds['eye_opening_avg']['au67_4']:
        au43 = 1.0
    else:
        au43 = 0.0

    # PSPI Formula
    pspi = au4 + max(au6, au7) + max(au9, au10) + au43

    # Convert 0-16 to 0-10
    intensity = (pspi / 16.0) * 10.0

    return {
        'AU4': au4,
        'AU6': au6,
        'AU7': au7,
        'AU9': au9,
        'AU10': au10,
        'AU43': au43,
        'PSPI': pspi,
        'intensity': round(min(intensity, 10.0), 2)
    }


def apply_calibrated_annotation(df, thresholds):
    """
    Apply calibrated PSPI to entire dataset
    """
    print("\n" + "=" * 70)
    print("🔄 APPLICATION DE L'ANNOTATION CALIBRÉE")
    print("=" * 70)

    # Calculate for each row
    pspi_results = df.apply(lambda row: calculate_pspi_calibrated(row, thresholds), axis=1)
    pspi_df = pd.DataFrame(pspi_results.tolist())

    # Add to dataframe
    for col in pspi_df.columns:
        df[col] = pspi_df[col]

    # Save
    df.to_csv('pain_dataset_pspi_calibrated.csv', index=False)

    print("\n📊 Nouvelles Statistiques d'Intensité:")
    print(f"   Moyenne:     {df['intensity'].mean():.2f}")
    print(f"   Écart-type:  {df['intensity'].std():.2f}")
    print(f"   Min:         {df['intensity'].min():.1f}")
    print(f"   Max:         {df['intensity'].max():.1f}")

    print(f"\n📈 Intensité Moyenne par Type de Douleur:")
    for pain_type in df['pain_type'].unique():
        mean_int = df[df['pain_type'] == pain_type]['intensity'].mean()
        std_int = df[df['pain_type'] == pain_type]['intensity'].std()
        print(f"   {pain_type:20s}: {mean_int:.2f} ± {std_int:.2f}")

    # Verification
    neutral_mean = df[df['pain_type'] == 'Neutral']['intensity'].mean()
    pain_mean = df[df['pain_type'] != 'Neutral']['intensity'].mean()

    print(f"\n🔍 Vérification de Cohérence:")
    if neutral_mean < 1.0 and neutral_mean < pain_mean:
        print(f"   ✅ Neutre ({neutral_mean:.2f}) proche de 0 et < Douleur ({pain_mean:.2f})")
        print(f"   ✅ Calibration RÉUSSIE!")
    else:
        print(f"   ⚠️  Neutre ({neutral_mean:.2f}) vs Douleur ({pain_mean:.2f})")
        print(f"   ⚠️  Ajustements supplémentaires nécessaires")

    return df


def visualize_calibration(df):
    """
    Visualize the calibrated results
    """
    print("\n" + "=" * 70)
    print("📊 CRÉATION DES VISUALISATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Intensity distribution by pain type
    pain_order = ['Neutral', 'Laser Pain', 'Algometer Pain', 'Posed Pain']

    ax1 = axes[0, 0]
    for pain_type in pain_order:
        subset = df[df['pain_type'] == pain_type]['intensity']
        ax1.hist(subset, alpha=0.6, label=pain_type, bins=20)
    ax1.set_xlabel('Intensité')
    ax1.set_ylabel('Fréquence')
    ax1.set_title('Distribution des Intensités par Type de Douleur')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot
    ax2 = axes[0, 1]
    df.boxplot(column='intensity', by='pain_type', ax=ax2)
    ax2.set_title('Box Plot - Intensité par Type de Douleur')
    ax2.set_xlabel('Type de Douleur')
    ax2.set_ylabel('Intensité')
    plt.sca(ax2)
    plt.xticks(rotation=45, ha='right')

    # 3. Mean intensity by pain type
    ax3 = axes[1, 0]
    means = df.groupby('pain_type')['intensity'].mean().sort_values()
    stds = df.groupby('pain_type')['intensity'].std()
    means.plot(kind='barh', ax=ax3, xerr=stds[means.index], capsize=5, color='skyblue')
    ax3.set_xlabel('Intensité Moyenne')
    ax3.set_title('Intensité Moyenne ± Écart-Type')
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Correlation between features and intensity
    ax4 = axes[1, 1]
    features = ['eyebrow_distance', 'left_eye_opening', 'right_eye_opening', 'mouth_opening']
    correlations = [df[feat].corr(df['intensity']) for feat in features]
    ax4.barh(features, correlations, color='coral')
    ax4.set_xlabel('Corrélation avec Intensité')
    ax4.set_title('Corrélation Features-Intensité')
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('pspi_calibration_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   ✅ Visualisation sauvegardée: pspi_calibration_results.png")


def main():
    print("\n" + "=" * 70)
    print("🎯 CALIBRATION DES SEUILS PSPI")
    print("=" * 70)

    # Step 1: Analyze distributions
    thresholds, df = analyze_feature_distributions()

    # Step 2: Apply calibrated annotation
    df = apply_calibrated_annotation(df, thresholds)

    # Step 3: Visualize
    visualize_calibration(df)

    # Save thresholds for later use
    import json
    with open('pspi_calibrated_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ CALIBRATION TERMINÉE!")
    print("=" * 70)
    print("\n📁 Fichiers générés:")
    print("   • pain_dataset_pspi_calibrated.csv")
    print("   • pspi_calibrated_thresholds.json")
    print("   • pspi_calibration_results.png")
    print("\n🚀 Prochaine étape: Ré-entraîner le modèle avec ce dataset calibré")
    print("=" * 70)


if __name__ == "__main__":
    main()