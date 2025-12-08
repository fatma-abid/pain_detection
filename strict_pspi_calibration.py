"""
Strict PSPI Calibration - Forces Neutral to be ~0
Uses neutral's 75th percentile as baseline (0 intensity)
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns


def strict_calibration():
    """
    Aggressive calibration to force neutral faces to ~0 intensity
    """
    print("=" * 70)
    print("🎯 CALIBRATION PSPI STRICTE")
    print("=" * 70)
    print("\n⚠️  Mode Strict: Force les visages neutres vers 0\n")

    df = pd.read_csv('pain_features_complete.csv')

    # Separate neutral and pain
    neutral_df = df[df['pain_type'] == 'Neutral']
    pain_df = df[df['pain_type'] != 'Neutral']

    print("📊 Analyse des Distributions:\n")

    # === EYEBROW DISTANCE (AU4) ===
    print("1️⃣  AU4 - Eyebrow Distance")
    print("-" * 70)

    # Use 75th percentile of neutral as "definitely no AU4"
    neutral_eyebrow = neutral_df['eyebrow_distance']

    # For AU4, LOWER distance = more furrowing (pain)
    # So neutral should have HIGHER values
    au4_baseline = neutral_eyebrow.quantile(0.25)  # 25th percentile = start of pain
    au4_neutral = neutral_eyebrow.quantile(0.75)   # 75th percentile = definitely neutral

    print(f"   Neutral 25th percentile: {au4_baseline:.1f}")
    print(f"   Neutral 75th percentile: {au4_neutral:.1f}")

    # Create thresholds where only values BELOW 25th percentile count as pain
    thresholds = {
        'eyebrow_distance': {
            'au4_0': au4_neutral,           # Above 75th percentile = no pain
            'au4_1': au4_baseline,          # At 25th percentile = mild
            'au4_2': au4_baseline - 20,     # Below = moderate
            'au4_3': au4_baseline - 40,     # Much below = strong
            'au4_4': au4_baseline - 60,
            'au4_5': au4_baseline - 80
        }
    }

    print(f"\n   ✅ Seuils Stricts (Eyebrow):")
    for k, v in thresholds['eyebrow_distance'].items():
        print(f"      {k}: < {v:.1f}")

    # === EYE OPENING (AU6/7) ===
    print(f"\n2️⃣  AU6/7 - Eye Opening")
    print("-" * 70)

    # Calculate eye opening average
    neutral_eye = (neutral_df['left_eye_opening'] + neutral_df['right_eye_opening']) / 2

    # For eyes, LOWER opening = more squeezing (pain)
    eye_baseline = neutral_eye.quantile(0.25)  # 25th percentile
    eye_neutral = neutral_eye.quantile(0.75)   # 75th percentile

    print(f"   Neutral 25th percentile: {eye_baseline:.1f}")
    print(f"   Neutral 75th percentile: {eye_neutral:.1f}")

    thresholds['eye_opening_avg'] = {
        'au67_0': eye_neutral,        # Wide open = no pain
        'au67_1': eye_baseline,       # At 25th percentile
        'au67_2': eye_baseline - 5,
        'au67_3': eye_baseline - 10,
        'au67_4': eye_baseline - 15,
        'au67_5': eye_baseline - 20
    }

    print(f"\n   ✅ Seuils Stricts (Eye):")
    for k, v in thresholds['eye_opening_avg'].items():
        print(f"      {k}: < {v:.1f}")

    # === MOUTH OPENING (AU9/10) ===
    print(f"\n3️⃣  AU9/10 - Mouth Opening")
    print("-" * 70)

    neutral_mouth = neutral_df['mouth_opening']

    # For mouth, HIGHER opening = pain expression
    # Use 90th percentile of neutral as baseline (anything above = pain)
    mouth_baseline = neutral_mouth.quantile(0.90)  # 90th percentile of neutral
    mouth_median = neutral_mouth.median()

    print(f"   Neutral median: {mouth_median:.1f}")
    print(f"   Neutral 90th percentile: {mouth_baseline:.1f}")

    # Only values ABOVE 90th percentile count as pain
    thresholds['mouth_opening'] = {
        'au910_0': mouth_baseline,           # At 90th percentile = start of pain
        'au910_1': mouth_baseline + 10,
        'au910_2': mouth_baseline + 20,
        'au910_3': mouth_baseline + 30,
        'au910_4': mouth_baseline + 40,
        'au910_5': mouth_baseline + 50
    }

    print(f"\n   ✅ Seuils Stricts (Mouth):")
    for k, v in thresholds['mouth_opening'].items():
        print(f"      {k}: > {v:.1f}")

    return thresholds, df


def calculate_strict_pspi(row, thresholds):
    """
    Calculate PSPI with strict thresholds
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

    # AU9/10 - Mouth
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

    # AU43 - Eye Closure
    if eye_avg < thresholds['eye_opening_avg']['au67_4']:
        au43 = 1.0
    else:
        au43 = 0.0

    # PSPI
    pspi = au4 + max(au6, au7) + max(au9, au10) + au43
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


def apply_strict_annotation(df, thresholds):
    """
    Apply strict annotation
    """
    print("\n" + "=" * 70)
    print("🔄 APPLICATION DES SEUILS STRICTS")
    print("=" * 70)

    # Add eye_opening_avg column for calculations
    df['eye_opening_avg'] = (df['left_eye_opening'] + df['right_eye_opening']) / 2

    # Calculate
    results = df.apply(lambda row: calculate_strict_pspi(row, thresholds), axis=1)
    results_df = pd.DataFrame(results.tolist())

    for col in results_df.columns:
        df[col] = results_df[col]

    # Save
    df.to_csv('pain_dataset_pspi_strict.csv', index=False)

    print(f"\n📊 Statistiques d'Intensité (Strict):")
    print(f"   Moyenne:     {df['intensity'].mean():.2f}")
    print(f"   Écart-type:  {df['intensity'].std():.2f}")
    print(f"   Min:         {df['intensity'].min():.1f}")
    print(f"   Max:         {df['intensity'].max():.1f}")

    print(f"\n📈 Par Type de Douleur:")
    for pain_type in ['Neutral', 'Laser Pain', 'Algometer Pain', 'Posed Pain']:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]
            mean = subset['intensity'].mean()
            std = subset['intensity'].std()
            median = subset['intensity'].median()
            print(f"   {pain_type:20s}: {mean:.2f} ± {std:.2f} (median={median:.2f})")

    # Verification
    neutral_mean = df[df['pain_type'] == 'Neutral']['intensity'].mean()
    pain_mean = df[df['pain_type'] != 'Neutral']['intensity'].mean()

    print(f"\n🔍 Vérification:")
    print(f"   Neutral moyen: {neutral_mean:.2f}")
    print(f"   Douleur moyen: {pain_mean:.2f}")

    if neutral_mean < 1.0:
        print(f"   ✅ SUCCESS! Neutral < 1.0")
    elif neutral_mean < 1.5:
        print(f"   ⚠️  Acceptable (neutral < 1.5)")
    else:
        print(f"   ❌ Neutral encore trop élevé")

    return df


def visualize_strict(df):
    """
    Visualize strict calibration results
    """
    print("\n" + "=" * 70)
    print("📊 VISUALISATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    pain_order = ['Neutral', 'Laser Pain', 'Algometer Pain', 'Posed Pain']
    colors = {'Neutral': 'green', 'Laser Pain': 'yellow',
              'Algometer Pain': 'orange', 'Posed Pain': 'red'}

    # 1. Histogram by pain type
    ax1 = axes[0, 0]
    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity']
            ax1.hist(subset, alpha=0.6, label=pain_type, bins=25, color=colors[pain_type])
    ax1.set_xlabel('Intensité')
    ax1.set_ylabel('Fréquence')
    ax1.set_title('Distribution des Intensités')
    ax1.legend()
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Seuil 1.0')
    ax1.grid(True, alpha=0.3)

    # 2. Box plot
    ax2 = axes[0, 1]
    box_data = [df[df['pain_type'] == pt]['intensity'].values
                for pt in pain_order if pt in df['pain_type'].values]
    box_labels = [pt for pt in pain_order if pt in df['pain_type'].values]
    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], [colors[pt] for pt in box_labels]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel('Intensité')
    ax2.set_title('Box Plot par Type')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Mean ± SD
    ax3 = axes[0, 2]
    means = df.groupby('pain_type')['intensity'].mean().reindex(pain_order)
    stds = df.groupby('pain_type')['intensity'].std().reindex(pain_order)
    x_pos = range(len(means))
    bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                   color=[colors[pt] for pt in means.index])
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(means.index, rotation=45, ha='right')
    ax3.set_ylabel('Intensité Moyenne')
    ax3.set_title('Moyenne ± Écart-Type')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=1)

    # 4. Neutral only - detailed
    ax4 = axes[1, 0]
    neutral_intensities = df[df['pain_type'] == 'Neutral']['intensity']
    ax4.hist(neutral_intensities, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax4.axvline(x=neutral_intensities.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean={neutral_intensities.mean():.2f}')
    ax4.axvline(x=neutral_intensities.median(), color='blue', linestyle='--',
                linewidth=2, label=f'Median={neutral_intensities.median():.2f}')
    ax4.set_xlabel('Intensité')
    ax4.set_ylabel('Fréquence')
    ax4.set_title('Détail: Neutral Only')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. CDF (Cumulative Distribution)
    ax5 = axes[1, 1]
    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity'].sort_values()
            cdf = np.arange(1, len(subset) + 1) / len(subset)
            ax5.plot(subset, cdf, label=pain_type, linewidth=2, color=colors[pain_type])
    ax5.axvline(x=1.0, color='red', linestyle='--', linewidth=1)
    ax5.set_xlabel('Intensité')
    ax5.set_ylabel('CDF')
    ax5.set_title('Fonction de Distribution Cumulée')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Percentage below threshold
    ax6 = axes[1, 2]
    thresholds_to_check = [0.5, 1.0, 1.5, 2.0]
    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity']
            percentages = [(subset <= t).mean() * 100 for t in thresholds_to_check]
            ax6.plot(thresholds_to_check, percentages, marker='o',
                    label=pain_type, linewidth=2, color=colors[pain_type])
    ax6.set_xlabel('Seuil d\'Intensité')
    ax6.set_ylabel('% en dessous du seuil')
    ax6.set_title('Pourcentage sous Seuil')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('strict_pspi_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   ✅ strict_pspi_calibration.png")


def main():
    print("\n" + "=" * 70)
    print("🎯 CALIBRATION PSPI STRICTE")
    print("   Objectif: Neutral → 0-0.5, Douleur → 3-10")
    print("=" * 70)

    # Step 1: Strict calibration
    thresholds, df = strict_calibration()

    # Step 2: Apply
    df = apply_strict_annotation(df, thresholds)

    # Step 3: Visualize
    visualize_strict(df)

    # Save thresholds
    with open('pspi_strict_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ CALIBRATION STRICTE TERMINÉE!")
    print("=" * 70)
    print("\n📁 Fichiers générés:")
    print("   • pain_dataset_pspi_strict.csv")
    print("   • pspi_strict_thresholds.json")
    print("   • strict_pspi_calibration.png")
    print("\n🚀 Prochaine étape:")
    print("   python train_calibrated_pspi_model.py")
    print("   (Remplacez 'pain_dataset_pspi_calibrated.csv' par 'pain_dataset_pspi_strict.csv')")
    print("=" * 70)


if __name__ == "__main__":
    main()