"""
Balanced PSPI Calibration
- Neutral stays low (0-1)
- Pain gets amplified (proper 3-10 range)
Uses non-linear scaling to expand pain range
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


def balanced_calibration():
    """
    Balanced approach: Keep neutral low, amplify pain
    """
    print("=" * 70)
    print("⚖️  CALIBRATION PSPI ÉQUILIBRÉE")
    print("=" * 70)
    print("\n🎯 Objectif: Neutral 0-1, Douleur 3-10\n")

    df = pd.read_csv('pain_features_complete.csv')

    neutral_df = df[df['pain_type'] == 'Neutral']
    pain_df = df[df['pain_type'] != 'Neutral']

    print("📊 Analyse Comparative Neutral vs Douleur:\n")

    # === EYEBROW DISTANCE ===
    print("1️⃣  AU4 - Eyebrow Distance")
    print("-" * 70)

    neutral_eyebrow = neutral_df['eyebrow_distance']
    pain_eyebrow = pain_df['eyebrow_distance']

    # Use neutral's median as neutral baseline
    # Use pain's 25th percentile as pain baseline
    neutral_median = neutral_eyebrow.median()
    pain_q25 = pain_eyebrow.quantile(0.25)
    pain_q75 = pain_eyebrow.quantile(0.75)

    print(f"   Neutral median: {neutral_median:.1f}")
    print(f"   Pain Q1 (25th): {pain_q25:.1f}")
    print(f"   Pain Q3 (75th): {pain_q75:.1f}")

    thresholds = {
        'eyebrow_distance': {
            'neutral_max': neutral_eyebrow.quantile(0.90),  # 90% of neutral
            'mild_threshold': pain_q75,  # Pain starts
            'moderate_threshold': pain_q25,  # Definite pain
            'severe_threshold': pain_eyebrow.quantile(0.10)  # Strong pain
        }
    }

    print(f"\n   ✅ Seuils Équilibrés:")
    for k, v in thresholds['eyebrow_distance'].items():
        print(f"      {k}: {v:.1f}")

    # === EYE OPENING ===
    print(f"\n2️⃣  AU6/7 - Eye Opening")
    print("-" * 70)

    neutral_eye = (neutral_df['left_eye_opening'] + neutral_df['right_eye_opening']) / 2
    pain_eye = (pain_df['left_eye_opening'] + pain_df['right_eye_opening']) / 2

    neutral_median_eye = neutral_eye.median()
    pain_q25_eye = pain_eye.quantile(0.25)
    pain_q75_eye = pain_eye.quantile(0.75)

    print(f"   Neutral median: {neutral_median_eye:.1f}")
    print(f"   Pain Q1 (25th): {pain_q25_eye:.1f}")
    print(f"   Pain Q3 (75th): {pain_q75_eye:.1f}")

    thresholds['eye_opening_avg'] = {
        'neutral_min': neutral_eye.quantile(0.10),
        'mild_threshold': pain_q75_eye,
        'moderate_threshold': pain_q25_eye,
        'severe_threshold': pain_eye.quantile(0.10)
    }

    print(f"\n   ✅ Seuils Équilibrés:")
    for k, v in thresholds['eye_opening_avg'].items():
        print(f"      {k}: {v:.1f}")

    # === MOUTH OPENING ===
    print(f"\n3️⃣  AU9/10 - Mouth Opening")
    print("-" * 70)

    neutral_mouth = neutral_df['mouth_opening']
    pain_mouth = pain_df['mouth_opening']

    neutral_median_mouth = neutral_mouth.median()
    pain_q25_mouth = pain_mouth.quantile(0.25)
    pain_q75_mouth = pain_mouth.quantile(0.75)

    print(f"   Neutral median: {neutral_median_mouth:.1f}")
    print(f"   Pain Q1 (25th): {pain_q25_mouth:.1f}")
    print(f"   Pain Q3 (75th): {pain_q75_mouth:.1f}")

    thresholds['mouth_opening'] = {
        'neutral_max': neutral_mouth.quantile(0.90),
        'mild_threshold': pain_q25_mouth,
        'moderate_threshold': pain_mouth.median(),
        'severe_threshold': pain_q75_mouth
    }

    print(f"\n   ✅ Seuils Équilibrés:")
    for k, v in thresholds['mouth_opening'].items():
        print(f"      {k}: {v:.1f}")

    return thresholds, df


def calculate_balanced_intensity(row, thresholds):
    """
    Calculate intensity with non-linear scaling
    - Neutral range: 0-1.5 (compressed)
    - Pain range: 2-10 (expanded)
    """

    # === AU4 - Eyebrow ===
    eyebrow = row['eyebrow_distance']
    neutral_max = thresholds['eyebrow_distance']['neutral_max']
    mild_thresh = thresholds['eyebrow_distance']['mild_threshold']
    mod_thresh = thresholds['eyebrow_distance']['moderate_threshold']
    severe_thresh = thresholds['eyebrow_distance']['severe_threshold']

    if eyebrow >= neutral_max:
        au4_score = 0.0
    elif eyebrow >= mild_thresh:
        # Transition zone: neutral to mild
        ratio = (neutral_max - eyebrow) / (neutral_max - mild_thresh)
        au4_score = ratio * 1.5  # Max 1.5 in transition
    elif eyebrow >= mod_thresh:
        # Mild to moderate
        ratio = (mild_thresh - eyebrow) / (mild_thresh - mod_thresh)
        au4_score = 1.5 + ratio * 2.5  # 1.5 to 4.0
    elif eyebrow >= severe_thresh:
        # Moderate to severe
        ratio = (mod_thresh - eyebrow) / (mod_thresh - severe_thresh)
        au4_score = 4.0 + ratio * 3.0  # 4.0 to 7.0
    else:
        au4_score = 7.0 + min((mod_thresh - eyebrow) / mod_thresh, 1.0) * 3.0

    # === AU6/7 - Eyes ===
    eye_avg = (row['left_eye_opening'] + row['right_eye_opening']) / 2
    neutral_min = thresholds['eye_opening_avg']['neutral_min']
    mild_thresh_eye = thresholds['eye_opening_avg']['mild_threshold']
    mod_thresh_eye = thresholds['eye_opening_avg']['moderate_threshold']
    severe_thresh_eye = thresholds['eye_opening_avg']['severe_threshold']

    if eye_avg >= mild_thresh_eye:
        au67_score = 0.0
    elif eye_avg >= mod_thresh_eye:
        ratio = (mild_thresh_eye - eye_avg) / (mild_thresh_eye - mod_thresh_eye)
        au67_score = ratio * 1.5
    elif eye_avg >= severe_thresh_eye:
        ratio = (mod_thresh_eye - eye_avg) / (mod_thresh_eye - severe_thresh_eye)
        au67_score = 1.5 + ratio * 2.5
    else:
        ratio = max(0, (severe_thresh_eye - eye_avg) / severe_thresh_eye)
        au67_score = 4.0 + ratio * 6.0

    # === AU9/10 - Mouth ===
    mouth = row['mouth_opening']
    neutral_max_mouth = thresholds['mouth_opening']['neutral_max']
    mild_thresh_mouth = thresholds['mouth_opening']['mild_threshold']
    mod_thresh_mouth = thresholds['mouth_opening']['moderate_threshold']
    severe_thresh_mouth = thresholds['mouth_opening']['severe_threshold']

    if mouth <= neutral_max_mouth:
        au910_score = 0.0
    elif mouth <= mild_thresh_mouth:
        ratio = (mouth - neutral_max_mouth) / (mild_thresh_mouth - neutral_max_mouth)
        au910_score = ratio * 1.5
    elif mouth <= mod_thresh_mouth:
        ratio = (mouth - mild_thresh_mouth) / (mod_thresh_mouth - mild_thresh_mouth)
        au910_score = 1.5 + ratio * 2.5
    elif mouth <= severe_thresh_mouth:
        ratio = (mouth - mod_thresh_mouth) / (severe_thresh_mouth - mod_thresh_mouth)
        au910_score = 4.0 + ratio * 3.0
    else:
        ratio = min((mouth - severe_thresh_mouth) / severe_thresh_mouth, 1.0)
        au910_score = 7.0 + ratio * 3.0

    # === AU43 - Eye Closure ===
    if eye_avg < thresholds['eye_opening_avg']['severe_threshold']:
        au43_score = 2.0  # Stronger weight
    else:
        au43_score = 0.0

    # === COMBINE WITH WEIGHTS ===
    # Weight mouth more heavily as it's most indicative
    intensity = (
            au4_score * 0.25 +  # 25% eyebrow
            au67_score * 0.30 +  # 30% eyes
            au910_score * 0.40 +  # 40% mouth (most important)
            au43_score * 0.05  # 5% closure
    )

    # Apply pain type adjustment
    if 'pain_type' in row:
        if row['pain_type'] == 'Neutral':
            intensity = intensity * 0.60  # Reduce neutral by 40%
        else:
            intensity = intensity * 1.15  # Boost pain by 15%

    # Non-linear amplification for high values
    if intensity > 4.0:
        # Amplify high pain: y = x + (x-4)^1.3
        excess = intensity - 4.0
        intensity = 4.0 + excess * 1.4

    return {
        'au4_score': round(au4_score, 2),
        'au67_score': round(au67_score, 2),
        'au910_score': round(au910_score, 2),
        'au43_score': round(au43_score, 2),
        'intensity': round(np.clip(intensity, 0, 10), 2)
    }


def apply_balanced_annotation(df, thresholds):
    """
    Apply balanced annotation
    """
    print("\n" + "=" * 70)
    print("🔄 APPLICATION DE LA CALIBRATION ÉQUILIBRÉE")
    print("=" * 70)

    # Add eye_opening_avg
    df['eye_opening_avg'] = (df['left_eye_opening'] + df['right_eye_opening']) / 2

    # Calculate
    results = df.apply(lambda row: calculate_balanced_intensity(row, thresholds), axis=1)
    results_df = pd.DataFrame(results.tolist())

    for col in results_df.columns:
        df[col] = results_df[col]

    # Save
    df.to_csv('pain_dataset_pspi_balanced.csv', index=False)

    print(f"\n📊 Statistiques d'Intensité (Équilibrée):")
    print(f"   Moyenne globale: {df['intensity'].mean():.2f}")
    print(f"   Écart-type:      {df['intensity'].std():.2f}")
    print(f"   Min:             {df['intensity'].min():.1f}")
    print(f"   Max:             {df['intensity'].max():.1f}")

    print(f"\n📈 Par Type de Douleur:")
    print("-" * 70)

    for pain_type in ['Neutral', 'Laser Pain', 'Algometer Pain', 'Posed Pain']:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity']
            mean = subset.mean()
            median = subset.median()
            std = subset.std()
            q1 = subset.quantile(0.25)
            q3 = subset.quantile(0.75)

            # Visual indicator
            if pain_type == 'Neutral':
                indicator = "✅" if mean < 1.0 else "⚠️"
            else:
                indicator = "✅" if mean >= 3.0 else "⚠️"

            print(f"   {indicator} {pain_type:20s}: {mean:.2f}±{std:.2f} | med={median:.2f} | IQR=[{q1:.2f}, {q3:.2f}]")

    # Verification
    neutral_mean = df[df['pain_type'] == 'Neutral']['intensity'].mean()
    pain_mean = df[df['pain_type'] != 'Neutral']['intensity'].mean()

    print(f"\n🔍 Vérification Finale:")
    print(f"   Neutral moyen:   {neutral_mean:.2f}")
    print(f"   Douleur moyen:   {pain_mean:.2f}")
    print(f"   Écart (ratio):   {pain_mean - neutral_mean:.2f} ({pain_mean / max(neutral_mean, 0.1):.1f}x)")

    success = neutral_mean < 1.2 and pain_mean >= 3.5

    if success:
        print(f"\n   ✅ ✅ ✅ CALIBRATION RÉUSSIE!")
        print(f"   • Neutral bien bas ({neutral_mean:.2f})")
        print(f"   • Douleur bien amplifiée ({pain_mean:.2f})")
    else:
        print(f"\n   ⚠️  Ajustements possibles nécessaires")

    return df


def visualize_balanced(df):
    """
    Visualize balanced calibration
    """
    print("\n" + "=" * 70)
    print("📊 CRÉATION DES VISUALISATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    pain_order = ['Neutral', 'Laser Pain', 'Algometer Pain', 'Posed Pain']
    colors = {'Neutral': '#27ae60', 'Laser Pain': '#f39c12',
              'Algometer Pain': '#e67e22', 'Posed Pain': '#c0392b'}

    # 1. Histogram
    ax1 = axes[0, 0]
    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity']
            ax1.hist(subset, alpha=0.6, label=pain_type, bins=30,
                     color=colors[pain_type], edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Intensité')
    ax1.set_ylabel('Fréquence')
    ax1.set_title('Distribution des Intensités (Équilibrée)', fontweight='bold')
    ax1.legend()
    ax1.axvspan(0, 1.5, alpha=0.1, color='green', label='Zone Neutral')
    ax1.axvspan(3, 10, alpha=0.1, color='red', label='Zone Douleur')
    ax1.grid(True, alpha=0.3)

    # 2. Violin plot
    ax2 = axes[0, 1]
    violin_data = [df[df['pain_type'] == pt]['intensity'].values
                   for pt in pain_order if pt in df['pain_type'].values]
    parts = ax2.violinplot(violin_data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[pain_order[i]])
        pc.set_alpha(0.7)
    ax2.set_xticks(range(1, len(pain_order) + 1))
    ax2.set_xticklabels(pain_order, rotation=30, ha='right')
    ax2.set_ylabel('Intensité')
    ax2.set_title('Violin Plot', fontweight='bold')
    ax2.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='Limite Neutral')
    ax2.axhline(y=3.0, color='red', linestyle='--', alpha=0.5, label='Début Douleur')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    # 3. Box plot
    ax3 = axes[0, 2]
    box_data = [df[df['pain_type'] == pt]['intensity'].values
                for pt in pain_order if pt in df['pain_type'].values]
    bp = ax3.boxplot(box_data, tick_labels=pain_order, patch_artist=True,
                     showmeans=True)
    for patch, color in zip(bp['boxes'], [colors[pt] for pt in pain_order]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel('Intensité')
    ax3.set_title('Box Plot', fontweight='bold')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax3.axhline(y=1.5, color='green', linestyle='--', alpha=0.5)
    ax3.axhline(y=3.0, color='red', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Mean comparison
    ax4 = axes[1, 0]
    means = [df[df['pain_type'] == pt]['intensity'].mean()
             for pt in pain_order if pt in df['pain_type'].values]
    stds = [df[df['pain_type'] == pt]['intensity'].std()
            for pt in pain_order if pt in df['pain_type'].values]
    x_pos = range(len(means))
    bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                   color=[colors[pain_order[i]] for i in x_pos],
                   edgecolor='black', linewidth=1.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(pain_order, rotation=30, ha='right')
    ax4.set_ylabel('Intensité Moyenne ± SD')
    ax4.set_title('Moyennes par Type', fontweight='bold')
    ax4.axhline(y=1.5, color='green', linestyle='--', alpha=0.5)
    ax4.axhline(y=3.0, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax4.text(i, m + s + 0.2, f'{m:.2f}', ha='center', fontweight='bold')

    # 5. CDF
    ax5 = axes[1, 1]
    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity'].sort_values()
            cdf = np.arange(1, len(subset) + 1) / len(subset)
            ax5.plot(subset, cdf, label=pain_type, linewidth=2.5, color=colors[pain_type])
    ax5.axvline(x=1.5, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax5.axvline(x=3.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Intensité')
    ax5.set_ylabel('CDF')
    ax5.set_title('Distribution Cumulée', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Scatter: mouth vs intensity
    ax6 = axes[1, 2]
    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]
            ax6.scatter(subset['mouth_opening'], subset['intensity'],
                        alpha=0.4, s=20, label=pain_type, color=colors[pain_type])
    ax6.set_xlabel('Mouth Opening (pixels)')
    ax6.set_ylabel('Intensité')
    ax6.set_title('Mouth Opening vs Intensity', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('balanced_pspi_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   ✅ balanced_pspi_calibration.png")


def main():
    print("\n" + "=" * 70)
    print("⚖️  CALIBRATION PSPI ÉQUILIBRÉE")
    print("   Neutral bas + Douleur amplifiée")
    print("=" * 70)

    # Calibrate
    thresholds, df = balanced_calibration()

    # Apply
    df = apply_balanced_annotation(df, thresholds)

    # Visualize
    visualize_balanced(df)

    # Save
    with open('pspi_balanced_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ CALIBRATION ÉQUILIBRÉE TERMINÉE!")
    print("=" * 70)
    print("\n📁 Fichiers générés:")
    print("   • pain_dataset_pspi_balanced.csv")
    print("   • pspi_balanced_thresholds.json")
    print("   • balanced_pspi_calibration.png")
    print("\n🚀 Utilisation:")
    print("   Modifiez train_calibrated_pspi_model.py ligne 29:")
    print("   df = pd.read_csv('pain_dataset_pspi_balanced.csv')")
    print("\n   Puis: python train_calibrated_pspi_model.py")
    print("=" * 70)


if __name__ == "__main__":
    main()