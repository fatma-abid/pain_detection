"""
Amplified Pain Calibration - Final Version
Neutral: 0-1 ✅
Pain: 4-10 (AGGRESSIVE AMPLIFICATION)
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


def amplified_calibration():
    """
    Keep neutral low, AGGRESSIVELY amplify pain
    """
    print("=" * 70)
    print("🚀 CALIBRATION FINALE - AMPLIFICATION MAXIMALE")
    print("=" * 70)
    print("\n🎯 Objectif: Neutral 0-1, Douleur 5-10\n")

    df = pd.read_csv('pain_features_complete.csv')

    neutral_df = df[df['pain_type'] == 'Neutral']
    pain_df = df[df['pain_type'] != 'Neutral']

    # Use same thresholds as balanced but with different scoring
    print("📊 Seuils de Base (identiques à balanced):\n")

    neutral_eyebrow = neutral_df['eyebrow_distance']
    pain_eyebrow = pain_df['eyebrow_distance']

    thresholds = {
        'eyebrow_distance': {
            'neutral_max': neutral_eyebrow.quantile(0.90),
            'mild_threshold': pain_eyebrow.quantile(0.75),
            'moderate_threshold': pain_eyebrow.quantile(0.25),
            'severe_threshold': pain_eyebrow.quantile(0.10)
        },
        'eye_opening_avg': {
            'neutral_min': (neutral_df['left_eye_opening'] + neutral_df['right_eye_opening']).quantile(0.10) / 2,
            'mild_threshold': (pain_df['left_eye_opening'] + pain_df['right_eye_opening']).quantile(0.75) / 2,
            'moderate_threshold': (pain_df['left_eye_opening'] + pain_df['right_eye_opening']).quantile(0.25) / 2,
            'severe_threshold': (pain_df['left_eye_opening'] + pain_df['right_eye_opening']).quantile(0.10) / 2
        },
        'mouth_opening': {
            'neutral_max': neutral_df['mouth_opening'].quantile(0.90),
            'mild_threshold': pain_df['mouth_opening'].quantile(0.25),
            'moderate_threshold': pain_df['mouth_opening'].median(),
            'severe_threshold': pain_df['mouth_opening'].quantile(0.75)
        }
    }

    print("✅ Seuils chargés\n")

    return thresholds, df


def calculate_amplified_intensity(row, thresholds):
    """
    AGGRESSIVE non-linear amplification for pain
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
        ratio = (neutral_max - eyebrow) / (neutral_max - mild_thresh)
        au4_score = ratio * 1.0  # Keep neutral zone compressed
    elif eyebrow >= mod_thresh:
        ratio = (mild_thresh - eyebrow) / (mild_thresh - mod_thresh)
        au4_score = 1.0 + ratio * 4.0  # Jump to 1-5
    elif eyebrow >= severe_thresh:
        ratio = (mod_thresh - eyebrow) / (mod_thresh - severe_thresh)
        au4_score = 5.0 + ratio * 3.0  # 5-8
    else:
        ratio = min((mod_thresh - eyebrow) / mod_thresh, 1.0)
        au4_score = 8.0 + ratio * 2.0  # 8-10

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
        au67_score = ratio * 1.0
    elif eye_avg >= severe_thresh_eye:
        ratio = (mod_thresh_eye - eye_avg) / (mod_thresh_eye - severe_thresh_eye)
        au67_score = 1.0 + ratio * 4.0  # 1-5
    else:
        ratio = max(0, (severe_thresh_eye - eye_avg) / severe_thresh_eye)
        au67_score = 5.0 + ratio * 5.0  # 5-10

    # === AU9/10 - Mouth (MOST IMPORTANT) ===
    mouth = row['mouth_opening']
    neutral_max_mouth = thresholds['mouth_opening']['neutral_max']
    mild_thresh_mouth = thresholds['mouth_opening']['mild_threshold']
    mod_thresh_mouth = thresholds['mouth_opening']['moderate_threshold']
    severe_thresh_mouth = thresholds['mouth_opening']['severe_threshold']

    if mouth <= neutral_max_mouth:
        au910_score = 0.0
    elif mouth <= mild_thresh_mouth:
        ratio = (mouth - neutral_max_mouth) / (mild_thresh_mouth - neutral_max_mouth)
        au910_score = ratio * 1.5  # 0-1.5
    elif mouth <= mod_thresh_mouth:
        ratio = (mouth - mild_thresh_mouth) / (mod_thresh_mouth - mild_thresh_mouth)
        au910_score = 1.5 + ratio * 4.5  # 1.5-6
    elif mouth <= severe_thresh_mouth:
        ratio = (mouth - mod_thresh_mouth) / (severe_thresh_mouth - mod_thresh_mouth)
        au910_score = 6.0 + ratio * 2.5  # 6-8.5
    else:
        ratio = min((mouth - severe_thresh_mouth) / severe_thresh_mouth, 1.0)
        au910_score = 8.5 + ratio * 1.5  # 8.5-10

    # === AU43 - Eye Closure ===
    if eye_avg < thresholds['eye_opening_avg']['severe_threshold']:
        au43_score = 2.5  # Strong boost
    else:
        au43_score = 0.0

    # === WEIGHTED COMBINATION ===
    # Mouth is KING - 50% weight
    intensity = (
            au4_score * 0.20 +  # 20% eyebrow
            au67_score * 0.25 +  # 25% eyes
            au910_score * 0.50 +  # 50% mouth (DOMINANT)
            au43_score * 0.05  # 5% closure
    )

    # === TYPE-SPECIFIC ADJUSTMENTS ===
    if 'pain_type' in row:
        if row['pain_type'] == 'Neutral':
            intensity = intensity * 0.55  # Heavy suppression for neutral
        else:
            # AGGRESSIVE PAIN BOOST
            intensity = intensity * 1.35  # +35% for pain types

    # === EXPONENTIAL AMPLIFICATION FOR HIGH VALUES ===
    if intensity > 3.0:
        # Apply power function: y = 3 + (x-3)^1.5
        excess = intensity - 3.0
        intensity = 3.0 + np.power(excess, 1.5)

    # === FINAL CLIPPING ===
    intensity = np.clip(intensity, 0, 10)

    return {
        'au4_score': round(au4_score, 2),
        'au67_score': round(au67_score, 2),
        'au910_score': round(au910_score, 2),
        'au43_score': round(au43_score, 2),
        'intensity': round(intensity, 2)
    }


def apply_amplified_annotation(df, thresholds):
    """
    Apply amplified annotation
    """
    print("=" * 70)
    print("🔄 APPLICATION DE L'AMPLIFICATION MAXIMALE")
    print("=" * 70)

    # Add eye_opening_avg
    df['eye_opening_avg'] = (df['left_eye_opening'] + df['right_eye_opening']) / 2

    # Calculate
    results = df.apply(lambda row: calculate_amplified_intensity(row, thresholds), axis=1)
    results_df = pd.DataFrame(results.tolist())

    for col in results_df.columns:
        df[col] = results_df[col]

    # Save
    df.to_csv('pain_dataset_pspi_amplified.csv', index=False)

    print(f"\n📊 Statistiques d'Intensité (Amplifiée):")
    print(f"   Moyenne globale: {df['intensity'].mean():.2f}")
    print(f"   Écart-type:      {df['intensity'].std():.2f}")
    print(f"   Min:             {df['intensity'].min():.1f}")
    print(f"   Max:             {df['intensity'].max():.1f}")

    print(f"\n📈 Par Type de Douleur:")
    print("=" * 70)

    for pain_type in ['Neutral', 'Laser Pain', 'Algometer Pain', 'Posed Pain']:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity']
            mean = subset.mean()
            median = subset.median()
            std = subset.std()
            min_val = subset.min()
            max_val = subset.max()
            q1 = subset.quantile(0.25)
            q3 = subset.quantile(0.75)

            # Evaluation
            if pain_type == 'Neutral':
                status = "✅" if mean < 1.0 else "⚠️"
                target = "< 1.0"
            elif pain_type == 'Laser Pain':
                status = "✅" if mean >= 3.0 else "⚠️"
                target = "> 3.0"
            else:
                status = "✅" if mean >= 4.0 else "⚠️"
                target = "> 4.0"

            print(f"   {status} {pain_type:20s} (target {target}):")
            print(f"      Mean={mean:.2f}±{std:.2f} | Median={median:.2f} | Range=[{min_val:.2f}, {max_val:.2f}]")

    # Final verification
    print(f"\n" + "=" * 70)
    print("🔍 VÉRIFICATION FINALE")
    print("=" * 70)

    neutral_mean = df[df['pain_type'] == 'Neutral']['intensity'].mean()
    neutral_max = df[df['pain_type'] == 'Neutral']['intensity'].max()

    laser_mean = df[df['pain_type'] == 'Laser Pain']['intensity'].mean()
    algometer_mean = df[df['pain_type'] == 'Algometer Pain']['intensity'].mean()
    posed_mean = df[df['pain_type'] == 'Posed Pain']['intensity'].mean()
    pain_mean = df[df['pain_type'] != 'Neutral']['intensity'].mean()

    print(f"\n   Neutral:")
    print(f"      • Moyenne: {neutral_mean:.2f}")
    print(f"      • Maximum: {neutral_max:.2f}")
    print(f"      Status: {'✅ PARFAIT' if neutral_mean < 1.0 else '⚠️ Encore trop haut'}")

    print(f"\n   Douleur:")
    print(f"      • Laser:     {laser_mean:.2f}")
    print(f"      • Algometer: {algometer_mean:.2f}")
    print(f"      • Posed:     {posed_mean:.2f}")
    print(f"      • Moyenne:   {pain_mean:.2f}")
    print(f"      Status: {'✅ BON' if pain_mean >= 4.0 else '⚠️ Peut être amélioré'}")

    print(f"\n   Séparation:")
    print(f"      • Écart absolu: {pain_mean - neutral_mean:.2f}")
    print(f"      • Ratio:        {pain_mean / max(neutral_mean, 0.1):.1f}x")

    success_neutral = neutral_mean < 1.0
    success_pain = pain_mean >= 4.0

    print(f"\n" + "=" * 70)
    if success_neutral and success_pain:
        print("   ✅ ✅ ✅ CALIBRATION PARFAITE!")
        print("   • Neutral bien bas")
        print("   • Douleur bien amplifiée")
        print("   • Prêt pour l'entraînement!")
    elif success_neutral:
        print("   ✅ Neutral parfait")
        print("   ⚠️  Douleur pourrait être plus haute")
        print("   → Mais utilisable en l'état")
    else:
        print("   ⚠️  Ajustements nécessaires")
    print("=" * 70)

    return df


def visualize_amplified(df):
    """
    Create visualization
    """
    print("\n📊 Création de la visualisation...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    pain_order = ['Neutral', 'Laser Pain', 'Algometer Pain', 'Posed Pain']
    colors = {'Neutral': '#27ae60', 'Laser Pain': '#f39c12',
              'Algometer Pain': '#e67e22', 'Posed Pain': '#c0392b'}

    # 1. Histogram
    ax1 = axes[0, 0]
    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity']
            ax1.hist(subset, alpha=0.6, label=pain_type, bins=35,
                     color=colors[pain_type], edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Intensité', fontsize=11)
    ax1.set_ylabel('Fréquence', fontsize=11)
    ax1.set_title('Distribution (Amplifiée)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.axvline(x=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Limite Neutral')
    ax1.axvline(x=4.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zone Douleur Forte')
    ax1.grid(True, alpha=0.3)

    # 2. Violin
    ax2 = axes[0, 1]
    violin_data = [df[df['pain_type'] == pt]['intensity'].values
                   for pt in pain_order if pt in df['pain_type'].values]
    parts = ax2.violinplot(violin_data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[pain_order[i]])
        pc.set_alpha(0.7)
    ax2.set_xticks(range(1, len(pain_order) + 1))
    ax2.set_xticklabels(pain_order, rotation=30, ha='right')
    ax2.set_ylabel('Intensité', fontsize=11)
    ax2.set_title('Violin Plot', fontsize=12, fontweight='bold')
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=4.0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Box plot
    ax3 = axes[0, 2]
    box_data = [df[df['pain_type'] == pt]['intensity'].values
                for pt in pain_order if pt in df['pain_type'].values]
    bp = ax3.boxplot(box_data, tick_labels=pain_order, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], [colors[pt] for pt in pain_order]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel('Intensité', fontsize=11)
    ax3.set_title('Box Plot', fontsize=12, fontweight='bold')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    ax3.axhline(y=4.0, color='red', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Bar chart
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
    ax4.set_ylabel('Intensité Moyenne', fontsize=11)
    ax4.set_title('Moyennes ± SD', fontsize=12, fontweight='bold')
    ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    ax4.axhline(y=4.0, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='y')

    for i, m in enumerate(means):
        ax4.text(i, m + stds[i] + 0.3, f'{m:.2f}', ha='center', fontweight='bold', fontsize=10)

    # 5. CDF
    ax5 = axes[1, 1]
    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity'].sort_values()
            cdf = np.arange(1, len(subset) + 1) / len(subset)
            ax5.plot(subset, cdf, label=pain_type, linewidth=2.5, color=colors[pain_type])
    ax5.axvline(x=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax5.axvline(x=4.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Intensité', fontsize=11)
    ax5.set_ylabel('CDF', fontsize=11)
    ax5.set_title('Distribution Cumulée', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Range comparison
    ax6 = axes[1, 2]
    for i, pain_type in enumerate(pain_order):
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity']
            min_val = subset.min()
            q1 = subset.quantile(0.25)
            median = subset.median()
            q3 = subset.quantile(0.75)
            max_val = subset.max()

            ax6.plot([min_val, q1], [i, i], color=colors[pain_type], linewidth=8, alpha=0.3)
            ax6.plot([q1, q3], [i, i], color=colors[pain_type], linewidth=8, alpha=0.7)
            ax6.plot([q3, max_val], [i, i], color=colors[pain_type], linewidth=8, alpha=0.3)
            ax6.plot(median, i, 'o', color='white', markersize=8, markeredgecolor='black', markeredgewidth=2)

    ax6.set_yticks(range(len(pain_order)))
    ax6.set_yticklabels(pain_order)
    ax6.set_xlabel('Intensité', fontsize=11)
    ax6.set_title('Plages de Valeurs', fontsize=12, fontweight='bold')
    ax6.axvline(x=1.0, color='green', linestyle='--', alpha=0.5)
    ax6.axvline(x=4.0, color='red', linestyle='--', alpha=0.5)
    ax6.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Calibration PSPI Amplifiée - Analyse Complète',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('amplified_pspi_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   ✅ amplified_pspi_calibration.png")


def main():
    print("\n" + "=" * 70)
    print("🚀 CALIBRATION FINALE - AMPLIFICATION MAXIMALE")
    print("=" * 70)

    thresholds, df = amplified_calibration()
    df = apply_amplified_annotation(df, thresholds)
    visualize_amplified(df)

    with open('pspi_amplified_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ CALIBRATION AMPLIFIÉE TERMINÉE!")
    print("=" * 70)
    print("\n📁 Fichiers:")
    print("   • pain_dataset_pspi_amplified.csv")
    print("   • pspi_amplified_thresholds.json")
    print("   • amplified_pspi_calibration.png")
    print("\n🚀 Utilisation:")
    print("   1. Modifier train_calibrated_pspi_model.py ligne 29:")
    print("      df = pd.read_csv('pain_dataset_pspi_amplified.csv')")
    print("   2. python train_calibrated_pspi_model.py")
    print("   3. python test_calibrated_realtime.py")
    print("=" * 70)


if __name__ == "__main__":
    main()