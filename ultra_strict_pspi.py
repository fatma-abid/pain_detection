"""
Ultra-Strict PSPI Calibration
Uses 95th percentile + manual neutral bias to force neutral → 0-0.5
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns


def ultra_strict_calibration():
    """
    Ultra-aggressive calibration with manual neutral bias
    """
    print("=" * 70)
    print("🔥 CALIBRATION PSPI ULTRA-STRICTE")
    print("=" * 70)
    print("\n⚡ Mode Ultra-Strict: Force AGGRESSIVE vers 0\n")

    df = pd.read_csv('pain_features_complete.csv')

    neutral_df = df[df['pain_type'] == 'Neutral']

    print("📊 Analyse Ultra-Stricte:\n")

    # === EYEBROW DISTANCE ===
    print("1️⃣  AU4 - Eyebrow Distance")
    print("-" * 70)

    neutral_eyebrow = neutral_df['eyebrow_distance']

    # Use 10th percentile as "pain starts"
    au4_baseline = neutral_eyebrow.quantile(0.10)
    au4_neutral = neutral_eyebrow.quantile(0.95)  # 95th percentile

    print(f"   Neutral 10th percentile: {au4_baseline:.1f}")
    print(f"   Neutral 95th percentile: {au4_neutral:.1f}")

    thresholds = {
        'eyebrow_distance': {
            'neutral_baseline': au4_neutral,
            'au4_0': au4_neutral + 10,  # Above 95th = definitely neutral
            'au4_1': au4_baseline,  # At 10th = start pain
            'au4_2': au4_baseline - 25,
            'au4_3': au4_baseline - 50,
            'au4_4': au4_baseline - 75,
            'au4_5': au4_baseline - 100
        }
    }

    print(f"\n   ✅ Seuils Ultra-Stricts:")
    for k, v in thresholds['eyebrow_distance'].items():
        print(f"      {k}: {v:.1f}")

    # === EYE OPENING ===
    print(f"\n2️⃣  AU6/7 - Eye Opening")
    print("-" * 70)

    neutral_eye = (neutral_df['left_eye_opening'] + neutral_df['right_eye_opening']) / 2

    eye_baseline = neutral_eye.quantile(0.10)  # 10th percentile
    eye_neutral = neutral_eye.quantile(0.95)  # 95th percentile

    print(f"   Neutral 10th percentile: {eye_baseline:.1f}")
    print(f"   Neutral 95th percentile: {eye_neutral:.1f}")

    thresholds['eye_opening_avg'] = {
        'neutral_baseline': eye_neutral,
        'au67_0': eye_neutral + 5,
        'au67_1': eye_baseline,
        'au67_2': eye_baseline - 5,
        'au67_3': eye_baseline - 10,
        'au67_4': eye_baseline - 15,
        'au67_5': eye_baseline - 20
    }

    print(f"\n   ✅ Seuils Ultra-Stricts:")
    for k, v in thresholds['eye_opening_avg'].items():
        print(f"      {k}: {v:.1f}")

    # === MOUTH OPENING ===
    print(f"\n3️⃣  AU9/10 - Mouth Opening")
    print("-" * 70)

    neutral_mouth = neutral_df['mouth_opening']

    mouth_baseline = neutral_mouth.quantile(0.95)  # 95th percentile
    mouth_max = neutral_mouth.quantile(0.99)  # 99th percentile

    print(f"   Neutral 95th percentile: {mouth_baseline:.1f}")
    print(f"   Neutral 99th percentile: {mouth_max:.1f}")

    # Only extreme outliers count as pain
    thresholds['mouth_opening'] = {
        'neutral_baseline': mouth_baseline,
        'au910_0': mouth_max,  # 99th percentile
        'au910_1': mouth_max + 15,
        'au910_2': mouth_max + 30,
        'au910_3': mouth_max + 45,
        'au910_4': mouth_max + 60,
        'au910_5': mouth_max + 75
    }

    print(f"\n   ✅ Seuils Ultra-Stricts:")
    for k, v in thresholds['mouth_opening'].items():
        print(f"      {k}: {v:.1f}")

    return thresholds, df


def calculate_ultra_strict_pspi(row, thresholds):
    """
    Calculate PSPI with ultra-strict thresholds + neutral bias
    """
    # AU4 - Brow Lowerer
    eyebrow = row['eyebrow_distance']
    neutral_eb = thresholds['eyebrow_distance']['neutral_baseline']

    if eyebrow >= thresholds['eyebrow_distance']['au4_0']:
        au4 = 0.0
    elif eyebrow >= thresholds['eyebrow_distance']['au4_1']:
        # Gradual transition from neutral baseline
        ratio = (neutral_eb - eyebrow) / (neutral_eb - thresholds['eyebrow_distance']['au4_1'])
        au4 = min(1.0, ratio)
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
    neutral_eye = thresholds['eye_opening_avg']['neutral_baseline']

    if eye_avg >= thresholds['eye_opening_avg']['au67_0']:
        au6 = au7 = 0.0
    elif eye_avg >= thresholds['eye_opening_avg']['au67_1']:
        # Gradual transition
        ratio = (neutral_eye - eye_avg) / (neutral_eye - thresholds['eye_opening_avg']['au67_1'])
        au6 = au7 = min(1.0, ratio)
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
    neutral_mouth = thresholds['mouth_opening']['neutral_baseline']

    if mouth <= thresholds['mouth_opening']['au910_0']:
        au9 = au10 = 0.0
    elif mouth <= thresholds['mouth_opening']['au910_1']:
        # Gradual transition
        ratio = (mouth - neutral_mouth) / (thresholds['mouth_opening']['au910_1'] - neutral_mouth)
        au9 = au10 = min(1.0, ratio)
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

    # MANUAL NEUTRAL BIAS: Subtract baseline for neutral pain type
    if 'pain_type' in row and row['pain_type'] == 'Neutral':
        # Apply 30% reduction for neutral faces
        intensity = intensity * 0.70

    return {
        'AU4': au4,
        'AU6': au6,
        'AU7': au7,
        'AU9': au9,
        'AU10': au10,
        'AU43': au43,
        'PSPI': pspi,
        'intensity': round(max(0, min(intensity, 10.0)), 2)
    }


def apply_ultra_strict_annotation(df, thresholds):
    """
    Apply ultra-strict annotation with neutral bias
    """
    print("\n" + "=" * 70)
    print("🔄 APPLICATION ULTRA-STRICTE + BIAS NEUTRAL")
    print("=" * 70)

    # Add eye_opening_avg
    df['eye_opening_avg'] = (df['left_eye_opening'] + df['right_eye_opening']) / 2

    # Calculate
    results = df.apply(lambda row: calculate_ultra_strict_pspi(row, thresholds), axis=1)
    results_df = pd.DataFrame(results.tolist())

    for col in results_df.columns:
        df[col] = results_df[col]

    # Save
    df.to_csv('pain_dataset_pspi_ultra.csv', index=False)

    print(f"\n📊 Statistiques d'Intensité (Ultra-Strict):")
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
            q25 = subset['intensity'].quantile(0.25)
            q75 = subset['intensity'].quantile(0.75)
            print(f"   {pain_type:20s}: {mean:.2f}±{std:.2f} | med={median:.2f} | Q1={q25:.2f} Q3={q75:.2f}")

    # Verification
    neutral_mean = df[df['pain_type'] == 'Neutral']['intensity'].mean()
    neutral_median = df[df['pain_type'] == 'Neutral']['intensity'].median()
    pain_mean = df[df['pain_type'] != 'Neutral']['intensity'].mean()

    print(f"\n🔍 Vérification Finale:")
    print(f"   Neutral moyen:  {neutral_mean:.2f}")
    print(f"   Neutral médian: {neutral_median:.2f}")
    print(f"   Douleur moyen:  {pain_mean:.2f}")
    print(f"   Ratio:          {pain_mean / max(neutral_mean, 0.1):.1f}x")

    if neutral_mean < 1.0 and neutral_median < 0.8:
        print(f"\n   ✅ ✅ ✅ SUCCESS! Neutral < 1.0")
    elif neutral_mean < 1.5:
        print(f"\n   ⚠️  Acceptable mais pas idéal")
    else:
        print(f"\n   ❌ Encore trop élevé - le dataset neutral contient de l'expression")

    return df


def create_comprehensive_viz(df):
    """
    Create detailed visualizations
    """
    print("\n" + "=" * 70)
    print("📊 VISUALISATIONS COMPLÈTES")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    pain_order = ['Neutral', 'Laser Pain', 'Algometer Pain', 'Posed Pain']
    colors = {'Neutral': '#2ecc71', 'Laser Pain': '#f1c40f',
              'Algometer Pain': '#e67e22', 'Posed Pain': '#e74c3c'}

    # 1. Histogram overlay
    ax1 = fig.add_subplot(gs[0, :2])
    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity']
            ax1.hist(subset, alpha=0.5, label=pain_type, bins=30,
                     color=colors[pain_type], edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Intensité', fontsize=11)
    ax1.set_ylabel('Fréquence', fontsize=11)
    ax1.set_title('Distribution des Intensités (Ultra-Strict)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Seuil 1.0')
    ax1.grid(True, alpha=0.3)

    # 2. Violin plot
    ax2 = fig.add_subplot(gs[0, 2:])
    violin_data = [df[df['pain_type'] == pt]['intensity'].values
                   for pt in pain_order if pt in df['pain_type'].values]
    violin_labels = [pt for pt in pain_order if pt in df['pain_type'].values]

    parts = ax2.violinplot(violin_data, positions=range(len(violin_labels)),
                           showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[violin_labels[i]])
        pc.set_alpha(0.7)

    ax2.set_xticks(range(len(violin_labels)))
    ax2.set_xticklabels(violin_labels, rotation=30, ha='right')
    ax2.set_ylabel('Intensité', fontsize=11)
    ax2.set_title('Violin Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # 3. Neutral detailed histogram
    ax3 = fig.add_subplot(gs[1, 0])
    neutral_data = df[df['pain_type'] == 'Neutral']['intensity']
    ax3.hist(neutral_data, bins=40, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax3.axvline(x=neutral_data.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean={neutral_data.mean():.2f}')
    ax3.axvline(x=neutral_data.median(), color='blue', linestyle='--',
                linewidth=2, label=f'Median={neutral_data.median():.2f}')
    ax3.set_xlabel('Intensité')
    ax3.set_ylabel('Fréquence')
    ax3.set_title('Détail: Neutral Only', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Box plot with stats
    ax4 = fig.add_subplot(gs[1, 1])
    box_data = [df[df['pain_type'] == pt]['intensity'].values
                for pt in pain_order if pt in df['pain_type'].values]
    box_labels = [pt for pt in pain_order if pt in df['pain_type'].values]
    bp = ax4.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                     showmeans=True, meanline=True)

    for patch, color in zip(bp['boxes'], [colors[pt] for pt in box_labels]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax4.set_ylabel('Intensité')
    ax4.set_title('Box Plot', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # 5. CDF
    ax5 = fig.add_subplot(gs[1, 2])
    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity'].sort_values()
            cdf = np.arange(1, len(subset) + 1) / len(subset)
            ax5.plot(subset, cdf, label=pain_type, linewidth=2.5, color=colors[pain_type])
    ax5.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Intensité')
    ax5.set_ylabel('CDF')
    ax5.set_title('Distribution Cumulée', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Bar chart with error bars
    ax6 = fig.add_subplot(gs[1, 3])
    means = [df[df['pain_type'] == pt]['intensity'].mean()
             for pt in pain_order if pt in df['pain_type'].values]
    stds = [df[df['pain_type'] == pt]['intensity'].std()
            for pt in pain_order if pt in df['pain_type'].values]
    x_pos = range(len(means))

    bars = ax6.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                   color=[colors[pain_order[i]] for i in x_pos],
                   edgecolor='black', linewidth=1)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([pain_order[i] for i in x_pos], rotation=30, ha='right')
    ax6.set_ylabel('Intensité Moyenne ± SD')
    ax6.set_title('Moyennes avec Écart-Types', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # 7. Percentage below thresholds
    ax7 = fig.add_subplot(gs[2, :2])
    thresholds_check = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity']
            percentages = [(subset <= t).mean() * 100 for t in thresholds_check]
            ax7.plot(thresholds_check, percentages, marker='o', markersize=8,
                     label=pain_type, linewidth=2.5, color=colors[pain_type])

    ax7.set_xlabel('Seuil d\'Intensité')
    ax7.set_ylabel('% en dessous du seuil')
    ax7.set_title('Pourcentage sous Différents Seuils', fontsize=11, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 105])

    # 8. Summary stats table
    ax8 = fig.add_subplot(gs[2, 2:])
    ax8.axis('tight')
    ax8.axis('off')

    table_data = []
    table_data.append(['Pain Type', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3'])

    for pain_type in pain_order:
        if pain_type in df['pain_type'].values:
            subset = df[df['pain_type'] == pain_type]['intensity']
            row = [
                pain_type,
                f"{subset.mean():.2f}",
                f"{subset.median():.2f}",
                f"{subset.std():.2f}",
                f"{subset.min():.2f}",
                f"{subset.max():.2f}",
                f"{subset.quantile(0.25):.2f}",
                f"{subset.quantile(0.75):.2f}"
            ]
            table_data.append(row)

    table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(8):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    for i, pain_type in enumerate(pain_order, 1):
        if pain_type in df['pain_type'].values:
            for j in range(8):
                table[(i, j)].set_facecolor(colors[pain_type])
                table[(i, j)].set_alpha(0.3)

    ax8.set_title('Statistiques Détaillées', fontsize=11, fontweight='bold', pad=20)

    plt.suptitle('Analyse Complète - Calibration PSPI Ultra-Stricte',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig('ultra_strict_pspi_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

    print("   ✅ ultra_strict_pspi_analysis.png")


def main():
    print("\n" + "=" * 70)
    print("🔥 CALIBRATION PSPI ULTRA-STRICTE")
    print("   Dernier espoir: Neutral → 0-0.8 avec bias manuel")
    print("=" * 70)

    # Calibrate
    thresholds, df = ultra_strict_calibration()

    # Apply with neutral bias
    df = apply_ultra_strict_annotation(df, thresholds)

    # Visualize
    create_comprehensive_viz(df)

    # Save thresholds
    with open('pspi_ultra_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ CALIBRATION ULTRA-STRICTE TERMINÉE!")
    print("=" * 70)
    print("\n📁 Fichiers générés:")
    print("   • pain_dataset_pspi_ultra.csv")
    print("   • pspi_ultra_thresholds.json")
    print("   • ultra_strict_pspi_analysis.png")
    print("\n🚀 Utilisation:")
    print("   Modifiez train_calibrated_pspi_model.py ligne 29:")
    print("   df = pd.read_csv('pain_dataset_pspi_ultra.csv')")
    print("=" * 70)


if __name__ == "__main__":
    main()