"""
Analyse Exploratoire des Données (EDA)
Visualisation des caractéristiques extraites pour comprendre les patterns de douleur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

def load_data(csv_path='pain_features_complete.csv'):
    """Charge les données"""
    print("=" * 70)
    print("📂 CHARGEMENT DES DONNÉES")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    print(f"✅ Fichier chargé: {csv_path}")
    print(f"   • Lignes: {len(df)}")
    print(f"   • Colonnes: {len(df.columns)}")
    print(f"   • Sujets: {df['subject_id'].nunique()}")
    print(f"   • Types de douleur: {df['pain_type'].nunique()}")
    
    return df

def explore_distributions(df, output_dir='analysis_plots'):
    """Visualise la distribution des features par type de douleur"""
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("📊 ANALYSE DES DISTRIBUTIONS")
    print("=" * 70)
    
    # Sélection des features numériques
    numeric_features = [
        'eyebrow_distance', 'left_eye_opening', 'right_eye_opening',
        'mouth_opening', 'mouth_width', 'left_cheek_elevation',
        'right_cheek_elevation', 'left_eyebrow_angle', 'right_eyebrow_angle'
    ]
    
    # Filtrer les features qui existent
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    print(f"Features analysées: {len(numeric_features)}")
    
    # 1. Boxplots par feature
    print("\n📦 Création des boxplots...")
    n_features = len(numeric_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(numeric_features):
        ax = axes[idx]
        df.boxplot(column=feature, by='pain_type', ax=ax)
        ax.set_title(f'{feature}')
        ax.set_xlabel('Type de Douleur')
        ax.set_ylabel('Valeur')
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
    
    # Supprimer les axes vides
    for idx in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Distribution des Features par Type de Douleur', fontsize=16, y=1.002)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplots_by_pain_type.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Sauvegardé: {output_dir}/boxplots_by_pain_type.png")
    
    # 2. Violin plots pour features clés
    print("\n🎻 Création des violin plots...")
    key_features = ['eyebrow_distance', 'mouth_opening', 'left_eye_opening', 'mouth_width']
    key_features = [f for f in key_features if f in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(key_features[:4]):
        sns.violinplot(data=df, x='pain_type', y=feature, ax=axes[idx])
        axes[idx].set_title(f'{feature}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Type de Douleur', fontsize=10)
        axes[idx].set_ylabel('Valeur', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Violin Plots - Features Principales', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/violin_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Sauvegardé: {output_dir}/violin_plots.png")

def correlation_analysis(df, output_dir='analysis_plots'):
    """Analyse de corrélation entre features"""
    print("\n" + "=" * 70)
    print("🔗 ANALYSE DE CORRÉLATION")
    print("=" * 70)
    
    # Sélection des colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Retirer les colonnes non pertinentes
    exclude = ['pain_label']
    numeric_cols = [col for col in numeric_cols if col not in exclude]
    
    # Matrice de corrélation
    corr_matrix = df[numeric_cols].corr()
    
    # Visualisation
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Matrice de Corrélation des Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Sauvegardé: {output_dir}/correlation_matrix.png")
    
    # Top corrélations
    print("\n📊 Top 10 Corrélations:")
    corr_pairs = corr_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs < 1]  # Exclure auto-corrélation
    top_corr = corr_pairs.abs().sort_values(ascending=False).head(10)
    
    for (feat1, feat2), value in top_corr.items():
        print(f"   • {feat1:25s} ↔ {feat2:25s}: {value:.3f}")

def compare_pain_types(df, output_dir='analysis_plots'):
    """Compare les moyennes par type de douleur"""
    print("\n" + "=" * 70)
    print("🔬 COMPARAISON PAR TYPE DE DOULEUR")
    print("=" * 70)
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f != 'pain_label']
    
    # Calcul des moyennes
    pain_means = df.groupby('pain_type')[numeric_features].mean()
    
    print("\n📊 Moyennes par Type de Douleur:")
    print(pain_means.to_string())
    
    # Visualisation des moyennes
    pain_means_T = pain_means.T
    
    fig, ax = plt.subplots(figsize=(14, 10))
    pain_means_T.plot(kind='barh', ax=ax, width=0.8)
    ax.set_xlabel('Valeur Moyenne', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Comparaison des Moyennes par Type de Douleur', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Type de Douleur', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/means_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Sauvegardé: {output_dir}/means_comparison.png")

def analyze_key_features(df, output_dir='analysis_plots'):
    """Analyse approfondie des features les plus discriminantes"""
    print("\n" + "=" * 70)
    print("🎯 ANALYSE DES FEATURES CLÉS")
    print("=" * 70)
    
    # Calcul de la variance inter-classes pour chaque feature
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f != 'pain_label']
    
    variances = {}
    for feature in numeric_features:
        pain_means = df.groupby('pain_type')[feature].mean()
        variance = pain_means.var()
        variances[feature] = variance
    
    # Trier par variance
    sorted_vars = sorted(variances.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📈 Features les Plus Discriminantes (par variance inter-classes):")
    for idx, (feature, var) in enumerate(sorted_vars[:10], 1):
        print(f"   {idx:2d}. {feature:30s}: {var:.2f}")
    
    # Visualisation des top 4 features
    top_features = [f[0] for f in sorted_vars[:4]]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        for pain_type in df['pain_type'].unique():
            data = df[df['pain_type'] == pain_type][feature]
            ax.hist(data, alpha=0.6, label=pain_type, bins=20)
        
        ax.set_xlabel('Valeur', fontsize=10)
        ax.set_ylabel('Fréquence', fontsize=10)
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution des Top 4 Features Discriminantes', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_discriminant_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Sauvegardé: {output_dir}/top_discriminant_features.png")

def generate_summary_report(df, output_file='analysis_report.txt'):
    """Génère un rapport textuel d'analyse"""
    print("\n" + "=" * 70)
    print("📝 GÉNÉRATION DU RAPPORT")
    print("=" * 70)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RAPPORT D'ANALYSE - DÉTECTION DE DOULEUR\n")
        f.write("=" * 70 + "\n\n")
        
        # Informations générales
        f.write("1. INFORMATIONS GÉNÉRALES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Nombre d'images: {len(df)}\n")
        f.write(f"Nombre de sujets: {df['subject_id'].nunique()}\n")
        f.write(f"Nombre de features: {len(df.columns) - 6}\n")  # Exclure métadonnées
        f.write(f"\nDistribution par type de douleur:\n")
        for pain_type, count in df['pain_type'].value_counts().items():
            f.write(f"  • {pain_type:20s}: {count:3d} images ({count/len(df)*100:.1f}%)\n")
        
        # Statistiques descriptives
        f.write("\n\n2. STATISTIQUES DESCRIPTIVES\n")
        f.write("-" * 70 + "\n")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'pain_label']
        
        stats = df[numeric_cols].describe()
        f.write(stats.to_string())
        
        # Moyennes par type de douleur
        f.write("\n\n3. MOYENNES PAR TYPE DE DOULEUR\n")
        f.write("-" * 70 + "\n")
        means = df.groupby('pain_type')[numeric_cols].mean()
        f.write(means.to_string())
        
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("Fin du rapport\n")
        f.write("=" * 70 + "\n")
    
    print(f"   ✅ Rapport sauvegardé: {output_file}")

def main():
    """Fonction principale"""
    print("\n" + "=" * 70)
    print("🔬 ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    print("   Détection Automatique de la Douleur")
    print("=" * 70)
    
    # Chargement
    df = load_data('pain_features_complete.csv')
    
    # Analyses
    explore_distributions(df)
    correlation_analysis(df)
    compare_pain_types(df)
    analyze_key_features(df)
    generate_summary_report(df)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSE TERMINÉE!")
    print("=" * 70)
    print("\n📁 Fichiers générés:")
    print("   • analysis_plots/boxplots_by_pain_type.png")
    print("   • analysis_plots/violin_plots.png")
    print("   • analysis_plots/correlation_matrix.png")
    print("   • analysis_plots/means_comparison.png")
    print("   • analysis_plots/top_discriminant_features.png")
    print("   • analysis_report.txt")
    print("\n🚀 Prochaine étape: Entraînement des modèles ML")
    print("=" * 70)

if __name__ == "__main__":
    main()