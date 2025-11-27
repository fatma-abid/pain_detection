"""
Évaluation du Modèle - Métriques de Performance
MSE, MAE, R², Accuracy par niveau, etc.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                            r2_score, confusion_matrix, classification_report)

def evaluate_model():
    print("=" * 60)
    print("📊 ÉVALUATION DU MODÈLE D'INTENSITÉ")
    print("=" * 60)
    
    # Charger données et modèle
    df = pd.read_csv('pain_dataset_annotated.csv')
    model_data = joblib.load('intensity_model.pkl')
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    print(f"✅ Dataset: {len(df)} samples")
    print(f"✅ Features: {len(feature_cols)}")
    
    # Préparer données
    X = df[feature_cols].values
    y = df['intensity'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # ============================================================
    # MÉTRIQUES DE RÉGRESSION
    # ============================================================
    print("\n" + "=" * 60)
    print("📈 MÉTRIQUES DE RÉGRESSION")
    print("=" * 60)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n   MSE  (Mean Squared Error):     {mse:.4f}")
    print(f"   RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"   MAE  (Mean Absolute Error):     {mae:.4f}")
    print(f"   R²   (Coefficient de Détermination): {r2:.4f}")
    
    # Interprétation
    print(f"\n   💡 Interprétation:")
    print(f"      • Erreur moyenne: ±{mae:.2f} points sur échelle 0-10")
    print(f"      • Le modèle explique {r2*100:.1f}% de la variance")
    
    # ============================================================
    # CONVERSION EN CLASSIFICATION (Niveaux de Douleur)
    # ============================================================
    print("\n" + "=" * 60)
    print("🎯 MÉTRIQUES DE CLASSIFICATION (par niveaux)")
    print("=" * 60)
    
    # Convertir en 3 classes
    def to_class(intensity):
        if intensity < 3: return 0      # Légère
        elif intensity < 6: return 1    # Modérée
        else: return 2                  # Sévère
    
    y_test_class = np.array([to_class(y) for y in y_test])
    y_pred_class = np.array([to_class(y) for y in y_pred])
    
    # Accuracy
    accuracy = np.mean(y_test_class == y_pred_class)
    print(f"\n   🎯 ACCURACY: {accuracy*100:.2f}%")
    
    # Rapport détaillé
    print(f"\n   📋 Rapport de Classification:")
    print("-" * 60)
    labels = ['Légère (0-3)', 'Modérée (3-6)', 'Sévère (6-10)']
    print(classification_report(y_test_class, y_pred_class, 
                                target_names=labels, digits=3))
    
    # ============================================================
    # CROSS-VALIDATION
    # ============================================================
    print("=" * 60)
    print("🔄 CROSS-VALIDATION (5-Fold)")
    print("=" * 60)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"\n   R² scores: {cv_scores}")
    print(f"   Moyenne:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    cv_mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"\n   MSE scores: {-cv_mse}")
    print(f"   Moyenne MSE: {-cv_mse.mean():.4f} ± {cv_mse.std():.4f}")
    
    # ============================================================
    # VISUALISATIONS
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 CRÉATION DES GRAPHIQUES")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Prédiction vs Réel
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.5, c='blue')
    ax1.plot([0, 10], [0, 10], 'r--', linewidth=2, label='Parfait')
    ax1.set_xlabel('Intensité Réelle')
    ax1.set_ylabel('Intensité Prédite')
    ax1.set_title(f'Prédiction vs Réel (R² = {r2:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution des erreurs
    ax2 = axes[0, 1]
    errors = y_pred - y_test
    ax2.hist(errors, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Erreur (Prédit - Réel)')
    ax2.set_ylabel('Fréquence')
    ax2.set_title(f'Distribution des Erreurs (MAE = {mae:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # 3. Matrice de confusion (par niveaux)
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test_class, y_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Légère', 'Modérée', 'Sévère'],
                yticklabels=['Légère', 'Modérée', 'Sévère'])
    ax3.set_xlabel('Prédiction')
    ax3.set_ylabel('Réel')
    ax3.set_title(f'Matrice de Confusion (Accuracy = {accuracy*100:.1f}%)')
    
    # 4. Feature Importance
    ax4 = axes[1, 1]
    importance = sorted(zip(feature_cols, model.feature_importances_), 
                       key=lambda x: x[1], reverse=True)
    feats, imps = zip(*importance)
    ax4.barh(feats, imps, color='skyblue')
    ax4.set_xlabel('Importance')
    ax4.set_title('Importance des Features')
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Graphique sauvegardé: model_evaluation.png")
    
    # ============================================================
    # RÉSUMÉ FINAL
    # ============================================================
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DES PERFORMANCES")
    print("=" * 60)
    print(f"""
    ┌────────────────────────────────────────┐
    │  MÉTRIQUES DE RÉGRESSION               │
    ├────────────────────────────────────────┤
    │  MSE:   {mse:.4f}                       │
    │  RMSE:  {rmse:.4f}                       │
    │  MAE:   {mae:.4f}                       │
    │  R²:    {r2:.4f}                       │
    ├────────────────────────────────────────┤
    │  ACCURACY (3 classes): {accuracy*100:.2f}%          │
    ├────────────────────────────────────────┤
    │  Cross-Val R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}       │
    └────────────────────────────────────────┘
    """)
    
    # Interprétation qualité
    print("   💡 INTERPRÉTATION:")
    if r2 > 0.9:
        print("      ⭐⭐⭐ Excellent modèle!")
    elif r2 > 0.7:
        print("      ⭐⭐ Bon modèle")
    elif r2 > 0.5:
        print("      ⭐ Modèle acceptable")
    else:
        print("      ⚠️ Modèle à améliorer")
    
    if accuracy > 0.9:
        print(f"      ⭐⭐⭐ Excellente accuracy par niveau!")
    elif accuracy > 0.75:
        print(f"      ⭐⭐ Bonne accuracy par niveau")
    else:
        print(f"      ⭐ Accuracy correcte")


if __name__ == "__main__":
    evaluate_model()