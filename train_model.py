"""
Entraînement de Modèles ML pour la Détection de Douleur
Teste plusieurs algorithmes et compare leurs performances
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, precision_score, recall_score)

# Modèles
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Configuration
sns.set_style("whitegrid")
np.random.seed(42)

class PainDetectionTrainer:
    """
    Classe pour entraîner et évaluer plusieurs modèles ML
    """
    
    def __init__(self, csv_path='pain_features_complete.csv'):
        self.csv_path = csv_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self, test_size=0.2):
        """
        Charge et prépare les données
        """
        print("=" * 70)
        print("📂 CHARGEMENT ET PRÉPARATION DES DONNÉES")
        print("=" * 70)
        
        # Chargement
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Données chargées: {len(self.df)} images")
        
        # Séparation features et labels
        feature_cols = [col for col in self.df.columns 
                       if col not in ['subject_id', 'pain_type', 'pain_label', 
                                     'frame_type', 'image_name', 'image_path']]
        
        X = self.df[feature_cols].values
        y = self.df['pain_label'].values
        
        print(f"   • Features: {len(feature_cols)}")
        print(f"   • Classes: {len(np.unique(y))}")
        
        # Features utilisées
        print(f"\n📊 Features utilisées:")
        for i, col in enumerate(feature_cols, 1):
            print(f"   {i:2d}. {col}")
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n✅ Split des données:")
        print(f"   • Training:   {len(self.X_train)} images ({(1-test_size)*100:.0f}%)")
        print(f"   • Test:       {len(self.X_test)} images ({test_size*100:.0f}%)")
        
        # Normalisation
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\n✅ Normalisation appliquée (StandardScaler)")
        
        # Distribution des classes
        print(f"\n📈 Distribution des classes (Training):")
        unique, counts = np.unique(self.y_train, return_counts=True)
        pain_types = ['Neutral', 'Posed Pain', 'Algometer Pain', 'Laser Pain']
        for label, count in zip(unique, counts):
            print(f"   • {pain_types[label]:20s}: {count:3d} images ({count/len(self.y_train)*100:.1f}%)")
    
    def train_models(self):
        """
        Entraîne plusieurs modèles ML
        """
        print("\n" + "=" * 70)
        print("🤖 ENTRAÎNEMENT DES MODÈLES")
        print("=" * 70)
        
        # Définition des modèles
        self.models = {
            'SVM (Linear)': SVC(kernel='linear', random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # Entraînement
        for name, model in self.models.items():
            print(f"\n🔧 Entraînement: {name}...", end=' ', flush=True)
            start_time = datetime.now()
            
            model.fit(self.X_train, self.y_train)
            
            duration = (datetime.now() - start_time).total_seconds()
            print(f"✅ ({duration:.2f}s)")
            
            # Prédictions
            y_pred = model.predict(self.X_test)
            
            # Métriques
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'duration': duration
            }
    
    def compare_models(self):
        """
        Compare les performances des modèles
        """
        print("\n" + "=" * 70)
        print("📊 COMPARAISON DES MODÈLES")
        print("=" * 70)
        
        # Tableau de comparaison
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Modèle': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1']:.4f}",
                'CV Score': f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f}",
                'Temps (s)': f"{result['duration']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Meilleur modèle
        best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_accuracy = self.results[best_model_name]['accuracy']
        
        print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
        print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        return comparison_df
    
    def plot_results(self, output_dir='model_results'):
        """
        Visualise les résultats
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\n📊 Création des visualisations...")
        
        # 1. Comparaison des accuracies
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        
        axes[0, 0].barh(models, accuracies, color='skyblue')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_title('Accuracy par Modèle')
        axes[0, 0].set_xlim([0, 1])
        for i, v in enumerate(accuracies):
            axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # F1-Score
        f1_scores = [self.results[m]['f1'] for m in models]
        axes[0, 1].barh(models, f1_scores, color='lightcoral')
        axes[0, 1].set_xlabel('F1-Score')
        axes[0, 1].set_title('F1-Score par Modèle')
        axes[0, 1].set_xlim([0, 1])
        for i, v in enumerate(f1_scores):
            axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # CV Scores
        cv_means = [self.results[m]['cv_mean'] for m in models]
        cv_stds = [self.results[m]['cv_std'] for m in models]
        axes[1, 0].barh(models, cv_means, xerr=cv_stds, color='lightgreen', capsize=5)
        axes[1, 0].set_xlabel('Cross-Validation Score')
        axes[1, 0].set_title('Cross-Validation (5-fold)')
        axes[1, 0].set_xlim([0, 1])
        
        # Temps d'entraînement
        durations = [self.results[m]['duration'] for m in models]
        axes[1, 1].barh(models, durations, color='plum')
        axes[1, 1].set_xlabel('Temps (secondes)')
        axes[1, 1].set_title("Temps d'Entraînement")
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/models_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ {output_dir}/models_comparison.png")
        
        # 2. Matrice de confusion pour le meilleur modèle
        best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        y_pred = self.results[best_model_name]['y_pred']
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Neutral', 'Posed', 'Algometer', 'Laser'],
                   yticklabels=['Neutral', 'Posed', 'Algometer', 'Laser'])
        plt.xlabel('Prédiction')
        plt.ylabel('Vérité')
        plt.title(f'Matrice de Confusion - {best_model_name}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_best.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ {output_dir}/confusion_matrix_best.png")
    
    def save_best_model(self, output_path='best_model.pkl'):
        """
        Sauvegarde le meilleur modèle
        """
        best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = self.results[best_model_name]['model']
        
        # Sauvegarde du modèle et du scaler
        joblib.dump({
            'model': best_model,
            'scaler': self.scaler,
            'model_name': best_model_name,
            'accuracy': self.results[best_model_name]['accuracy']
        }, output_path)
        
        print(f"\n💾 Meilleur modèle sauvegardé: {output_path}")
        print(f"   Modèle: {best_model_name}")
        print(f"   Accuracy: {self.results[best_model_name]['accuracy']:.4f}")
    
    def detailed_report(self, output_file='training_report.txt'):
        """
        Génère un rapport détaillé
        """
        best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        y_pred = self.results[best_model_name]['y_pred']
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("RAPPORT D'ENTRAÎNEMENT - DÉTECTION DE DOULEUR\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Dataset: {self.csv_path}\n")
            f.write(f"Total images: {len(self.df)}\n")
            f.write(f"Training set: {len(self.X_train)}\n")
            f.write(f"Test set: {len(self.X_test)}\n\n")
            
            f.write("2. RÉSULTATS PAR MODÈLE\n")
            f.write("-" * 70 + "\n")
            for name, result in self.results.items():
                f.write(f"\n{name}:\n")
                f.write(f"  Accuracy:  {result['accuracy']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall:    {result['recall']:.4f}\n")
                f.write(f"  F1-Score:  {result['f1']:.4f}\n")
                f.write(f"  CV Score:  {result['cv_mean']:.4f} ± {result['cv_std']:.4f}\n")
            
            f.write(f"\n\n3. MEILLEUR MODÈLE\n")
            f.write("-" * 70 + "\n")
            f.write(f"Modèle: {best_model_name}\n")
            f.write(f"Accuracy: {self.results[best_model_name]['accuracy']:.4f}\n\n")
            
            f.write("Rapport de Classification:\n")
            f.write(classification_report(self.y_test, y_pred, 
                                         target_names=['Neutral', 'Posed', 'Algometer', 'Laser']))
            
        print(f"\n📝 Rapport détaillé: {output_file}")

def main():
    """
    Fonction principale
    """
    print("\n" + "=" * 70)
    print("🎯 ENTRAÎNEMENT DE MODÈLES ML")
    print("   Détection Automatique de la Douleur")
    print("=" * 70)
    
    # Initialisation
    trainer = PainDetectionTrainer('pain_features_complete.csv')
    
    # Préparation des données
    trainer.load_and_prepare_data(test_size=0.2)
    
    # Entraînement
    trainer.train_models()
    
    # Comparaison
    comparison_df = trainer.compare_models()
    
    # Visualisation
    trainer.plot_results()
    
    # Sauvegarde
    trainer.save_best_model('best_pain_model.pkl')
    
    # Rapport
    trainer.detailed_report('training_report.txt')
    
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("=" * 70)
    print("\n📁 Fichiers générés:")
    print("   • best_pain_model.pkl (modèle entraîné)")
    print("   • model_results/models_comparison.png")
    print("   • model_results/confusion_matrix_best.png")
    print("   • training_report.txt")
    print("\n🚀 Le modèle est prêt à être utilisé pour la prédiction!")
    print("=" * 70)

if __name__ == "__main__":
    main()