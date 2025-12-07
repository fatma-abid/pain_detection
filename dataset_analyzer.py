import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

class DatasetAnalyzer:
    """
    Analyse la structure du dataset de détection de douleur
    avant l'extraction des features
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.stats = defaultdict(lambda: defaultdict(int))
        
    def analyze(self):
        """
        Analyse complète de la structure du dataset
        """
        print("=" * 70)
        print("🔍 ANALYSE DE LA STRUCTURE DU DATASET")
        print("=" * 70)
        print(f"📁 Chemin: {self.dataset_path}")
        print()
        
        if not self.dataset_path.exists():
            print(f"❌ ERREUR: Le chemin n'existe pas!")
            return
        
        # Extensions d'images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Compteurs globaux
        total_images = 0
        subjects = []
        pain_types = set()
        frame_types = set()
        
        # Détails par sujet
        subject_details = []
        
        # Parcours de la structure
        for subject_folder in sorted(self.dataset_path.iterdir()):
            if not subject_folder.is_dir() or not subject_folder.name.startswith('S'):
                continue
            
            subject_id = subject_folder.name
            subjects.append(subject_id)
            subject_total = 0
            
            for pain_type_folder in subject_folder.iterdir():
                if not pain_type_folder.is_dir():
                    continue
                
                pain_type = pain_type_folder.name
                pain_types.add(pain_type)
                
                for frame_folder in pain_type_folder.iterdir():
                    if not frame_folder.is_dir():
                        continue
                    
                    frame_type = frame_folder.name
                    frame_types.add(frame_type)
                    
                    # Compte les images
                    images = [f for f in frame_folder.iterdir() 
                             if f.suffix.lower() in image_extensions]
                    
                    count = len(images)
                    total_images += count
                    subject_total += count
                    
                    self.stats[subject_id][f"{pain_type}_{frame_type}"] = count
                    
                    # Détails pour le tableau
                    subject_details.append({
                        'Sujet': subject_id,
                        'Type Douleur': pain_type,
                        'Type Frame': frame_type,
                        'Nb Images': count
                    })
        
        # Affichage des résultats
        print("📊 STATISTIQUES GLOBALES")
        print("-" * 70)
        print(f"👥 Nombre de sujets: {len(subjects)}")
        print(f"   Sujets: {', '.join(subjects)}")
        print()
        print(f"🎭 Types de douleur détectés: {len(pain_types)}")
        for pain_type in sorted(pain_types):
            print(f"   • {pain_type}")
        print()
        print(f"🎨 Types de frames détectés: {len(frame_types)}")
        for frame_type in sorted(frame_types):
            print(f"   • {frame_type}")
        print()
        print(f"🖼️  Total d'images: {total_images:,}")
        print()
        
        # Tableau détaillé
        print("=" * 70)
        print("📋 DÉTAILS PAR SUJET")
        print("=" * 70)
        
        df_details = pd.DataFrame(subject_details)
        
        # Vue groupée par sujet et type de douleur
        pivot = df_details.pivot_table(
            index=['Sujet', 'Type Douleur'],
            columns='Type Frame',
            values='Nb Images',
            fill_value=0,
            aggfunc='sum'
        )
        
        print(pivot.to_string())
        print()
        
        # Statistiques par type de douleur
        print("=" * 70)
        print("📈 DISTRIBUTION PAR TYPE DE DOULEUR")
        print("=" * 70)
        pain_summary = df_details.groupby('Type Douleur')['Nb Images'].sum().sort_values(ascending=False)
        for pain_type, count in pain_summary.items():
            percentage = (count / total_images) * 100
            print(f"{pain_type:20s}: {count:5d} images ({percentage:5.1f}%)")
        print()
        
        # Statistiques par type de frame
        print("=" * 70)
        print("🎨 DISTRIBUTION PAR TYPE DE FRAME")
        print("=" * 70)
        frame_summary = df_details.groupby('Type Frame')['Nb Images'].sum().sort_values(ascending=False)
        for frame_type, count in frame_summary.items():
            percentage = (count / total_images) * 100
            print(f"{frame_type:25s}: {count:5d} images ({percentage:5.1f}%)")
        print()
        
        # Recommandations
        print("=" * 70)
        print("💡 RECOMMANDATIONS")
        print("=" * 70)
        
        # Vérifier l'équilibre des classes
        pain_counts = df_details.groupby('Type Douleur')['Nb Images'].sum()
        max_count = pain_counts.max()
        min_count = pain_counts.min()
        
        if max_count / min_count > 2:
            print("⚠️  DÉSÉQUILIBRE détecté entre les classes de douleur!")
            print(f"   Ratio max/min: {max_count/min_count:.1f}x")
            print("   → Considérez l'augmentation de données (data augmentation)")
            print("   → Ou utilisez des poids de classes lors de l'entraînement")
        else:
            print("✅ Les classes sont relativement équilibrées")
        
        print()
        
        # Recommandation sur le type de frame
        print("📌 Type de frame recommandé:")
        if 'colour_frames' in frame_types:
            print("   → 'colour_frames' (images couleur standard)")
            print("   Avantage: Informations de couleur (rougissement, pâleur)")
        if 'colour_oval_frames' in frame_types:
            print("   → 'colour_oval_frames' (visage détouré)")
            print("   Avantage: Moins de bruit de fond, focus sur le visage")
        
        print()
        print("=" * 70)
        
        return df_details
    
    def check_image_quality(self, sample_size=10):
        """
        Vérifie la qualité d'un échantillon d'images
        """
        import cv2
        import random
        
        print("\n" + "=" * 70)
        print("🔬 VÉRIFICATION DE LA QUALITÉ DES IMAGES")
        print("=" * 70)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_images = []
        
        for image_path in self.dataset_path.rglob('*'):
            if image_path.suffix.lower() in image_extensions:
                all_images.append(image_path)
        
        if len(all_images) == 0:
            print("❌ Aucune image trouvée!")
            return
        
        sample = random.sample(all_images, min(sample_size, len(all_images)))
        
        resolutions = []
        sizes_kb = []
        corrupted = 0
        
        for img_path in sample:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupted += 1
                    continue
                
                h, w = img.shape[:2]
                resolutions.append((w, h))
                
                size_kb = img_path.stat().st_size / 1024
                sizes_kb.append(size_kb)
                
            except Exception as e:
                corrupted += 1
        
        if resolutions:
            avg_width = sum(r[0] for r in resolutions) / len(resolutions)
            avg_height = sum(r[1] for r in resolutions) / len(resolutions)
            avg_size = sum(sizes_kb) / len(sizes_kb)
            
            print(f"📏 Résolution moyenne: {avg_width:.0f} x {avg_height:.0f} pixels")
            print(f"💾 Taille moyenne: {avg_size:.1f} KB")
            print(f"🖼️  Images testées: {len(sample)}")
            print(f"❌ Images corrompues: {corrupted}")
            
            if corrupted > 0:
                print("\n⚠️  ATTENTION: Certaines images sont corrompues!")
        
        print("=" * 70)


# Utilisation
if __name__ == "__main__":
    # Chemin vers votre dataset
    dataset_path = r"D:\SUPCOM\Computer_Vision\dataset\Pictures\Modified"
    
    # Création de l'analyseur
    analyzer = DatasetAnalyzer(dataset_path)
    
    # Lancement de l'analyse
    df = analyzer.analyze()
    
    # Vérification optionnelle de la qualité des images
    analyzer.check_image_quality(sample_size=20)
    
    # Sauvegarde de l'analyse dans un fichier
    if df is not None:
        df.to_csv('dataset_structure_analysis.csv', index=False)
        print("\n💾 Analyse détaillée sauvegardée dans: dataset_structure_analysis.csv")