import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import argparse


class NIHDataProcessor:
   def __init__(self, data_root: str):
       self.data_root = data_root
       self.images_dir = os.path.join(data_root, 'images')
       self.metadata_path = os.path.join(data_root, 'Data_Entry_2017_v2020.csv')
       self.processed_df = None
       
       os.makedirs(os.path.join(self.data_root, 'processed'), exist_ok=True)
       os.makedirs(os.path.join(self.data_root, 'reports'), exist_ok=True)
   
   def load_metadata(self) -> pd.DataFrame:
       """Load and parse metadata CSV file"""
       if not os.path.exists(self.metadata_path):
           raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
       
       metadata = pd.read_csv(self.metadata_path)
       print(f"Loaded metadata: {len(metadata)} records")
       
       # Clean column names
       metadata.columns = metadata.columns.str.strip()
       
       return metadata
   
   def load_split_files(self) -> Tuple[List[str], List[str]]:
       """Load train/val and test split files"""
       train_val_path = os.path.join(self.data_root, 'train_val_list.txt')
       test_path = os.path.join(self.data_root, 'test_list.txt')
       
       if not os.path.exists(train_val_path):
           raise FileNotFoundError("train_val_list.txt not found")
       if not os.path.exists(test_path):
           raise FileNotFoundError("test_list.txt not found")
       
       with open(train_val_path, 'r') as f:
           train_val_files = [line.strip() for line in f if line.strip()]
       
       with open(test_path, 'r') as f:
           test_files = [line.strip() for line in f if line.strip()]
           
       print(f"Train/val files: {len(train_val_files)}")
       print(f"Test files: {len(test_files)}")
       
       return train_val_files, test_files
   
   def create_base_dataframe(self) -> pd.DataFrame:
       """Create base dataframe with metadata and split information"""
       metadata = self.load_metadata()
       train_val_files, test_files = self.load_split_files()
       
       # Create split mapping
       split_map = {}
       for file in train_val_files:
           split_map[file] = 'train_val'
       for file in test_files:
           split_map[file] = 'test'
       
       # Filter metadata to only include files in splits
       metadata['official_split'] = metadata['Image Index'].map(split_map)
       df = metadata[metadata['official_split'].notna()].copy()
       
       # Add image path column
       df['image_path'] = df['Image Index'].apply(
           lambda x: os.path.join(self.images_dir, x)
       )
       
       # Verify image files exist
       print("Verifying image files...")
       df['file_exists'] = df['image_path'].apply(os.path.exists)
       missing_count = (~df['file_exists']).sum()
       if missing_count > 0:
           print(f"Warning: {missing_count} images not found on disk")
           df = df[df['file_exists']].copy()
       
       df = df.drop('file_exists', axis=1)
       
       # Rename columns for consistency
       df = df.rename(columns={
           'Image Index': 'image_filename',
           'Patient ID': 'patient_id',
           'Patient Age': 'age',
           'Patient Sex': 'sex',
           'View Position': 'view_position',
           'Finding Labels': 'finding_labels'
       })
       
       print(f"Base dataframe created: {len(df)} records")
       print(f"Age range: {df['age'].min()} - {df['age'].max()}")
       print(f"Sex distribution:\n{df['sex'].value_counts()}")
       print(f"View position distribution:\n{df['view_position'].value_counts()}")
       
       return df
   
   def create_patient_wise_split(self, df: pd.DataFrame, 
                               val_size: float = 0.083, 
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
       """Split data by patient to avoid data leakage"""
       official_test_df = df[df['official_split'] == 'test'].copy()
       train_val_df = df[df['official_split'] == 'train_val'].copy()
       
       train_val_patients = train_val_df['patient_id'].unique()
       
       print(f"Train/val patients: {len(train_val_patients)}")
       print(f"Test patients: {official_test_df['patient_id'].nunique()}")
       
       # Split patients (not images) to avoid data leakage
       train_patients, val_patients = train_test_split(
           train_val_patients,
           test_size=val_size,
           random_state=random_state,
           shuffle=True
       )
       
       print(f"Train patients: {len(train_patients)}")
       print(f"Validation patients: {len(val_patients)}")
       
       train_df = train_val_df[train_val_df['patient_id'].isin(train_patients)].copy()
       val_df = train_val_df[train_val_df['patient_id'].isin(val_patients)].copy()
       test_df = official_test_df.copy()
       
       train_df['final_split'] = 'train'
       val_df['final_split'] = 'val'
       test_df['final_split'] = 'test'
       
       final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
       
       return final_df, train_df, val_df, test_df
   
   def validate_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
       """Validate no patient overlap between splits"""
       train_patients = set(train_df['patient_id'].unique())
       val_patients = set(val_df['patient_id'].unique())
       test_patients = set(test_df['patient_id'].unique())
       
       overlaps = {
           'train_val': len(train_patients & val_patients),
           'train_test': len(train_patients & test_patients),
           'val_test': len(val_patients & test_patients)
       }
       
       if any(overlaps.values()):
           print("Data leakage detected:")
           for overlap_type, count in overlaps.items():
               if count > 0:
                   print(f"  {overlap_type} overlap: {count} patients")
           return False
       
       print("No data leakage detected")
       return True
   
   def generate_statistics(self, final_df: pd.DataFrame) -> Dict:
       """Generate comprehensive dataset statistics"""
       stats = {
           'total_images': len(final_df),
           'total_patients': final_df['patient_id'].nunique(),
           'age_stats': {
               'mean': float(final_df['age'].mean()),
               'std': float(final_df['age'].std()),
               'min': int(final_df['age'].min()),
               'max': int(final_df['age'].max()),
               'median': float(final_df['age'].median())
           },
           'sex_distribution': final_df['sex'].value_counts().to_dict(),
           'view_position_distribution': final_df['view_position'].value_counts().to_dict(),
           'splits': {}
       }
       
       for split in ['train', 'val', 'test']:
           split_df = final_df[final_df['final_split'] == split]
           stats['splits'][split] = {
               'images': len(split_df),
               'patients': split_df['patient_id'].nunique(),
               'age_mean': float(split_df['age'].mean()),
               'age_std': float(split_df['age'].std()),
               'sex_distribution': split_df['sex'].value_counts().to_dict()
           }
       
       return stats
   
   def visualize_distribution(self, final_df: pd.DataFrame):
       """Create comprehensive visualization of dataset distribution"""
       fig = plt.figure(figsize=(18, 12))
       
       # Image distribution across splits
       plt.subplot(3, 3, 1)
       split_counts = final_df['final_split'].value_counts()
       plt.bar(split_counts.index, split_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
       plt.title('Image Distribution Across Splits')
       plt.ylabel('Count')
       
       # Patient distribution across splits
       plt.subplot(3, 3, 2)
       patient_counts = final_df.groupby('final_split')['patient_id'].nunique()
       plt.bar(patient_counts.index, patient_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
       plt.title('Patient Distribution Across Splits')
       plt.ylabel('Count')
       
       # Images per patient distribution
       plt.subplot(3, 3, 3)
       images_per_patient = final_df.groupby('patient_id').size()
       plt.hist(images_per_patient, bins=30, alpha=0.7, edgecolor='black')
       plt.title('Images per Patient Distribution')
       plt.xlabel('Images per Patient')
       plt.ylabel('Frequency')
       
       # Age distribution by split
       plt.subplot(3, 3, 4)
       for split in ['train', 'val', 'test']:
           split_data = final_df[final_df['final_split'] == split]['age']
           plt.hist(split_data, bins=30, alpha=0.5, label=split)
       plt.title('Age Distribution by Split')
       plt.xlabel('Age')
       plt.ylabel('Frequency')
       plt.legend()
       
       # Age distribution overall
       plt.subplot(3, 3, 5)
       plt.hist(final_df['age'], bins=30, edgecolor='black', alpha=0.7)
       plt.title(f"Age Distribution (mean={final_df['age'].mean():.1f}±{final_df['age'].std():.1f})")
       plt.xlabel('Age')
       plt.ylabel('Frequency')
       
       # Sex distribution
       plt.subplot(3, 3, 6)
       sex_counts = final_df['sex'].value_counts()
       plt.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', startangle=90)
       plt.title('Sex Distribution')
       
       # View position distribution
       plt.subplot(3, 3, 7)
       view_counts = final_df['view_position'].value_counts()
       plt.bar(view_counts.index, view_counts.values)
       plt.title('View Position Distribution')
       plt.ylabel('Count')
       plt.xticks(rotation=45)
       
       # Split ratios
       plt.subplot(3, 3, 8)
       split_ratios = final_df['final_split'].value_counts(normalize=True)
       plt.pie(split_ratios.values, labels=split_ratios.index, autopct='%1.1f%%', startangle=90)
       plt.title('Dataset Split Ratios')
       
       # Age boxplot by split
       plt.subplot(3, 3, 9)
       split_ages = [final_df[final_df['final_split'] == split]['age'].values 
                     for split in ['train', 'val', 'test']]
       plt.boxplot(split_ages, labels=['train', 'val', 'test'])
       plt.title('Age Distribution by Split (Boxplot)')
       plt.ylabel('Age')
       
       plt.tight_layout()
       plt.savefig(os.path.join(self.data_root, 'reports', 'data_distribution.png'), 
                  dpi=300, bbox_inches='tight')
       print(f"Visualization saved to {os.path.join(self.data_root, 'reports', 'data_distribution.png')}")
       plt.close()
   
   def save_processed_data(self, final_df: pd.DataFrame):
       """Save processed dataframes to CSV files"""
       # Save full dataset
       output_path = os.path.join(self.data_root, 'processed', 'dataset.csv')
       final_df.to_csv(output_path, index=False)
       print(f"Full dataset saved: {output_path}")
       
       # Save individual splits
       splits = {
           'train': final_df[final_df['final_split'] == 'train'],
           'val': final_df[final_df['final_split'] == 'val'],
           'test': final_df[final_df['final_split'] == 'test']
       }
       
       for split_name, split_df in splits.items():
           split_path = os.path.join(self.data_root, 'processed', f'{split_name}_set.csv')
           split_df.to_csv(split_path, index=False)
           print(f"{split_name.capitalize()} set saved: {split_path}")
   
   def save_statistics_report(self, stats: Dict):
       """Save detailed statistics report"""
       stats_path = os.path.join(self.data_root, 'reports', 'statistics.json')
       with open(stats_path, 'w') as f:
           json.dump(stats, f, indent=2)
       print(f"Statistics report saved: {stats_path}")
       
       # Create text report
       text_report_path = os.path.join(self.data_root, 'reports', 'statistics.txt')
       with open(text_report_path, 'w') as f:
           f.write("=" * 60 + "\n")
           f.write("NIH CHEST X-RAY DATASET STATISTICS\n")
           f.write("=" * 60 + "\n\n")
           
           f.write(f"Total Images: {stats['total_images']}\n")
           f.write(f"Total Patients: {stats['total_patients']}\n\n")
           
           f.write("Age Statistics:\n")
           f.write(f"  Mean: {stats['age_stats']['mean']:.2f} ± {stats['age_stats']['std']:.2f}\n")
           f.write(f"  Range: {stats['age_stats']['min']} - {stats['age_stats']['max']}\n")
           f.write(f"  Median: {stats['age_stats']['median']:.2f}\n\n")
           
           f.write("Sex Distribution:\n")
           for sex, count in stats['sex_distribution'].items():
               f.write(f"  {sex}: {count}\n")
           f.write("\n")
           
           f.write("View Position Distribution:\n")
           for view, count in stats['view_position_distribution'].items():
               f.write(f"  {view}: {count}\n")
           f.write("\n")
           
           f.write("=" * 60 + "\n")
           f.write("SPLIT STATISTICS\n")
           f.write("=" * 60 + "\n\n")
           
           for split_name, split_stats in stats['splits'].items():
               f.write(f"{split_name.upper()} SET:\n")
               f.write(f"  Images: {split_stats['images']}\n")
               f.write(f"  Patients: {split_stats['patients']}\n")
               f.write(f"  Age: {split_stats['age_mean']:.2f} ± {split_stats['age_std']:.2f}\n")
               f.write(f"  Sex Distribution: {split_stats['sex_distribution']}\n\n")
       
       print(f"Text report saved: {text_report_path}")
   
   def create_config_file(self, final_df: pd.DataFrame):
       """Create configuration file for model training"""
       config = {
           'dataset_info': {
               'name': 'NIH Chest X-ray',
               'total_images': len(final_df),
               'total_patients': final_df['patient_id'].nunique(),
               'task': 'age_estimation',
               'target_variable': 'age',
               'image_size': [320, 320],
               'channels': 1,
               'data_root': self.data_root
           },
           'data_splits': {
               'train': {
                   'csv': 'processed/train_set.csv',
                   'images': len(final_df[final_df['final_split'] == 'train']),
                   'patients': final_df[final_df['final_split'] == 'train']['patient_id'].nunique()
               },
               'validation': {
                   'csv': 'processed/val_set.csv',
                   'images': len(final_df[final_df['final_split'] == 'val']),
                   'patients': final_df[final_df['final_split'] == 'val']['patient_id'].nunique()
               },
               'test': {
                   'csv': 'processed/test_set.csv',
                   'images': len(final_df[final_df['final_split'] == 'test']),
                   'patients': final_df[final_df['final_split'] == 'test']['patient_id'].nunique()
               }
           },
           'preprocessing': {
               'normalization': 'imagenet',
               'resize': [320, 320],
               'grayscale': True
           },
           'augmentation': {
               'train': {
                   'random_rotation': [-20, 20],
                   'random_horizontal_flip': False,
                   'random_padding': True,
                   'brightness_contrast': True
               },
               'val_test': {
                   'random_rotation': False,
                   'random_horizontal_flip': False
               }
           }
       }
       
       config_path = os.path.join(self.data_root, 'processed', 'config.json')
       with open(config_path, 'w') as f:
           json.dump(config, f, indent=2)
       
       print(f"Configuration file saved: {config_path}")


def process_nih_dataset(data_root: str, val_size: float = 0.083, random_state: int = 42):
   """Main processing pipeline for NIH Chest X-ray dataset"""
   print("=" * 60)
   print("NIH CHEST X-RAY DATASET PROCESSING")
   print("=" * 60)
   
   processor = NIHDataProcessor(data_root)
   
   try:
       # Create base dataframe with metadata
       print("\nStep 1: Loading data and metadata...")
       base_df = processor.create_base_dataframe()
       
       # Perform patient-wise splitting
       print("\nStep 2: Performing patient-wise data splitting...")
       final_df, train_df, val_df, test_df = processor.create_patient_wise_split(
           base_df, val_size=val_size, random_state=random_state
       )
       
       # Validate split
       print("\nStep 3: Validating data split...")
       processor.validate_split(train_df, val_df, test_df)
       
       # Generate statistics
       print("\nStep 4: Generating dataset statistics...")
       stats = processor.generate_statistics(final_df)
       
       # Print summary
       print("\n" + "=" * 60)
       print("DATASET SUMMARY")
       print("=" * 60)
       print(f"Total images: {stats['total_images']}")
       print(f"Total patients: {stats['total_patients']}")
       print(f"Age: {stats['age_stats']['mean']:.1f} ± {stats['age_stats']['std']:.1f} years")
       print(f"Age range: {stats['age_stats']['min']} - {stats['age_stats']['max']} years")
       print(f"\nTrain set: {stats['splits']['train']['images']} images, {stats['splits']['train']['patients']} patients")
       print(f"Validation set: {stats['splits']['val']['images']} images, {stats['splits']['val']['patients']} patients")
       print(f"Test set: {stats['splits']['test']['images']} images, {stats['splits']['test']['patients']} patients")
       
       # Create visualizations
       print("\nStep 5: Creating data distribution visualizations...")
       processor.visualize_distribution(final_df)
       
       # Save processed data
       print("\nStep 6: Saving processed data...")
       processor.save_processed_data(final_df)
       
       # Save statistics
       processor.save_statistics_report(stats)
       
       # Create config file
       processor.create_config_file(final_df)
       
       print("\n" + "=" * 60)
       print("PROCESSING COMPLETE")
       print("=" * 60)
       print("Output files created in:")
       print(f"  - {os.path.join(data_root, 'processed')}")
       print(f"  - {os.path.join(data_root, 'reports')}")
       
   except Exception as e:
       print(f"\nError during processing: {str(e)}")
       raise


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Process NIH Chest X-ray dataset for age estimation')
   parser.add_argument('--data_root', type=str, required=True,
                      help='Path to the NIH dataset root directory')
   parser.add_argument('--val_size', type=float, default=0.083,
                      help='Validation set size (default: 0.083, ~8.3%%)')
   parser.add_argument('--random_state', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
   
   args = parser.parse_args()
   
   process_nih_dataset(args.data_root, args.val_size, args.random_state)

