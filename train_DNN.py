import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import argparse
import warnings
warnings.filterwarnings('ignore')


class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_path, transform=None, augment=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.augment = augment
        
        # Filter out invalid ages
        self.df = self.df[(self.df['age'] >= 0) & (self.df['age'] <= 120)]
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        print(f"Age range: {self.df['age'].min()}-{self.df['age'].max()}, mean: {self.df['age'].mean():.2f}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = row['image_path']
        image = Image.open(img_path).convert('L')  # Grayscale
        
        # Convert to RGB for pretrained models
        image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get age label
        age = torch.tensor(row['age'], dtype=torch.float32)
        
        return image, age


def get_transforms(img_size=320, augment=False):
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


class BaseAgeEstimator(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, dropout=0.5):
        super(BaseAgeEstimator, self).__init__()
        self.backbone_name = backbone_name
        self.backbone = self._create_backbone(backbone_name, pretrained)
        self.num_features = self._get_num_features()
        self.head = self._create_regression_head(dropout)
    
    def _create_backbone(self, backbone_name, pretrained):
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            backbone.fc = nn.Identity()  # Remove final FC layer
        elif backbone_name == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
            backbone.fc = nn.Identity()
        elif backbone_name == 'resnet152':
            backbone = models.resnet152(pretrained=pretrained)
            backbone.fc = nn.Identity()
        elif backbone_name == 'densenet121':
            backbone = models.densenet121(pretrained=pretrained)
            backbone.classifier = nn.Identity()
        elif backbone_name == 'densenet169':
            backbone = models.densenet169(pretrained=pretrained)
            backbone.classifier = nn.Identity()
        elif backbone_name == 'densenet201':
            backbone = models.densenet201(pretrained=pretrained)
            backbone.classifier = nn.Identity()
        elif backbone_name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(pretrained=pretrained)
            backbone.classifier = nn.Identity()
        elif backbone_name == 'efficientnet_b3':
            backbone = models.efficientnet_b3(pretrained=pretrained)
            backbone.classifier = nn.Identity()
        elif backbone_name == 'vgg16':
            backbone = models.vgg16(pretrained=pretrained)
            backbone.classifier = nn.Identity()
        elif backbone_name == 'vgg19':
            backbone = models.vgg19(pretrained=pretrained)
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return backbone
    
    def _get_num_features(self):
        if 'resnet' in self.backbone_name:
            return 2048 if self.backbone_name in ['resnet101', 'resnet152'] else 2048
        elif 'densenet121' in self.backbone_name:
            return 1024
        elif 'densenet169' in self.backbone_name:
            return 1664
        elif 'densenet201' in self.backbone_name:
            return 1920
        elif 'efficientnet_b0' in self.backbone_name:
            return 1280
        elif 'efficientnet_b3' in self.backbone_name:
            return 1536
        elif 'vgg' in self.backbone_name:
            return 25088  # VGG features before classifier
        else:
            # Default: probe the model
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                features = self.backbone(dummy_input)
            return features.shape[1]
    
    def _create_regression_head(self, dropout):
        head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1)
        )
        return head
    
    def forward(self, x):
        features = self.backbone(x)
        # Handle VGG which outputs 4D tensor
        if len(features.shape) == 4:
            features = features.view(features.size(0), -1)
        output = self.head(features)
        return output.squeeze(-1)


def calculate_comprehensive_metrics(targets, predictions):
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    pearson_r, pearson_p = pearsonr(targets, predictions)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p
    }
    
    return metrics


class AgeEstimationTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup loss and optimizer
        self.criterion = nn.L1Loss()  # MAE loss
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': [],
            'val_r2': [],
            'val_pearson': [],
            'learning_rate': []
        }
        
        self.best_val_mae = float('inf')
        self.best_epoch = 0
        self.best_model_predictions = None
        self.best_model_targets = None
        
        # Create output directory
        os.makedirs(config['save_dir'], exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, ages in pbar:
            images = images.to(self.device)
            ages = ages.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, ages)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_preds.extend(predictions.detach().cpu().numpy())
            all_targets.extend(ages.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        mae = mean_absolute_error(all_targets, all_preds)
        
        return avg_loss, mae
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, ages in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                ages = ages.to(self.device)
                
                predictions = self.model(images)
                loss = self.criterion(predictions, ages)
                
                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(ages.cpu().numpy())
        
        # Calculate comprehensive metrics for validation
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        
        avg_loss = total_loss / len(self.val_loader)
        mae = mean_absolute_error(all_targets, all_preds)
        mse = mean_squared_error(all_targets, all_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_preds)
        pearson_r, _ = pearsonr(all_targets, all_preds)
        
        return avg_loss, mae, mse, rmse, r2, pearson_r, all_preds, all_targets

    
    def train(self, num_epochs):
        print(f"Training on device: {self.device}")
        print(f"Model: {self.config['backbone']}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Update history structure to include MSE and RMSE
        self.history = {
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': [],
            'val_mse': [],
            'val_rmse': [],
            'val_r2': [],
            'val_pearson': [],
            'learning_rate': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_mae = self.train_epoch()
            
            # Validate - now returns additional metrics
            val_loss, val_mae, val_mse, val_rmse, val_r2, val_pearson, val_preds, val_targets = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_mae)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history with new metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_mse'].append(val_mse)
            self.history['val_rmse'].append(val_rmse)
            self.history['val_r2'].append(val_r2)
            self.history['val_pearson'].append(val_pearson)
            self.history['learning_rate'].append(current_lr)
            
            # Print comprehensive metrics
            print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            print(f"Val MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}")
            print(f"Val R²: {val_r2:.4f}, Val Pearson r: {val_pearson:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_epoch = epoch + 1
                self.best_model_predictions = val_preds
                self.best_model_targets = val_targets
                self.save_checkpoint('best_model.pth', epoch, val_mae)
                print(f"✓ New best model saved (MAE: {val_mae:.4f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_mae)
            
            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_training_history()
                self.plot_prediction_scatter(val_targets, val_preds, epoch+1)
        
        print(f"\nTraining completed!")
        print(f"Best model at epoch {self.best_epoch} with MAE: {self.best_val_mae:.4f}")
        
        # Calculate comprehensive metrics for best model
        self.calculate_and_save_best_metrics()
        
        # Save final history
        self.save_training_history()

    
    def calculate_and_save_best_metrics(self):
        if self.best_model_predictions is None or self.best_model_targets is None:
            print("Warning: Best model predictions not available")
            return
        
        print("\n" + "=" * 60)
        print("BEST MODEL COMPREHENSIVE METRICS")
        print("=" * 60)
        
        metrics = calculate_comprehensive_metrics(
            self.best_model_targets, 
            self.best_model_predictions
        )
        
        # Print metrics
        print(f"Mean Absolute Error (MAE):     {metrics['MAE']:.4f}")
        print(f"Mean Squared Error (MSE):      {metrics['MSE']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
        print(f"R² Score:                      {metrics['R2']:.4f}")
        print(f"Pearson Correlation (r):       {metrics['Pearson_r']:.4f}")
        print(f"Pearson p-value:               {metrics['Pearson_p']:.4e}")
        print("=" * 60)
        
        # Save metrics to JSON
        metrics_path = os.path.join(self.config['save_dir'], 'best_model_metrics.json')
        metrics_dict = {
            'epoch': self.best_epoch,
            'metrics': {k: float(v) for k, v in metrics.items()}
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"\nBest model metrics saved to {metrics_path}")
        
        # Create detailed metrics visualization
        self.plot_best_model_analysis(metrics)
    
    def plot_best_model_analysis(self, metrics):
        fig = plt.figure(figsize=(18, 12))
        
        targets = np.array(self.best_model_targets)
        predictions = np.array(self.best_model_predictions)
        errors = predictions - targets
        
        # Scatter plot with regression line
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(targets, predictions, alpha=0.5, s=10)
        min_age = min(targets.min(), predictions.min())
        max_age = max(targets.max(), predictions.max())
        ax1.plot([min_age, max_age], [min_age, max_age], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Age (years)')
        ax1.set_ylabel('Predicted Age (years)')
        ax1.set_title('Prediction vs Actual Age')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Prediction Error (years)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Error Distribution (Mean: {errors.mean():.2f}±{errors.std():.2f})')
        ax2.grid(True, alpha=0.3)
        
        # Bland-Altman plot
        ax3 = plt.subplot(2, 3, 3)
        mean_ages = (targets + predictions) / 2
        ax3.scatter(mean_ages, errors, alpha=0.5, s=10)
        ax3.axhline(y=0, color='r', linestyle='-', linewidth=2)
        ax3.axhline(y=errors.mean() + 1.96*errors.std(), color='b', linestyle='--', linewidth=1.5, label='±1.96 SD')
        ax3.axhline(y=errors.mean() - 1.96*errors.std(), color='b', linestyle='--', linewidth=1.5)
        ax3.set_xlabel('Mean Age (years)')
        ax3.set_ylabel('Prediction Error (years)')
        ax3.set_title('Bland-Altman Plot')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Absolute error by age group
        ax4 = plt.subplot(2, 3, 4)
        age_bins = [0, 20, 40, 60, 80, 120]
        age_groups = pd.cut(targets, bins=age_bins)
        abs_errors = np.abs(errors)
        df_temp = pd.DataFrame({'age_group': age_groups, 'abs_error': abs_errors})
        grouped = df_temp.groupby('age_group')['abs_error'].mean()
        grouped.plot(kind='bar', ax=ax4)
        ax4.set_xlabel('Age Group')
        ax4.set_ylabel('Mean Absolute Error (years)')
        ax4.set_title('MAE by Age Group')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Metrics summary table
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        metrics_text = (
            f"Best Model Performance Metrics\n"
            f"{'='*40}\n"
            f"Epoch: {self.best_epoch}\n\n"
            f"MAE:  {metrics['MAE']:.4f} years\n"
            f"MSE:  {metrics['MSE']:.4f}\n"
            f"RMSE: {metrics['RMSE']:.4f} years\n"
            f"R²:   {metrics['R2']:.4f}\n"
            f"r:    {metrics['Pearson_r']:.4f}\n"
            f"p:    {metrics['Pearson_p']:.4e}\n"
        )
        ax5.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Error percentiles
        ax6 = plt.subplot(2, 3, 6)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        abs_error_percentiles = [np.percentile(abs_errors, p) for p in percentiles]
        ax6.plot(percentiles, abs_error_percentiles, marker='o', linewidth=2, markersize=8)
        ax6.set_xlabel('Percentile')
        ax6.set_ylabel('Absolute Error (years)')
        ax6.set_title('Error Percentiles')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['save_dir'], 'best_model_analysis.png'), dpi=300)
        plt.close()
        
        print(f"Best model analysis plot saved")
    
    def save_checkpoint(self, filename, epoch, val_mae):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_mae': val_mae,
            'history': self.history,
            'config': self.config
        }
        
        path = os.path.join(self.config['save_dir'], filename)
        torch.save(checkpoint, path)
    
    def plot_training_history(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE curves
        axes[0, 1].plot(self.history['train_mae'], label='Train MAE')
        axes[0, 1].plot(self.history['val_mae'], label='Val MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (years)')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MSE and RMSE curves
        axes[0, 2].plot(self.history['val_mse'], label='Val MSE', color='purple')
        axes[0, 2].plot(self.history['val_rmse'], label='Val RMSE', color='brown')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Error')
        axes[0, 2].set_title('MSE and RMSE')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # R² curve
        axes[1, 0].plot(self.history['val_r2'], label='Val R²', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Validation R² Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Pearson correlation
        axes[1, 1].plot(self.history['val_pearson'], label='Val Pearson r', color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Pearson r')
        axes[1, 1].set_title('Validation Pearson Correlation')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Learning rate
        axes[1, 2].plot(self.history['learning_rate'], label='Learning Rate', color='red')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['save_dir'], 'training_history.png'), dpi=300)
        plt.close()

    
    def plot_prediction_scatter(self, targets, predictions, epoch):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.scatter(targets, predictions, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_age = min(min(targets), min(predictions))
        max_age = max(max(targets), max(predictions))
        ax.plot([min_age, max_age], [min_age, max_age], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Age (years)', fontsize=12)
        ax.set_ylabel('Predicted Age (years)', fontsize=12)
        ax.set_title(f'Age Prediction vs Actual (Epoch {epoch})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add metrics text
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        pearson_r, _ = pearsonr(targets, predictions)
        
        text = f'MAE: {mae:.2f} years\nR²: {r2:.3f}\nPearson r: {pearson_r:.3f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['save_dir'], f'predictions_epoch_{epoch}.png'), dpi=300)
        plt.close()
    
    def save_training_history(self):
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                # Convert each element in list to Python native type
                history_serializable[key] = [float(v) if hasattr(v, 'item') else v for v in value]
            else:
                history_serializable[key] = value
        
        history_path = os.path.join(self.config['save_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"Training history saved to {history_path}")




def main(args):
    # Load config
    config = {
        'data_root': args.data_root,
        'backbone': args.backbone,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'img_size': args.img_size,
        'num_workers': args.num_workers,
        'dropout': args.dropout,
        'save_dir': args.save_dir
    }
    
    print("=" * 60)
    print("AGE ESTIMATION TRAINING")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Setup data paths
    train_csv = os.path.join(args.data_root, 'processed', 'train_set.csv')
    val_csv = os.path.join(args.data_root, 'processed', 'val_set.csv')
    
    # Create datasets
    print("Loading datasets...")
    train_transform = get_transforms(img_size=args.img_size, augment=True)
    val_transform = get_transforms(img_size=args.img_size, augment=False)
    
    train_dataset = NIHChestXrayDataset(train_csv, transform=train_transform, augment=True)
    val_dataset = NIHChestXrayDataset(val_csv, transform=val_transform, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\nInitializing {args.backbone} model...")
    model = BaseAgeEstimator(
        backbone_name=args.backbone,
        pretrained=True,
        dropout=args.dropout
    )
    
    # Create trainer
    trainer = AgeEstimationTrainer(model, train_loader, val_loader, config)
    
    # Start training
    print("\nStarting training...")
    trainer.train(args.num_epochs)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN for age estimation on NIH Chest X-ray dataset')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to NIH dataset root directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101', 'resnet152', 
                               'densenet121', 'densenet169', 'densenet201',
                               'efficientnet_b0', 'efficientnet_b3',
                               'vgg16', 'vgg19'],
                       help='Backbone architecture (default: resnet50)')
    parser.add_argument('--img_size', type=int, default=320,
                       help='Input image size (default: 320)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    args = parser.parse_args()
    
    main(args)



