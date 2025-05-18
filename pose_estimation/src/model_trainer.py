import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold
import copy
import os
from pathlib import Path


class ModelTrainer:
    """
    A comprehensive trainer class for ASL recognition models.
    Includes advanced training techniques like learning rate scheduling,
    early stopping, regularization, and model checkpointing.
    """
    
    def __init__(self, model, train_dataset, val_dataset=None,
                 criterion=None, optimizer=None, 
                 device=None, batch_size=32, patience=10, 
                 checkpoint_dir='./checkpoints'):
        """
        Initialize the trainer with model and training settings.
        
        Args:
            model: The PyTorch model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset. If None, will use a portion of train_dataset
            criterion: Loss function (defaults to CrossEntropyLoss)
            optimizer: Optimizer (defaults to Adam)
            device: Device to train on (defaults to cuda if available, else cpu)
            batch_size: Batch size for training
            patience: Number of epochs to wait before early stopping
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.patience = patience
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.001)
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create data loaders
        if self.val_dataset is None:
            # Split training data for validation
            train_size = int(0.8 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size]
            )
            
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def get_lr_scheduler(self, scheduler_type='cosine', **kwargs):
        """
        Create a learning rate scheduler.
        
        Args:
            scheduler_type: Type of scheduler ('step', 'cosine', 'plateau', etc)
            **kwargs: Arguments for the specific scheduler
            
        Returns:
            PyTorch learning rate scheduler
        """
        if scheduler_type == 'step':
            step_size = kwargs.get('step_size', 5)
            gamma = kwargs.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        elif scheduler_type == 'cosine':
            t_max = kwargs.get('t_max', 10)
            eta_min = kwargs.get('eta_min', 0)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
        
        elif scheduler_type == 'plateau':
            mode = kwargs.get('mode', 'min')
            factor = kwargs.get('factor', 0.1)
            patience = kwargs.get('patience', 3)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=mode, factor=factor, patience=patience, verbose=True
            )
            
        elif scheduler_type == 'one_cycle':
            max_lr = kwargs.get('max_lr', 0.01)
            epochs = kwargs.get('epochs', 10)
            steps_per_epoch = len(self.train_loader)
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch
            )
            
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def train_epoch(self, epoch, epochs, use_mixup=False, alpha=0.2):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            epochs: Total number of epochs
            use_mixup: Whether to use mixup data augmentation
            alpha: Alpha parameter for mixup
            
        Returns:
            Average training loss and accuracy for the epoch
        """
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Apply mixup if enabled
            if use_mixup:
                inputs, targets_a, targets_b, lam = self._mixup_data(inputs, targets, alpha)
                
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss (with mixup if enabled)
            if use_mixup:
                loss = self._mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = self.criterion(outputs, targets)
                
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            
            if use_mixup:
                # For accuracy calculation with mixup, we'll use the primary target
                total += targets_a.size(0)
                correct += (predicted == targets_a).sum().item()
            else:
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        train_loss /= len(self.train_loader)
        train_acc = correct / total
        
        return train_loss, train_acc
    
    def validate(self):
        """
        Validate the model on the validation set.
        
        Returns:
            Validation loss and accuracy
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.val_loader, desc="Validation")):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        val_loss /= len(self.val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(self, epochs=50, scheduler_type=None, scheduler_params=None, 
              use_mixup=False, mixup_alpha=0.2, use_early_stopping=True):
        """
        Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train
            scheduler_type: Type of learning rate scheduler
            scheduler_params: Parameters for the scheduler
            use_mixup: Whether to use mixup data augmentation
            mixup_alpha: Alpha parameter for mixup
            use_early_stopping: Whether to use early stopping
            
        Returns:
            Trained model and training history
        """
        # Create scheduler if specified
        scheduler = None
        if scheduler_type:
            scheduler_params = scheduler_params or {}
            scheduler = self.get_lr_scheduler(scheduler_type, epochs=epochs, **scheduler_params)
        
        # Setup for early stopping
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_weights = None
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(
                epoch, epochs, use_mixup=use_mixup, alpha=mixup_alpha
            )
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print metrics
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Update learning rate if using scheduler
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # Save the best model weights
                best_model_weights = copy.deepcopy(self.model.state_dict())
                # Save checkpoint
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=True)
                print(f"Model improved! Best val loss: {best_val_loss:.4f}")
            else:
                early_stop_counter += 1
                # Save regular checkpoint
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=False)
                print(f"Early stopping counter: {early_stop_counter}/{self.patience}")
                
            # Check for early stopping
            if use_early_stopping and early_stop_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load the best model weights
        if best_model_weights:
            self.model.load_state_dict(best_model_weights)
            
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes")
        
        return self.model, self.history
    
    def _mixup_data(self, x, y, alpha=1.0):
        """
        Applies mixup augmentation to the batch.
        
        Args:
            x: Input features
            y: Target labels
            alpha: Mixup interpolation strength
            
        Returns:
            Mixed inputs, targets_a, targets_b, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def _mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """
        Computes the mixup loss.
        
        Args:
            criterion: Loss function
            pred: Model predictions
            y_a: First set of targets
            y_b: Second set of targets
            lam: Mixup interpolation factor
            
        Returns:
            Mixup loss
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def _save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            val_acc: Validation accuracy
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # If this is the best model, also save it separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a saved checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Epoch number
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']
    
    def plot_history(self):
        """
        Plot training and validation loss and accuracy.
        
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.history['train_loss'], label='Training Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid()
        
        # Plot accuracies
        ax2.plot(self.history['train_acc'], label='Training Acc')
        ax2.plot(self.history['val_acc'], label='Validation Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid()
        
        plt.tight_layout()
        
        return fig
    
    def k_fold_cross_validation(self, k=5, epochs=50, scheduler_type=None, 
                                scheduler_params=None, use_mixup=False):
        """
        Perform k-fold cross-validation.
        
        Args:
            k: Number of folds
            epochs: Number of epochs per fold
            scheduler_type: Type of learning rate scheduler
            scheduler_params: Parameters for the scheduler
            use_mixup: Whether to use mixup data augmentation
            
        Returns:
            List of validation losses and accuracies for each fold
        """
        # Set up k-fold cross validation
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        # Convert train_dataset to numpy arrays for splitting
        if isinstance(self.train_dataset, Dataset):
            # Get all data
            all_data = []
            all_targets = []
            for i in range(len(self.train_dataset)):
                data, target = self.train_dataset[i]
                all_data.append(data)
                all_targets.append(target)
            
            all_data = torch.stack(all_data)
            all_targets = torch.tensor(all_targets)
            
        fold_results = []
        
        # Loop through folds
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
            print(f"\nFold {fold+1}/{k}")
            
            # Create train and validation subsets
            train_subset = torch.utils.data.TensorDataset(
                all_data[train_idx], all_targets[train_idx]
            )
            val_subset = torch.utils.data.TensorDataset(
                all_data[val_idx], all_targets[val_idx]
            )
            
            # Reset model for each fold
            self.model = self._reset_model()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Update data loaders
            self.train_loader = DataLoader(
                train_subset, batch_size=self.batch_size, shuffle=True
            )
            self.val_loader = DataLoader(
                val_subset, batch_size=self.batch_size, shuffle=False
            )
            
            # Train the model
            trained_model, history = self.train(
                epochs=epochs, 
                scheduler_type=scheduler_type, 
                scheduler_params=scheduler_params,
                use_mixup=use_mixup
            )
            
            # Save fold results
            val_loss = min(history['val_loss'])
            val_acc = max(history['val_acc'])
            fold_results.append({
                'fold': fold + 1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'model_weights': copy.deepcopy(trained_model.state_dict())
            })
            
            # Save fold model
            torch.save(
                trained_model.state_dict(), 
                self.checkpoint_dir / f"fold{fold+1}_model.pt"
            )
            
        # Calculate average results
        avg_val_loss = np.mean([r['val_loss'] for r in fold_results])
        avg_val_acc = np.mean([r['val_acc'] for r in fold_results])
        
        # Print results summary
        print("\n===== Cross-validation Results =====")
        for fold_result in fold_results:
            print(f"Fold {fold_result['fold']}: Val Loss = {fold_result['val_loss']:.4f}, "
                  f"Val Acc = {fold_result['val_acc']:.4f}")
        print(f"Average: Val Loss = {avg_val_loss:.4f}, Val Acc = {avg_val_acc:.4f}")
        
        # Find the best fold and save its model as the best overall model
        best_fold = min(fold_results, key=lambda x: x['val_loss'])
        torch.save(
            best_fold['model_weights'],
            self.checkpoint_dir / "best_cv_model.pt"
        )
        print(f"Best model from fold {best_fold['fold']} saved.")
        
        return fold_results
    
    def _reset_model(self):
        """
        Reset the model to its initial state.
        
        Returns:
            A new instance of the model
        """
        # This is a simplified version - in practice, you'd want to recreate the model
        # with the same architecture and hyperparameters
        model_class = type(self.model)
        if hasattr(self.model, '_init_args'):
            # If you store initialization arguments
            new_model = model_class(**self.model._init_args)
        else:
            # Default case - this might not work for all models
            new_model = model_class.__new__(model_class)
            new_model.__dict__.update({k: v for k, v in self.model.__dict__.items() 
                                     if k not in ['parameters', '_parameters', 
                                                 '_buffers', '_modules']})
        new_model = new_model.to(self.device)
        return new_model
