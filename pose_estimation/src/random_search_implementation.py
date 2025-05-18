# Random search implementation for hyperparameter optimization
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def random_search_hyperparameter_optimization(
    model_class, input_size, output_size, train_dataset, val_dataset, device, 
    n_trials=20, max_time_seconds=3600):
    """
    Perform random search hyperparameter optimization for the model.
    
    Args:
        model_class: The model class to instantiate
        input_size: Input dimension for the model
        output_size: Output dimension (number of classes)
        train_dataset: PyTorch Dataset for training
        val_dataset: PyTorch Dataset for validation
        device: Device to train on (cuda/cpu)
        n_trials: Maximum number of trials to run
        max_time_seconds: Maximum time in seconds to run the search
    
    Returns:
        best_params: Dictionary of best hyperparameters
        results: List of all trials results
    """
    # Define hyperparameter search spaces
    param_distributions = {
        'learning_rate': (1e-4, 1e-1, 'log'),  # min, max, scale
        'hidden_size': [16, 32, 64],  # categorical values
        'dropout_rate': (0.2, 0.8, 'linear'),  # min, max, scale
        'batch_size': [4, 8, 16, 32, 64],  # categorical values
        'weight_decay': (1e-4, 1e-2, 'log'),  # min, max, scale
        'num_layers': [1, 2, 3],  # categorical values
    }
    
    # Storage for results
    results = []
    best_val_loss = float('inf')
    best_params = {}
    
    # Track time
    start_time = time.time()
    
    for trial in range(n_trials):
        # Check time limit
        if time.time() - start_time > max_time_seconds:
            print(f"Time limit of {max_time_seconds} seconds reached. Stopping search.")
            break
            
        # Sample hyperparameters randomly
        params = {}
        for param_name, param_config in param_distributions.items():
            if isinstance(param_config, list):
                # Categorical parameter
                params[param_name] = int(np.random.choice(param_config))
            else:
                # Continuous parameter
                min_val, max_val, scale = param_config
                if scale == 'log':
                    params[param_name] = float(np.exp(np.random.uniform(np.log(min_val), np.log(max_val))))
                else:
                    params[param_name] = float(np.random.uniform(min_val, max_val))
        # Ensure integer parameters are Python ints
        for k in ['hidden_size', 'num_layers', 'batch_size']:
            params[k] = int(params[k])
        
        print(f"\nTrial {trial+1}/{n_trials}")
        print("Parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        # Create model with the sampled hyperparameters
        model = model_class(
            input_size=input_size, 
            hidden_size=params['hidden_size'], 
            output_size=output_size, 
            num_layers=params['num_layers']
        ).to(device)
        
        # Update dropout rates if the model has dropout layers
        if hasattr(model, 'dropout1'):
            model.dropout1 = nn.Dropout(params['dropout_rate'])
        if hasattr(model, 'dropout2'):
            model.dropout2 = nn.Dropout(params['dropout_rate'])
            
        # Setup optimizer with weight decay
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], 
                               weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=int(params['batch_size']), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=int(params['batch_size']), shuffle=False)
        
        # Training loop with early stopping
        num_epochs = 20  # Fewer epochs for tuning
        patience = 5
        trial_best_val_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Check for early stopping
            if val_loss < trial_best_val_loss:
                trial_best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"Early stopping counter: {early_stop_counter}/{patience}")
                if early_stop_counter >= patience:
                    print("Early stopping triggered.")
                    break
        
        # Record trial results
        trial_result = {
            'params': params,
            'val_loss': trial_best_val_loss
        }
        results.append(trial_result)
        
        # Update best parameters if needed
        if trial_best_val_loss < best_val_loss:
            best_val_loss = trial_best_val_loss
            best_params = params
            print(f"New best validation loss: {best_val_loss:.4f}")
    
    # Print final results
    print("\nRandom Search completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    return best_params, results


def plot_random_search_results(results):
    """
    Plot the results of random search.
    
    Args:
        results: List of dictionaries with trial results
    """
    # Sort results by validation loss
    sorted_results = sorted(results, key=lambda x: x['val_loss'])
    
    # Extract validation losses for plotting
    val_losses = [r['val_loss'] for r in results]
    trial_numbers = list(range(1, len(results) + 1))
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    plt.plot(trial_numbers, val_losses, 'o-')
    plt.axhline(y=min(val_losses), color='r', linestyle='--', label=f'Best Loss: {min(val_losses):.4f}')
    plt.xlabel('Trial Number')
    plt.ylabel('Validation Loss')
    plt.title('Random Search Optimization History')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Create importance-like plot based on rank correlation
    param_names = sorted_results[0]['params'].keys()
    param_importances = {}
    
    for param_name in param_names:
        # Extract parameter values and corresponding validation losses
        param_values = []
        losses = []
        
        for result in results:
            param_values.append(result['params'][param_name])
            losses.append(result['val_loss'])
            
        # Calculate a simple importance metric based on variance of the parameter in top results
        # (This is a simplified approach compared to Optuna's feature importance)
        top_k = max(3, len(results) // 3)  # Take top third of results
        top_values = [sorted_results[i]['params'][param_name] for i in range(top_k)]
        
        # Normalize parameter values if they're not categorical
        if not isinstance(param_values[0], (int, np.integer)) or param_name in ['num_layers']:
            # Normalize values to [0, 1] range for fair comparison
            if max(param_values) > min(param_values):
                top_values = [(v - min(param_values)) / (max(param_values) - min(param_values)) 
                             for v in top_values]
            else:
                top_values = [0.5 for _ in top_values]
                
        # Use inverse of standard deviation as importance (lower variance → more important)
        if len(top_values) > 1:
            importance = 1.0 / (np.std(top_values) + 1e-10)
        else:
            importance = 0
            
        param_importances[param_name] = importance
    
    # Normalize importances
    total_importance = sum(param_importances.values())
    if total_importance > 0:
        param_importances = {k: v / total_importance for k, v in param_importances.items()}
    
    # Plot parameter importances
    plt.figure(figsize=(10, 6))
    names = list(param_importances.keys())
    values = list(param_importances.values())
    
    # Sort by importance
    indices = np.argsort(values)
    names = [names[i] for i in indices]
    values = [values[i] for i in indices]
    
    plt.barh(names, values)
    plt.xlabel('Relative Importance')
    plt.title('Parameter Importances (Estimated)')
    plt.tight_layout()
    plt.show()
