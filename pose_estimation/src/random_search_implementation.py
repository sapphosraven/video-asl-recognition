# Random search implementation for hyperparameter optimization
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm.notebook import tqdm
import itertools

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
        
        #print(f"\nTrial {trial+1}/{n_trials}")
        #print("Parameters:")
        #for k, v in params.items():
        #    print(f"  {k}: {v}")
        
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
            
            #print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Check for early stopping
            if val_loss < trial_best_val_loss:
                trial_best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                #print(f"Early stopping counter: {early_stop_counter}/{patience}")
                if early_stop_counter >= patience:
                    #print("Early stopping triggered.")
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
            #print(f"New best validation loss: {best_val_loss:.4f}")
    
    # Print final results
    print("\nRandom Search completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    return best_params, results


def grid_search_hyperparameter_optimization(
    model_class, input_size, output_size, train_dataset, val_dataset, device,
    param_grid, max_time_seconds=3600, num_epochs=5):
    """
    Perform grid search hyperparameter optimization for the model.
    
    Args:
        model_class: The model class to instantiate
        input_size: Input dimension for the model
        output_size: Output dimension (number of classes)
        train_dataset: PyTorch Dataset for training
        val_dataset: PyTorch Dataset for validation
        device: Device to train on (cuda/cpu)
        param_grid: Dictionary of parameter lists (the grid)
        max_time_seconds: Maximum time in seconds to run the search
    
    Returns:
        best_params: Dictionary of best hyperparameters
        results: List of all trials results
    """
    # Generate all combinations of hyperparameters
    keys = list(param_grid.keys())
    all_combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    results = []
    best_val_loss = float('inf')
    best_params = {}
    start_time = time.time()
    
    for i, values in enumerate(all_combinations):
        # Check time limit
        if time.time() - start_time > max_time_seconds:
            print(f"Time limit of {max_time_seconds} seconds reached. Stopping search.")
            break
        
        params = dict(zip(keys, values))
        
        print(f"\nGrid Search Trial {i+1}/{len(all_combinations)}")
        print("Parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        # Create model with the sampled hyperparameters
        model = model_class(
            input_size=input_size, 
            hidden_size=int(params['hidden_size']), 
            output_size=output_size, 
            num_layers=int(params['num_layers'])
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
        num_epochs = num_epochs  # Fewer epochs for tuning
        patience = 5
        trial_best_val_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            # Train without showing epoch progress
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                # Support datasets that return (inputs, targets, mask) or just (inputs, targets)
                if len(batch) == 3:
                    inputs, targets, mask = batch
                else:
                    inputs, targets = batch
                    mask = None
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, mask) if mask is not None else model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        inputs, targets, mask = batch
                    else:
                        inputs, targets = batch
                        mask = None
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs, mask) if mask is not None else model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            # Only update best loss and early stopping, do not print per-epoch results
            if val_loss < trial_best_val_loss:
                trial_best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    break
                    
        # Print only the final result for this trial
        print(f"Trial result - Val Loss: {trial_best_val_loss:.4f}")
        
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
    
    print("\nGrid Search completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    return best_params, results
