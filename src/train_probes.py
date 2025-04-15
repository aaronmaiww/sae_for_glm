import pandas as pd
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import copy
import utils
from typing import List, Dict, Any, Tuple, Optional


def load_token_dataframes(base_path: str, subset_indices: Tuple =(0,)) -> Dict[str, Any]:
    """Load and combine token dataframes of annotated, tokenized sequences that serve
    as validation sets.

    Args:
        base_path (str): Base path to the directory containing token dataframes
        subset_indices (tuple): Indices of subsets to load (e.g., (0,) or (0,1,2))

    Returns:
        dict: Dictionary of dataframes and sequence IDs for each subset
    """
    dataframes = {}
    sequence_ids = {}

    for idx in subset_indices:
        path = f"{base_path}/token_df_1k_ss{idx}_standardized.pkl"
        try:
            # Load the dataframe and save it in the dictionary
            df = pd.read_pickle(path)
            dataframes[f's{idx}'] = df

            # Extract unique sequence IDs, sort them, and save them in the dictionary
            unique_seq_ids = set(df['seq_id'])
            sorted_seq_ids = sorted(list(unique_seq_ids)) # ascending order
            sequence_ids[f's{idx}'] = sorted_seq_ids

        except FileNotFoundError:
            raise FileNotFoundError(f"Token dataframe not found at {path}")

    return {
        'dataframes': dataframes,
        'sequence_ids': sequence_ids
    }

def get_sequences_from_ids(test_data: pd.DataFrame, seq_ids: List[int]) -> pd.Series:
    """Extract sequences from test data using sequence IDs.

    Args:
        test_data (pd.DataFrame): Full test dataset
        seq_ids (list): List of sequence IDs to extract

    Returns:
        pd.Series: Extracted sequences
    """
    df = test_data.iloc[seq_ids]['sequence']
    return df

def get_model_activations(
    model: nn.Module,
    tokenizer: Any,
    sequences: List[str],
    layer_num: int = 11,
    batch_size: int = 128,
    max_length: int = 512,
    device='cuda'
) -> torch.Tensor:
    
    """Get MLP outputs for a set of sequences.

    Args:
        model: In this case a transformer model (BERT-style)
        tokenizer: Tokenizer for processing sequences
        sequences (list): Input sequences
        layer_num (int): Layer to extract activations from
        batch_size (int): Processing batch size
        max_length (int): Maximum sequence length
        device (str): Device to use for computation

    Returns:
        torch.Tensor: Processed and normalized activations
    """
    # Tokenize sequences
    tokens = tokenizer(
        sequences.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

    # Calculate numb batches for progress bar
    total_tokens = tokens['input_ids'].shape[0] * tokens['input_ids'].shape[1]
    num_batches = (total_tokens + batch_size - 1) // batch_size

    all_acts = [] 

    # We don't want to train the model, just get the activations
    model.eval() 
    with torch.no_grad():
        pbar = tqdm(total=num_batches, desc="Processing batches")
        for i in range(num_batches):
            pbar.update(1)

            # Prepare batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_tokens)
            batch_input_ids = tokens['input_ids'][start_idx:end_idx].to(device)
            batch_attention_mask = tokens['attention_mask'][start_idx:end_idx].to(device)

            # Get MLP-outputs of model at layer_N
            mlp_act = utils.get_layer_activations(
                model.to(device),
                batch_input_ids,
                batch_attention_mask,
                layer_N=layer_num
            )

            assert len(mlp_act) == 1, (
                f"Expected 1 activation tensor, got {len(mlp_act)}"
            )

            # unnest & reshape to combine batch size and sequence length into a single dimension
            mlp_act = mlp_act[0] # there's only a sinle tensor in the list
            mlp_act = mlp_act.reshape(-1, model.config.hidden_size) # reshape tensor(batch_s, seq_len, hidden_d) to tensor(batch_s*seq_len, hidden_d)

            # Verify shape
            expected_shape = batch_input_ids.shape[0] * batch_input_ids.shape[1]
            assert mlp_act.shape[0] == expected_shape, (
                f"MLP activation shape {mlp_act.shape[0]} does not match "
                f"expected shape {expected_shape}"
            )

            all_acts.append(mlp_act)

    # Move activations to CPU and concatenate to one torch.Tensor
    all_acts = [x.cpu() for x in all_acts]
    activations = torch.cat(all_acts, dim=0)

    # Normalize all activation vectors
    normalized_acts = (activations - activations.mean(dim=0)) / activations.std(dim=0)

    return normalized_acts

def prepare_probe_inputs(
    test_data,
    model,
    tokenizer,
    token_df_path,
    layer_num=11,
    subset_indices=(0,),
    batch_size=128
):
    """Prepare inputs for linear probe training.

    Args:
        test_data (pd.DataFrame): Test dataset
        model: Neural network model
        tokenizer: Tokenizer
        token_df_path (str): Path to token dataframes
        layer_num (int): Layer to extract activations from
        subset_indices (tuple): Which subsets to use
        batch_size (int): Batch size for processing

    Returns:
        dict: Processed inputs for probe training
    """
    # Load token dataframes
    token_data = load_token_dataframes(token_df_path, subset_indices)

    # Get sequences by first getting seq_ids for all subset indices
    seq_ids = [token_data['sequence_ids'][f's{subset_indices[i]}'] for i in range(len(subset_indices))]
    seq_ids = [item for sublist in seq_ids for item in sublist]
    sequences = get_sequences_from_ids(test_data, seq_ids)

    # Get activations
    activations = get_model_activations(
        model=model,
        tokenizer=tokenizer,
        sequences=sequences,
        layer_num=layer_num,
        batch_size=batch_size
    )

    return {
        'activations': activations,
        'token_data': token_data,
        'sequences': sequences
    }



def create_binary_labels(df, annotation_column, annotations, target_value=None):
    """Create binary labels based on whether annotations contain any value from a list.

    Args:
        df (pd.DataFrame): DataFrame containing annotations
        annotation_column (str): Name of column containing annotations
        annotations (list or str): List of target annotations to search for
        target_value (str, optional): Alternative target to search for (e.g., 'TAG')

    Returns:
        pd.Series: Binary labels (0 or 1)
    """
    # Convert single string annotation to list for consistent handling
    if isinstance(annotations, str):
        annotations = [annotations]
    elif not isinstance(annotations, list):
        raise ValueError("Annotations must be a string or list of strings")

    # If using token-based labeling
    if target_value:
        print(f"Using token-based labeling with target value: {target_value}")
        return df['tokens'].apply(lambda x: 1 if target_value in x else 0)
    else:
        print(f"Using annotation-based labeling with annotations: {annotations}")
        
        # Fix for list-type annotation columns
        def check_for_annotation(annotation_list):
                
            # Check if any annotation is a substring of any element in the list
            return 1 if any(
                any(ann in list_item for list_item in annotation_list)
                for ann in annotations
            ) else 0
            
        result = df[annotation_column].apply(check_for_annotation)
        print(f"Found {result.sum()} positive examples")
        return result

def compute_class_statistics(labels):
    """Compute basic statistics about class distribution.

    Args:
        labels (pd.Series): Binary labels

    Returns:
        dict: Statistics including positive count, base rate accuracy, class frequency
    """
    positive_count = labels.sum()
    if positive_count == 0:
        return {
            'positive_count': 0,
            'base_rate_accuracy': None,
            'class_frequency': 0
        }

    class_freq = positive_count / len(labels)
    base_rate_acc = max(class_freq, 1 - class_freq)

    return {
        'positive_count': positive_count,
        'base_rate_accuracy': base_rate_acc,
        'class_frequency': class_freq
    }

def prepare_tensors(features, labels, device='cuda'):
    """Prepare feature and label tensors for PyTorch training.

    Args:
        features (torch.Tensor): Input features
        labels (pd.Series): Binary labels
        device (str): Target device for tensors

    Returns:
        tuple: (feature tensor, label tensor)
    """
    x_tensor = features.clone().detach().to(dtype=torch.float32)
    y_tensor = torch.tensor(labels.values, dtype=torch.float32)

    return x_tensor, y_tensor

def process_annotation(df, xs, target_value =None, annotation='CMV enhancer', annotation_column='token_annotations'):
    """Process a single annotation and prepare data for training.

    Args:
        df (pd.DataFrame): Input DataFrame with annotations
        xs (torch.Tensor): Feature tensor
        annotation (str): Target annotation
        annotation_column (str): Column containing annotations

    Returns:
        dict: Processed data and statistics

    Raises:
        ValueError: If no positive examples found for annotation
    """
    # Create labels
    labels = create_binary_labels(df, annotation_column, annotation, target_value=target_value)

    # Compute statistics
    stats = compute_class_statistics(labels)
    if stats['positive_count'] == 0:
        raise ValueError(f"No positive examples found for annotation: {annotation}")

    # Prepare tensors
    x_tensor, y_tensor = prepare_tensors(xs, labels)

    return {
        'features': x_tensor,
        'labels': y_tensor,
        'statistics': stats
    }


def generate_shuffled_label_matrix(original_labels, num_probes, real_probe_idx=0):
    """Generate matrix of shuffled labels for multiple probes
    
    Args:
        original_labels (torch.Tensor): Original binary labels [n_samples]
        num_probes (int): Number of probes to generate labels for
        real_probe_idx (int): Index of the probe to use real (unshuffled) labels
        
    Returns:
        torch.Tensor: Label matrix [n_samples, num_probes]
    """
    n_samples = original_labels.size(0)
    label_matrix = torch.zeros(n_samples, num_probes)
    
    # Set the real labels for one probe
    label_matrix[:, real_probe_idx] = original_labels
    
    # Create shuffled versions for all other probes
    for i in range(num_probes):
        if i == real_probe_idx:
            continue
        # Get shuffled indices while preserving class distribution
        shuffled_indices = torch.randperm(n_samples)
        label_matrix[:, i] = original_labels[shuffled_indices]
    
    return label_matrix



#Training Classes
class ProbeTrainer:
    def __init__(self, model, config, loss_weight = None):
        self.model = model
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss() if loss_weight is None else nn.BCEWithLogitsLoss(pos_weight=loss_weight) ## acounting for class imb
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        # Early stopping parameters
        self.patience = 5  # Number of epochs to wait for improvement
        self.min_delta = 1e-4  # Minimum change in validation F1 to qualify as an improvement


    def create_data_loaders(self, x_tensor, y_tensor):
        """Create train and validation data loaders with stratified split"""
        dataset = TensorDataset(x_tensor, y_tensor)


        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1-self.config['train_split'],
            random_state=self.config['random_seed']
        )
        train_idx, val_idx = next(sss.split(x_tensor, y_tensor))

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        train_loss = 0.0

        for inputs, targets in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs.cuda())
            loss = self.criterion(outputs.squeeze().cpu(), targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss / len(train_loader)

    def evaluate(self, val_loader):
        """Evaluate model on validation set"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs.cuda())
                loss = self.criterion(outputs.squeeze().cpu(), targets)
                val_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs))
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        metrics = {
            'loss': val_loss / len(val_loader),
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds),
            'recall': recall_score(all_targets, all_preds),
            'f1': f1_score(all_targets, all_preds)
        }

        return metrics

    def train(self, x_tensor, y_tensor, verbose = True):
        """Full training loop with validation and early stopping"""
        train_loader, val_loader = self.create_data_loaders(x_tensor, y_tensor)
        best_f1 = 0.0
        best_metrics = None
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)

            # Check if current F1 score is better than best F1
            if metrics['f1'] > best_f1 + self.min_delta:
                best_f1 = metrics['f1']
                best_metrics = metrics
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose:
                print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Metrics: {metrics}")
                print(f"Best F1: {best_f1:.4f}")
                print(f"Patience Counter: {patience_counter}/{self.patience}")

            # Early stopping check
            if patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        print("Training complete")
        print(f"Best F1: {best_f1:.4f}")
        print(f"Best Metrics: {best_metrics}")

        return best_metrics




class BatchProbeTrainer:
    def __init__(self, num_probes: int, input_dim: int, config: dict, loss_weights=None):
        """Initialize trainer for multiple probes at once"""
        self.config = config
        self.num_probes = num_probes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a single linear layer with multiple outputs
        self.model = nn.Linear(input_dim, num_probes).to(self.device)
            
        # Handle class imbalance with probe-specific weights
        if loss_weights is not None:
            # Make sure loss_weights is on the correct device
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(self.device))
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Early stopping parameters
        self.patience = 5
        self.min_delta = 1e-4

    def create_data_loaders(self, x_tensor, y_tensor_matrix):
        """Create train and validation data loaders with stratified split"""
        # Move tensors to CPU for dataset creation (DataLoader will handle device transfer)
        x_cpu = x_tensor.cpu()
        y_cpu = y_tensor_matrix.cpu()
        
        dataset = TensorDataset(x_cpu, y_cpu)
        
        # Create validation split
        n_samples = x_cpu.shape[0]
        indices = torch.randperm(n_samples)
        split = int(n_samples * self.config['train_split'])
        
        train_indices = indices[:split]
        val_indices = indices[split:]
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        return train_loader, val_loader

    def train_epoch(self, train_loader):
        """Train all probes for one epoch"""
        self.model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            # Explicitly move batch data to the right device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
                
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            
        return train_loss / len(train_loader)

    def evaluate(self, val_loader):
        """Evaluate all probes on validation set"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Explicitly move batch data to the right device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                    
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                
                # Move back to CPU for metric calculation
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
        # Concatenate batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics for each probe
        metrics = []
        for i in range(self.num_probes):
            probe_preds = all_preds[:, i].float()
            probe_targets = all_targets[:, i].float()
            
            metrics.append({
                'accuracy': accuracy_score(probe_targets, probe_preds),
                'precision': precision_score(probe_targets, probe_preds),
                'recall': recall_score(probe_targets, probe_preds),
                'f1': f1_score(probe_targets, probe_preds)
            })
            
        return {
            'loss': val_loss / len(val_loader),
            'probe_metrics': metrics,
            'avg_f1': sum(m['f1'] for m in metrics) / len(metrics)
        }

    def train(self, x_tensor, y_tensor_matrix, verbose=True):
        """Train all probes with early stopping based on average F1"""
        
        train_loader, val_loader = self.create_data_loaders(x_tensor, y_tensor_matrix)
        best_avg_f1 = 0.0
        best_metrics = None
        best_model_state = None
        patience_counter = 0
        
        # Create progress bar for epochs
        pbar = tqdm(total=self.config['num_epochs'], desc="Training", 
                    disable=not verbose, ncols=100)
        
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            
            # Check if average F1 is better
            if metrics['avg_f1'] > best_avg_f1 + self.min_delta:
                best_avg_f1 = metrics['avg_f1']
                best_metrics = metrics
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'loss': f"{train_loss:.4f}",
                'val_loss': f"{metrics['loss']:.4f}", 
                'f1': f"{metrics['avg_f1']:.4f}",
                'patience': f"{patience_counter}/{self.patience}"
            })
            pbar.update(1)
            
            # Additional detailed output if verbose
            if verbose and epoch % 5 == 0:  # Show detailed metrics every 5 epochs
                print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}:")
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {metrics['loss']:.4f}")
                print(f"Avg F1: {metrics['avg_f1']:.4f}, Best F1: {best_avg_f1:.4f}")
            
            # Early stopping check
            if patience_counter >= self.patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Close progress bar
        pbar.close()
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        if verbose:
            print(f"Training complete. Best avg F1: {best_avg_f1:.4f}")
        
        # Extract individual probe weights
        probe_weights = self.model.weight.data
        
        return {
            'metrics': best_metrics,
            'probe_weights': probe_weights,
            'model': self.model
        }