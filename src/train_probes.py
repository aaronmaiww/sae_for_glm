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


def get_model_activations(
    model: nn.Module,
    tokenizer: Any,
    sequences: pd.Series,
    layer_num: int = 11,
    batch_size: int = 128,
    max_length: int = 512,
    device='cuda'
) -> torch.Tensor:
    
    """Get MLP outputs for a set of input sequences.

    Args:
        model: In this case a transformer model (BERT-style)
        tokenizer: Tokenizer for processing sequences
        sequences (pd.Series): Input sequences
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

    # Calculate num batches for progress bar
    n_seqs, _ = tokens["input_ids"].shape
    total_batches = (n_seqs + batch_size - 1) // batch_size # round up (n_seqs / batch_size)

    all_acts = [] 

    # We don't want to train the model, just get the activations
    model.to(device)
    model.eval() 
    with torch.no_grad():

        pbar = tqdm(total = total_batches, desc = "Processing batches")

        for i in range(0, n_seqs, batch_size):
            # Update progress bar
            pbar.update(1)

            # Get batch of input IDs and attention masks
            batch_input_ids = tokens['input_ids'][i:i + batch_size].to(device)
            batch_attention_mask = tokens['attention_mask'][i:i + batch_size].to(device)
    
            # Get MLP-outputs of model at layer_N
            mlp_act = utils.get_layer_activations(
                model,
                batch_input_ids,
                batch_attention_mask,
                layer_N=layer_num
            )

            assert len(mlp_act) == 1, (
                f"Expected 1 activation tensor, got {len(mlp_act)}"
            )

            # unnest & reshape to combine batch size and sequence length into a single dimension
            mlp_act = mlp_act[0] # there's only a sinle tensor in the list
            mlp_flat_act = mlp_act.reshape(-1, model.config.hidden_size) # reshape tensor(batch_s, seq_len, hidden_d) to tensor(batch_s*seq_len, hidden_d)
            mlp_flat_act = mlp_flat_act.detach().cpu() # move to CPU

            # Verify shape
            expected_shape = batch_input_ids.shape[0] * batch_input_ids.shape[1]
            assert mlp_flat_act.shape[0] == expected_shape, (
                f"MLP activation shape {mlp_flat_act.shape[0]} does not match "
                f"expected shape {expected_shape}"
            )

            all_acts.append(mlp_flat_act)
        
        pbar.close()

    # Concatenate acts to one torch.Tensor
    activations = torch.cat(all_acts, dim=0)

    # Normalize all activation vectors
    std = activations.std(dim=0).clamp(min=1e-6) # avoid division by zero
    normalized_acts = (activations - activations.mean(dim = 0)) / std

    return normalized_acts

def prepare_probe_inputs(
    test_data: pd.DataFrame,
    model: nn.Module,
    tokenizer: Any,
    token_df_path: str,
    layer_num: int = 11,
    subset_indices: tuple =(0,),
    batch_size: int = 128
) -> Dict[str, Any]:
    
    """Prepare inputs for linear probe training.

    Args:
        test_data (pd.DataFrame): Test dataset
        model: transformer model
        tokenizer: Tokenizer
        token_df_path (str): Path to token dataframes
        layer_num (int): Layer to extract activations from
        subset_indices (tuple): Which subsets to use
        batch_size (int): Batch size for processing

    Returns:
        dict: Processed inputs for probe training
    """
    # Load one or more dataframes of annotated, tokenized DNA sequences
    token_data = load_token_dataframes(token_df_path, subset_indices)
    
    # To get activations for each token in token_data, we need to get the 
    # corresponding sequences they are from (stored in test_data)
    seq_ids = get_seq_ids_from_token_data(token_data, subset_indices)
    sequences = test_data.iloc[seq_ids]['sequence']

    # Get MLP-outputs at layer_num for the sequences
    activations = get_model_activations(
        model=model,
        tokenizer=tokenizer,
        sequences=sequences,
        layer_num=layer_num,
        batch_size=batch_size
    )
    # activations is a tensor of shape (n_tokens, hidden_dim)

    return {
        'activations': activations,
        'token_data': token_data,
        'sequences': sequences
    }

def get_seq_ids_from_token_data(token_data: Dict, subset_indices: Tuple[int, ...]) -> List[str]:
    """
    Extract and flatten sequence IDs from token data for specified subsets.
    
    Args:
        token_data: Dictionary containing sequence_ids
        subset_indices: Tuple of subset indices to extract
        
    Returns:
        List of flattened sequence IDs
    """
    # Extract sequence IDs for each subset
    all_subset_ids = []
    for i, subset_idx in enumerate(subset_indices):
        subset_key = f's{subset_idx}'
        subset_sequence_ids = token_data['sequence_ids'][subset_key]
        all_subset_ids.append(subset_sequence_ids)

    # Flatten the list of sequence IDs
    flat_seq_ids = []
    for subset_ids in all_subset_ids:
        flat_seq_ids.extend(subset_ids)
        
    return flat_seq_ids


def create_binary_labels(df: pd.DataFrame, 
                         annotation_column: str, 
                         annotations: List[str], 
                         kmer: str = None) -> pd.Series:
    """Create binary labels based on whether annotations contain any value from a list
    or if a specific k-mer (target-value) is present in the tokens.

    Args:
        df (pd.DataFrame): DataFrame containing annotations
        annotation_column (str): Name of column containing annotations
        annotations (list or str): List of target annotations to search for
        kmer (str, optional): Alternative target to search for (e.g., 'TAG')

    Returns:
        pd.Series: Binary labels (0 or 1)
    """
    # Convert single string annotation to list for consistent handling
    if isinstance(annotations, str):
        annotations = [annotations]

    elif not isinstance(annotations, list):
        raise ValueError("Annotations must be a string or list of strings")

    # We label tokens based on either k-mers being present or annotations
    # If using token-based labeling
    if kmer:

        print(f"Using token-based labeling with target value: {kmer}")
        labels = df['tokens'].apply(lambda x: 1 if kmer in x else 0).astype(int)

    # Default: using annotation-based labeling
    else:

        print(f"Using annotation-based labeling with annotations: {annotations}")
        # Fix for list-type annotation columns
        def check_for_annotation(annotation_list):
                
            # Check if any annotation is a substring of any element in the list
            label = 1 if any(
                any(ann in list_item for list_item in annotation_list)
                for ann in annotations) else 0
            
            return label
            
        labels = df[annotation_column].apply(check_for_annotation)


    print(f"Found {labels.sum()} positive examples")
    return labels

def compute_class_statistics(labels: pd.Series) -> Dict[str, float]:
    """Compute basic statistics about class distribution.

    Args:
        labels (pd.Series): Binary labels

    Returns:
        dict: Statistics including positive count, base rate accuracy, class frequency
    """
    # First, handle edge case where all labels are negative
    positive_count = labels.sum()
    if positive_count == 0:
        return {
            'positive_count': 0,
            'base_rate_accuracy': None,
            'class_frequency': 0
        }

    # Calc accuracy achieved by always predicting the majority class ('base rate accuracy')
    class_freq = positive_count / len(labels)
    base_rate_acc = max(class_freq, 1 - class_freq)

    return {
        'positive_count': positive_count,
        'base_rate_accuracy': base_rate_acc,
        'class_frequency': class_freq
    }

def prepare_tensors(features: torch.Tensor, 
                    labels: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare feature and label tensors for PyTorch training.

    Args:
        features (torch.Tensor): Input features
        labels (pd.Series): Binary labels

    Returns:
        tuple: (feature tensor, label tensor)
    """
    # Detach activation values from computation graph & convert labels to tensor
    x_tensor = features.clone().detach().to(dtype=torch.float32) # also convert to float32, in case of mixed precision
    y_tensor = torch.tensor(labels.values, dtype=torch.float32)

    return x_tensor, y_tensor

def process_annotation(df: pd.DataFrame,
                        xs: torch.Tensor,
                        target_value: str = None,
                        annotation: str ='CMV enhancer', 
                        annotation_column: str ='token_annotations') -> Dict[str, Any]:

    """Process a single annotation and prepare data for training.

    Args:
        df (pd.DataFrame): Input DataFrame with annotations
        xs (torch.Tensor): Feature tensor
        target_value (str): kmer for token-based labeling 
        annotation (str): Target annotation
        annotation_column (str): Column containing annotations

    Returns:
        dict: Processed data and statistics

    Raises:
        ValueError: If no positive examples found for annotation
    """
    # Create labels
    labels = create_binary_labels(df, annotation_column, annotation, kmer = target_value)

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


def generate_shuffled_label_matrix(original_labels: torch.Tensor, 
                                    num_probes: int, 
                                    real_probe_indices: list = [0]) -> torch.Tensor:
    """Generate matrix of shuffled labels for multiple probes
    
    Args:
        original_labels (torch.Tensor or list of torch.Tensor): Original binary labels
            If a single tensor [n_samples], will use the same labels for all real probes
            If a list of tensors, each element corresponds to different real labels
        num_probes (int): Number of probes to generate labels for
        real_probe_indices (list or None): Indices of probes to use real (unshuffled) labels
            If None, only the first probe (index 0) will use real labels
        
    Returns:
        torch.Tensor: Label matrix [n_samples, num_probes]
    """
    # Default to only first probe having real labels
    if real_probe_indices is None:
        real_probe_indices = [0]
    
    # If original_labels is a single tensor, use it for all real probes
    if not isinstance(original_labels, list):
        original_labels = [original_labels] * len(real_probe_indices)
    
    # Ensure we have the right number of label sets
    assert len(original_labels) == len(real_probe_indices), "Number of label sets must match number of real probe indices"
    
    # Get dimensions
    n_samples = original_labels[0].size(0)
    label_matrix = torch.zeros(n_samples, num_probes)
    
    # First, shuffle all labels
    for i in range(num_probes):
        # Get shuffled indices while preserving class distribution
        shuffled_indices = torch.randperm(n_samples)
        label_matrix[:, i] = original_labels[0][shuffled_indices]
    
    # Then set the real labels for specified probes
    for idx, probe_idx in enumerate(real_probe_indices):
        label_matrix[:, probe_idx] = original_labels[idx]
    
    return label_matrix



# Training Class
class BatchProbeTrainer:

    def __init__(self, num_probes: int, input_dim: int, config: dict, loss_weights=None):
        """Initialize trainer for multiple probes at once"""
        self.config = config
        self.num_probes = num_probes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a single linear layer classifier for all probes
        # each column of the weight matrix and each output corresponds to a probe 
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

    def create_data_loaders(self, 
                            x_tensor: torch.Tensor, 
                            y_tensor_matrix: torch.Tensor) -> Tuple[DataLoader, DataLoader]:

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

    def train_epoch(self, train_loader) -> float:
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

    def evaluate(self, val_loader) -> Dict[str, Any]:
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

    def train(self, 
             x_tensor: torch.Tensor, 
             y_tensor_matrix: torch.Tensor,
             verbose: bool = True) -> Dict[str, Any]:
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