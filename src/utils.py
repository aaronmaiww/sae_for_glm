import pandas as pd
import torch
from typing import List, Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import threading
import queue
from torch.cuda.amp import autocast, GradScaler
import wandb
import os

### SAE ARCHs


class JumpReLU(torch.autograd.Function):
    """
    JumpReLU with straight-through estimator for gradient through the discontinuity.
    Following Rajamanoharan et al (2024) approach but allowing gradient to flow to all parameters.
    """
    @staticmethod
    def forward(ctx, x, threshold):
        """Forward pass applies threshold: output = x if x > threshold else 0"""
        ctx.save_for_backward(x, threshold)
        return torch.where(x > threshold, x, torch.zeros_like(x))
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass with straight-through estimator"""
        x, threshold = ctx.saved_tensors
        # Gradient of 1 where x > threshold, 0 otherwise
        grad_x = grad_output.clone()
        grad_x = torch.where(x > threshold, grad_x, torch.zeros_like(grad_x))
        
        # Epsilon for the window around the threshold
        epsilon = 2.0
        
        # Compute gradient with respect to threshold
        # STE is non-zero in a narrow window around the threshold
        window_mask = (x - threshold).abs() < (epsilon / 2)
        grad_threshold = torch.zeros_like(threshold)
        if window_mask.any():
            # For points near the threshold, provide a gradient signal
            grad_threshold = -threshold * window_mask.float() * grad_output / epsilon
        
        return grad_x, grad_threshold

class AnthropicsAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_mlp = cfg["d_mlp"]                         # Input/output dimension
        d_hidden = cfg["d_mlp"] * cfg["dict_mult"]   # Hidden layer dimension
        dtype = cfg.get("enc_dtype", torch.float32)  # Data type
        
        # Sparsity hyperparameters
        self.sparsity_coeff = cfg.get("l0_coeff", 1.0)   # Lambda_S
        self.preact_coeff = cfg.get("preact_coeff", 3e-6) # Lambda_P
        self.c = cfg.get("tanh_scaling", 4.0)            # Tanh scaling factor
        self.epsilon = cfg.get("epsilon", 2.0)           # Gradient window width
        
        # Initialize log threshold parameter
        initial_threshold = cfg.get("activation_threshold", 0.1)
        self.log_threshold = nn.Parameter(torch.full(
            (d_hidden,), math.log(initial_threshold), dtype=dtype
        ))
        
        # Initialize weights with careful initialization
        # Decoder initialization from U(-1/n, 1/n)
        self.W_dec = nn.Parameter(
            torch.empty(d_hidden, d_mlp, dtype=dtype).uniform_(-1/d_mlp, 1/d_mlp)
        )
        
        # Encoder initialization from W_dec^T with scaling n/m
        self.W_enc = nn.Parameter(
            (d_mlp/d_hidden) * self.W_dec.t().clone()
        )
        
        # Initialize biases
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        
        # Store dimensions and device
        self.d_mlp = d_mlp
        self.d_hidden = d_hidden
        self.to("cuda")
        
        # Feature activation tracking
        self.register_buffer("feature_activation_count", torch.zeros(d_hidden))
        self.register_buffer("total_samples_seen", torch.tensor(0))
        
        print(f"Initialized ImprovedAutoEncoder with dimensions: {d_mlp} -> {d_hidden} -> {d_mlp}")
        print(f"Sparsity coefficient: {self.sparsity_coeff}, Pre-activation coefficient: {self.preact_coeff}")
    
    def _normalize_decoder(self):
        """Normalize decoder columns to unit L2 norm"""
        with torch.no_grad():
            norm = self.W_dec.norm(dim=1, keepdim=True)
            self.W_dec.data = self.W_dec / norm
    
    def calculate_bias_for_activation_rate(self, data_sample, target_rate=1/10000):
        """
        Examine a data subset to pick encoder bias values that achieve
        a target activation rate per feature.
        """
        with torch.no_grad():
            # Forward pass through encoder without bias
            pre_acts_no_bias = data_sample @ self.W_enc.t()
            
            # Find threshold for each feature to achieve target activation rate
            target_percentile = 100 * (1 - target_rate)
            thresholds = torch.tensor([
                torch.quantile(col, target_percentile/100) 
                for col in pre_acts_no_bias.t()
            ])
            
            # Set bias to negative of these thresholds
            return -thresholds
    
    def initialize_bias_from_data(self, dataloader, max_batches=10, input_key='input_ids'):
        """Initialize encoder bias based on actual data distribution"""
        sample_data = []
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(self.W_enc.device)
            elif isinstance(batch, dict):
                # Handle dictionary inputs - try common keys
                if input_key in batch:
                    x = batch[input_key].to(self.W_enc.device)
                elif 'x' in batch:
                    x = batch['x'].to(self.W_enc.device)
                elif 'inputs' in batch:
                    x = batch['inputs'].to(self.W_enc.device)
                else:
                    # Try the first value in the dictionary
                    x = next(iter(batch.values())).to(self.W_enc.device)
            else:
                x = batch.to(self.W_enc.device)
            
            sample_data.append(x)
        
        if not sample_data:
            raise ValueError("No data could be extracted from the dataloader")
        
        sample_data = torch.cat(sample_data, dim=0) 
        # Center the data as we do in forward pass
        sample_data = (sample_data - sample_data.mean(dim=1, keepdim=True))
        sample_data = sample_data / sample_data.std(dim=1, keepdim=True)
        
        # Calculate appropriate bias values
        bias_values = self.calculate_bias_for_activation_rate(sample_data)
        self.b_enc.data = bias_values
        print(f"Initialized encoder bias for approximately 10,000 active features per datapoint")
    
    def forward(self, x):
        """Forward pass with JumpReLU and sparsity objectives"""
        with torch.amp.autocast('cuda'):
            # Ensure x is float and normalize
            x = x.float()
            x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)
            
            # Store original for loss calculation
            x_orig = x
            
            # Encoding
            pre_acts = torch.addmm(self.b_enc, x - self.b_dec, self.W_enc)
            
            # Calculate threshold (exp of log threshold)
            threshold = torch.exp(self.log_threshold)
            
            # Apply JumpReLU activation
            acts = JumpReLU.apply(pre_acts, threshold)
            
            # Track active features for analysis
            with torch.no_grad():
                active_feats = (acts > 0).float()
                self.feature_activation_count += active_feats.sum(dim=0)
                self.total_samples_seen += x.shape[0]
            
            # Decoding
            x_reconstruct = torch.addmm(self.b_dec, acts, self.W_dec)
            
            # Calculate reconstruction loss
            l2_loss = F.mse_loss(x_reconstruct, x_orig, reduction='none').sum(dim=1).mean()
            
            # Calculate normalized MSE for monitoring
            with torch.no_grad():
                nmse = torch.norm(x_orig - x_reconstruct, p=2) / torch.norm(x_orig, p=2)
            
            # Calculate sparsity loss - tanh of feature size
            feat_size = acts.abs() * torch.norm(self.W_dec, dim=1)**2
            sparsity_loss = torch.tanh(self.c * feat_size).sum(dim=1).mean()
            
            # Calculate pre-activation loss (penalty for inactive features)
            preact_penalty = F.relu(threshold - pre_acts) * torch.norm(self.W_dec, dim=1)**2
            preact_loss = preact_penalty.sum(dim=1).mean()
            
            # Calculate L0 and L1 norms for monitoring
            with torch.no_grad():
                true_l0 = (acts > 0).float().sum(dim=1).mean()
                l1_loss = acts.abs().sum(dim=1).mean()
            
            # Total loss
            loss = l2_loss + self.sparsity_coeff * sparsity_loss + self.preact_coeff * preact_loss
            
            return loss, x_reconstruct, acts, l2_loss, nmse, l1_loss, true_l0
    
    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        """
        Maintain unit norm constraint by removing the parallel component 
        of gradients to maintain the normalization.
        """
        if self.W_dec.grad is not None:
            # Normalize W_dec
            W_dec_normed = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)
            
            # Calculate component of gradient parallel to normalized W_dec
            W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(1, keepdim=True) * W_dec_normed
            
            # Remove parallel component from gradient
            self.W_dec.grad -= W_dec_grad_proj
    
    def get_activation_stats(self):
        """Return statistics about feature activations"""
        if self.total_samples_seen == 0:
            return {"avg_active_features": 0, "feature_use_ratio": 0}
        
        # Calculate average number of active features per sample
        avg_active = self.feature_activation_count.sum() / self.total_samples_seen
        
        # Calculate what fraction of features are being used
        active_ratio = (self.feature_activation_count > 0).float().mean()
        
        return {"avg_active_features": avg_active.item(), 
                "feature_use_ratio": active_ratio.item()}
    
    def reset_activation_stats(self):
        """Reset the activation statistics"""
        self.feature_activation_count.zero_()
        self.total_samples_seen.zero_()
    
    @torch.no_grad()
    def normalize_for_analysis(self):
        """
        Return a model with identical predictions but normalized
        columns in W_dec (unit L2 norm).
        """
        new_model = ImprovedAutoEncoder({"d_mlp": self.d_mlp, "dict_mult": self.d_hidden // self.d_mlp})
        
        # Calculate norms for decoder columns
        W_dec_norm = self.W_dec.norm(dim=1, keepdim=True)
        
        # Copy and normalize parameters
        new_model.W_enc.data = self.W_enc * W_dec_norm.t()
        new_model.b_enc.data = self.b_enc * W_dec_norm.squeeze()
        new_model.W_dec.data = self.W_dec / W_dec_norm
        new_model.b_dec.data = self.b_dec.clone()
        new_model.log_threshold.data = self.log_threshold.clone()
        
        return new_model

# Optimizer setup function
def setup_training(model, lr=2e-4, warmup_steps=1000, total_steps=10000, 
                   final_sparsity_coeff=20.0, weight_decay=0.0):
    """Configure optimizer and learning rate/sparsity schedules"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, 
                                 betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=weight_decay)
    
    # Learning rate scheduler with linear warmup and decay
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup
        else:
            # Linear decay over the last 20% of training
            decay_start = int(0.8 * total_steps)
            if step <= decay_start:
                return 1.0
            else:
                return max(0.0, (total_steps - step) / (total_steps - decay_start))
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Sparsity coefficient warmup function (linear increase)
    def get_sparsity_coeff(step):
        return min(final_sparsity_coeff, (step / total_steps) * final_sparsity_coeff)
    
    return optimizer, lr_scheduler, get_sparsity_coeff

# Training step function
def training_step(model, batch, optimizer, get_sparsity_coeff, step, clip_grad_norm=1.0, input_key='input_ids'):
    """Perform one training step with adaptive sparsity coefficient"""
    # Update sparsity coefficient
    model.sparsity_coeff = get_sparsity_coeff(step)
    
    # Extract inputs from batch (could be tensor, tuple, dict, etc.)
    if isinstance(batch, dict):
        if input_key in batch:
            inputs = batch[input_key].to(model.W_enc.device)
        elif 'x' in batch:
            inputs = batch['x'].to(model.W_enc.device)
        elif 'inputs' in batch:
            inputs = batch['inputs'].to(model.W_enc.device)
        else:
            # Try the first value in the dictionary
            inputs = next(iter(batch.values())).to(model.W_enc.device)
    elif isinstance(batch, (tuple, list)):
        inputs = batch[0].to(model.W_enc.device)
    else:
        inputs = batch.to(model.W_enc.device)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    loss, reconstructions, acts, l2_loss, nmse, l1_loss, true_l0 = model(inputs)
    
    # Backward pass
    loss.backward()
    
    # Remove parallel component of gradients to maintain unit-norm constraint
    model.remove_parallel_component_of_grads()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    
    # Update weights
    optimizer.step()
    
    # Return metrics
    metrics = {
        "total_loss": loss.item(),
        "l2_loss": l2_loss.item(),
        "nmse": nmse.item(),
        "l1": l1_loss.item(),
        "l0": true_l0.item(),
        "sparsity_coeff": model.sparsity_coeff
    }
    
    return metrics



class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["d_mlp"] * cfg["dict_mult"]
        d_mlp = cfg["d_mlp"]
        self.l0_coeff = cfg.get("l0_coeff", 1)  # Let's print this in init
        #print(f"Initializing with l0_coeff: {self.l0_coeff}")
        self.threshold = cfg.get("activation_threshold", 0.3)
        self.temperature = cfg.get("temperature", 0.05) # changed from 1 to 0.05
        dtype = cfg["enc_dtype"]

        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.to("cuda")

    def get_continuous_l0(self, x):
        """Debug L0 calculation"""
        abs_x = x.abs()
        shifted = abs_x - self.threshold
        proxy = torch.sigmoid(shifted / self.temperature)
        return proxy

    def forward(self, x):
        with torch.amp.autocast('cuda'):

            x = x.float()
            # Add diagnostics

             # Add normalization
            x = (x - x.mean()) / x.std()  # or use running statistics



            # Encoding
            pre_acts = torch.addmm(self.b_enc, x - self.b_dec, self.W_enc)
            acts = F.relu(pre_acts)

            # Compute continuous L0 approximation before thresholding
            l0_proxy = self.get_continuous_l0(acts)

            # Apply hard threshold
            with torch.no_grad():
                mask = (acts.abs() > self.threshold).float()
            acts_sparse = acts * mask

            # Decoding
            x_reconstruct = torch.addmm(self.b_dec, acts_sparse, self.W_dec)

            # Losses
            l2_loss = F.mse_loss(x_reconstruct.float(), x.float(), reduction='none')
            l2_loss = l2_loss.sum(-1)
            l2_loss = l2_loss.mean()

            with torch.no_grad():
                nmse = torch.norm(x - x_reconstruct, p=2) / torch.norm(x, p=2)

            l0_loss = l0_proxy.sum(dim=1).mean()
            true_l0 = (acts_sparse.float().abs() > 0).float().sum(dim=1).mean()
            l1_loss = acts_sparse.float().abs().sum(-1).mean()

            loss = l2_loss + self.l0_coeff * true_l0

        return loss, x_reconstruct, acts_sparse, l2_loss, nmse, l1_loss, true_l0
    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        if self.W_dec.grad is not None:
            W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
            W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
            self.W_dec.grad -= W_dec_grad_proj


### Train SAE

def train_sae(sae_model, dataset, batch_size, num_epochs, learning_rate, device, start_chunk, wandb_log=False, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sae_model = sae_model.to(device)
    optimizer = torch.optim.AdamW(sae_model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    
    # Calculate total number of batches across all chunks for proper progress bar
    num_chunks = (len(dataset) + dataset.chunk_size - 1) // dataset.chunk_size
    total_samples = len(dataset) - (start_chunk * dataset.chunk_size)
    total_batches_per_epoch = (total_samples + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        epoch_metrics = {'loss': 0.0, 'l2_loss': 0.0, 'l1_loss': 0.0, 'l0_loss': 0.0, 'nmse': 0.0}
        processed_batches = 0
        
        # Create single progress bar for the entire epoch
        pbar = tqdm(total=total_batches_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}")

        for chunk_idx in range(start_chunk, num_chunks):
            torch.cuda.empty_cache()

            chunk_start = chunk_idx * dataset.chunk_size
            chunk_end = min(chunk_start + dataset.chunk_size, len(dataset))
            chunk_indices = range(chunk_start, chunk_end)

            chunk_loader = DataLoader(
                torch.utils.data.Subset(dataset, chunk_indices),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )

            # Process batches in this chunk
            for batch_idx, batch in enumerate(chunk_loader):
                batch = batch.to(device, non_blocking=True)

                # Forward pass
                with torch.amp.autocast('cuda'):
                    loss, x_reconstruct, acts, l2_loss, nmse, l1_loss, l0_loss = sae_model(batch)

                # Clear gradients
                optimizer.zero_grad(set_to_none=True)

                # Backward pass with scaled loss
                scaler.scale(loss).backward()

                # Remove parallel components and update
                sae_model.remove_parallel_component_of_grads()
                scaler.step(optimizer)
                scaler.update()

                # Record metrics
                metrics = {
                    'loss': loss.item(),
                    'l2_loss': l2_loss.item(),
                    'l1_loss': l1_loss.item(),
                    'l0_loss': l0_loss.item(),
                    'nmse': nmse.item()
                }

                # Update running metrics
                for k, v in metrics.items():
                    epoch_metrics[k] += v
                processed_batches += 1

                # Update single progress bar with latest metrics
                pbar.set_postfix({k: f"{v/processed_batches:.4f}" for k, v in epoch_metrics.items()})
                pbar.update(1)

                # Clean up
                del loss, x_reconstruct, acts, l2_loss, nmse, l1_loss, l0_loss, batch
                torch.cuda.empty_cache()

            # End of chunk cleanup
            del chunk_loader
            torch.cuda.empty_cache()

        # Close progress bar at the end of the epoch
        pbar.close()

        # Calculate and print epoch averages
        avg_metrics = {k: v/processed_batches for k, v in epoch_metrics.items()}
        print(f"\nEpoch {epoch+1} Summary:")
        for k, v in avg_metrics.items():
            print(f"Average {k}: {v:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'sae_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': sae_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_metrics['loss'],
            'metrics': avg_metrics,
        }, checkpoint_path)

        if wandb_log:
            wandb.log({
                'epoch': epoch,
                'loss': avg_metrics['loss'],
                'l2_loss': avg_metrics['l2_loss'],
                'l1_loss': avg_metrics['l1_loss'],
                'l0_loss': avg_metrics['l0_loss'],
                'nmse': avg_metrics['nmse'],
            })

        # Save best model if needed
        if avg_metrics['loss'] < best_loss:
            best_loss = avg_metrics['loss']
            best_model_path = os.path.join(save_dir)
            torch.save({
                'epoch': epoch,
                'model_state_dict': sae_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_metrics['loss'],
                'metrics': avg_metrics,
            }, best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    return sae_model

##### ACTIVATION EXTRACTION

def get_layer_activations(model, input_ids, attention_mask=None, layer_N=11, model_name = 'NT'):
    # Ensure the model is in evaluation mode
    model.eval()

    # Create a dictionary to store activations
    activations = []

    # Define a forward hook
    def hook_fn(module, input, output):
        activations.append(output)

    # Register the hook on one layers
    hooks = []
    if model_name == 'NT':
        hooks.append(model.esm.encoder.layer[layer_N].output.dense.register_forward_hook(hook_fn))

    elif model_name == 'metagene-1':
        hooks.append(model.model.layers[layer_N].input_layernorm.register_forward_hook(hook_fn))

    # Perform a forward pass
    with torch.no_grad():
        if attention_mask != None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        else:
            outputs = model(input_ids=input_ids)


    # Remove the hooks
    for hook in hooks:
        hook.remove()

    return activations


def get_residual_activations(model, input_ids, attention_mask=None, layer_N=11, position='post_mlp'):
    model.eval()
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(input[0])
    
    hooks = []
    if position == 'pre_mlp':
        hooks.append(model.esm.encoder.layer[layer_N].intermediate.register_forward_hook(hook_fn))
    elif position == 'post_mlp':
        # Hook after the entire layer to get full residual stream
        hooks.append(model.esm.encoder.layer[layer_N].register_forward_hook(hook_fn))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask) if attention_mask is not None else model(input_ids=input_ids)
    
    for hook in hooks:
        hook.remove()
        
    return activations





##### ANNOTATION

def list_flatten(nested_list) -> list:
    """
    Flattens a list of lists
    """
    return [x for y in nested_list for x in y]


def get_seq_annotation(token_spec: Dict[str, Any], df_annotated: pd.DataFrame, special_tokens: List[str], descriptor_col: str) -> List[str]:
    """
    Get the annotation(s) for a given token, including special tokens and regular annotations.

    Args:
    token_spec (Dict[str, Any]): A dictionary specifying the token, with keys 'seq_id', 'start', 'end', and 'token'.
    df_annotated (pd.DataFrame): A DataFrame containing the annotations.
    special_tokens (List[str]): A list of special tokens to check for.

    Returns:
    List[str]: A list of annotations that overlap with the specified token, including special token annotations.
    """
    required_columns = ['seq_id', 'qstart', 'qend']
    if not all(col in df_annotated.columns for col in required_columns):
        raise ValueError(f"DataFrame is missing one or more required columns: {required_columns}")

    if token_spec['seq_id'] not in df_annotated['seq_id'].unique():
        raise ValueError(f"Sequence ID '{token_spec['seq_id']}' not found in the DataFrame.")
    
    if token_spec['start'] >= token_spec['end']:
        raise ValueError("Start location must be less than end location.")

    if 'token' not in token_spec:
        raise ValueError("Token specification must include 'token' key.")

    token_annotations = []
    
    # Check if it's a special token
    if token_spec['token'] in special_tokens:
        token_annotations.append(f"special token: {token_spec['token']}")
        return token_annotations, None, None 
    
    # Always check for overlapping annotations, regardless of whether it's a special token or not

    # to-do: make sure (1) we want 'qstart/qend' and (2) any overlap should be enough

    annotated_regions = df_annotated[
        (df_annotated['seq_id'] == token_spec['seq_id']) &
        (df_annotated['qstart'] < token_spec['end']) &
        (df_annotated['qend'] > token_spec['start'])
    ]
    token_annotations.extend(annotated_regions[descriptor_col].tolist()) ## choice: could also use 'Description' for more detail

    # additionally provide information about quality of the annotation
    evalue_match = annotated_regions['evalue'].values
    pident_match = annotated_regions['pident'].values

    return token_annotations, evalue_match, pident_match

def create_context(batch: List[str], position: int, len_prefix: int, len_suffix: int) -> str:
    """
    Create a context string for a token with specified prefix and suffix lengths.

    Args:
        batch (List[str]): The list of tokens.
        position (int): The position of the current token.
        len_prefix (int): The desired length of the prefix.
        len_suffix (int): The desired length of the suffix.

    Returns:
        str: A formatted context string.
    """
    prefix_start = max(0, position - len_prefix)
    suffix_end = min(len(batch), position + 1 + len_suffix)

    prefix = "".join(batch[prefix_start:position])
    current_token = batch[position]
    suffix = "".join(batch[position + 1:suffix_end])

    return f"{prefix} |{current_token}| {suffix}"

from tqdm.auto import tqdm

def make_token_df_new(
    tokens: torch.Tensor,
    tokenizer,
    df_annotated: pd.DataFrame,
    seq_ids: List[str],
    len_prefix: int = 5,
    len_suffix: int = 2,
    nucleotides_per_token: int = 6,
    descriptor_col: str = 'Feature'
    ) -> pd.DataFrame:
    """
    Create a DataFrame with token information, context, and annotations for batched input.
    Includes progress bars for batch and token processing.

    Args:
        tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
        tokenizer: The tokenizer object.
        df_annotated (pd.DataFrame): DataFrame with sequence annotations.
        seq_ids (List[str]): List of sequence identifiers for each batch item.
        len_prefix (int): Length of context prefix.
        len_suffix (int): Length of context suffix.
        nucleotides_per_token (int): Number of nucleotides represented by each token.

    Returns:
        pd.DataFrame: DataFrame containing token information and annotations.
    """
    if tokens.numel() == 0:
        return pd.DataFrame(columns=['seq_id', 'tokens', 'context', 'token_annotations', 'context_annotations'])

    batch_size, seq_len = tokens.shape
    special_tokens = tokenizer.all_special_tokens
    data = []

    # Main progress bar for batches
    batch_progress = tqdm(range(batch_size), desc="Processing batches", unit="batch")
    
    for batch_idx in batch_progress:
        seq_tokens = tokens[batch_idx]
        seq_id = seq_ids[batch_idx]
        
        # Decode each token id to the corresponding string
        str_tokens = [tokenizer.decode(token).replace(' ', '') for token in seq_tokens]
        
        
        for position, token in enumerate(str_tokens):
            token_start = position * nucleotides_per_token
            token_end = token_start + nucleotides_per_token

            # Token annotation
            token_spec = {
                'seq_id': f'valseq_{seq_id}',
                'start': token_start,
                'end': token_end,
                'token': token
            }
            token_annotation, evalue_match, pident_match = get_seq_annotation(token_spec, df_annotated, special_tokens, descriptor_col)

            # Context and its annotation
            context = create_context(str_tokens, position, len_prefix, len_suffix)
            context_start = max(0, token_start - len_prefix * nucleotides_per_token)
            context_end = min(len(str_tokens) * nucleotides_per_token, token_end + len_suffix * nucleotides_per_token)
            context_spec = {
                'seq_id': f'valseq_{seq_id}',
                'start': context_start,
                'end': context_end,
                'token': context
            }

            context_annotation,_,_ = get_seq_annotation(context_spec, df_annotated, special_tokens, descriptor_col)

            data.append({
                'seq_id': seq_id,
                'token_pos': position,
                'tokens': token,
                'context': context,
                'token_annotations': token_annotation,
                'context_annotations': context_annotation,
                'e-value annotation': evalue_match,
                'percentage match': pident_match
            })

        # Update batch progress description
        batch_progress.set_description(f"Completed batch {batch_idx+1}/{batch_size}")

    return pd.DataFrame(data)



#### Activation Storage

class ActivationsDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, cache_size=10000):
        self.h5_path = h5_path
        self.cache_size = cache_size
        self.cache = {}

        # Read metadata once
        with h5py.File(h5_path, 'r') as f:
            self.length = f['activations'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]

        # Read data in blocks to minimize file operations
        block_start = (idx // self.cache_size) * self.cache_size
        block_end = min(block_start + self.cache_size, self.length)

        with h5py.File(self.h5_path, 'r') as f:
            block_data = f['activations'][block_start:block_end]

        # Cache the block
        for i, data in enumerate(block_data):
            cache_idx = block_start + i
            self.cache[cache_idx] = torch.FloatTensor(data)

        return self.cache[idx]

import queue
import threading
import h5py
import torch
from torch.utils.data import Dataset


class BackgroundLoader:
    def __init__(self, h5_file, chunk_size):
        self.h5_file = h5_file
        self.chunk_size = chunk_size
        self.queue = queue.Queue(maxsize=2)  # Keep 2 chunks in memory
        self.thread = None
        self.current_chunk_idx = 0

    def load_chunk(self, chunk_idx):
        data = self.h5_file['activations']
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, len(data))
        return torch.from_numpy(data[start_idx:end_idx][:]).float()

    def loader_thread(self, next_chunk_idx):
        chunk = self.load_chunk(next_chunk_idx)
        self.queue.put((next_chunk_idx, chunk))

    def get_next_chunk(self):
        # Start loading next chunk in background if not already loading
        if self.thread is None or not self.thread.is_alive():
            next_chunk_idx = self.current_chunk_idx + 1
            self.thread = threading.Thread(target=self.loader_thread, args=(next_chunk_idx,))
            self.thread.start()

        # Get current chunk
        chunk_idx, chunk = self.queue.get()
        self.current_chunk_idx = chunk_idx
        return chunk

class ChunkedActivationsDataset(Dataset):
    def __init__(self, h5_path, batch_size=2048*4, chunks_in_memory=4, max_chunks=None):
        self.h5_file = h5py.File(h5_path, 'r')
        self.data = self.h5_file['activations']
        self.length = self.data.shape[0]
        self.batch_size = batch_size

        # Make chunk size a multiple of batch_size
        self.chunk_size = (chunks_in_memory * batch_size)

        # Adjust length if max_chunks is specified
        if max_chunks:
            self.length = min(self.length, max_chunks * self.chunk_size)

        # Initialize loader
        self.loader = BackgroundLoader(self.h5_file, self.chunk_size)
        # Initialize first chunk
        self.current_chunk = self.loader.load_chunk(0)
        self.current_chunk_idx = 0
        self.loader.get_next_chunk()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        if chunk_idx != self.current_chunk_idx:
            self.current_chunk = self.loader.get_next_chunk()
            self.current_chunk_idx = chunk_idx

        local_idx = idx % self.chunk_size
        return self.current_chunk[local_idx]

    def __del__(self):
        self.h5_file.close()
