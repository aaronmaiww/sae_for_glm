import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import h5py
import numpy as np
import threading
import queue
import wandb
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Tuple



### SAE ARCHs

class AutoEncoder(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        d_hidden = cfg["d_mlp"] * cfg["dict_mult"]
        d_mlp = cfg["d_mlp"]
        self.l0_coeff = cfg.get("l0_coeff", 1)  
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

    def get_continuous_l0(self, x: torch.Tensor) -> torch.Tensor:
        """Debug L0 calculation"""
        abs_x = x.abs()
        shifted = abs_x - self.threshold
        proxy = torch.sigmoid(shifted / self.temperature)
        return proxy

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with torch.amp.autocast('cuda'):
            
            # Check for empty or single-element tensors
            if x.numel() <= 1:
                raise ValueError(f"Input tensor is empty or has only one element (shape: {x.shape}, numel: {x.numel()}). This will cause issues with normalization.")
            
            x = x.float()

             # Add normalization
            x = (x - x.mean()) / (x.std() + 1e-5)  # Add epsilon for stability

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

            # Get Losses
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
        """
        Remove the parallel component of the gradients of W_dec.
        So that each decoder vector is updated with an orthogonal gradient.
        To do: explain 
        """
       
        if self.W_dec.grad is not None:
            W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
            W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
            self.W_dec.grad -= W_dec_grad_proj


### Train & Val SAE

def train_sae(sae_model: nn.Module,  # The SAE model to train
              dataset: Dataset,  # The dataset to train on
              batch_size: int,  # Batch size for training
              num_epochs: int,  # Number of epochs to train 
              learning_rate: float,  # Learning rate for the optimizer 
              device: str,  # Device to run training on (e.g., 'cuda' or 'cpu') 
              start_chunk: int = 0,  # Which chunk to start training from 
              wandb_log: bool = False,  # Whether to log metrics to wandb
              save_dir: str = 'checkpoints',  # Directory to save checkpoints
) -> nn.Module:

    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    sae_model = sae_model.to(device)
    optimizer = torch.optim.AdamW(sae_model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda')

    # To save best model later
    best_loss = float('inf')
    
    # Calculate total number of batches across all chunks for proper progress bar
    num_chunks = (len(dataset) + dataset.chunk_size - 1) // dataset.chunk_size
    total_batches_per_epoch = 0

    for chunk_idx in range(start_chunk, num_chunks):
        chunk_start = chunk_idx * dataset.chunk_size
        chunk_end = min(chunk_start + dataset.chunk_size, len(dataset))
        chunk_indices = range(chunk_start, chunk_end)

        chunk_loader = DataLoader(
            Subset(dataset, chunk_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        total_batches_per_epoch += len(chunk_loader) # get the correct number of batches from the dataloader.

    
    # Start training
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

                # Add before the forward pass
                if torch.isnan(batch).any():
                    print("NaN detected in input batch")
                    continue

                # Forward pass
                with torch.amp.autocast('cuda'):
                    loss, x_reconstruct, acts, l2_loss, nmse, l1_loss, l0_loss = sae_model(batch)
                
                # Add after the forward pass
                if torch.isnan(x_reconstruct).any():
                    print("NaN detected in reconstruction")
                    continue

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

def validate_sae(sae_model: nn.Module, 
                 dataset: Dataset, 
                 batch_size: int, 
                 device: str, 
                 start_chunk: int =0, 
                 num_chunks: Optional[int] = None,
                 wandb_log: bool = False) -> Dict[str, float]:
    """
    Validate a trained SAE model on a validation dataset.
    Similar to train_sae but without the training/optimization steps.

    """
    sae_model = sae_model.to(device)
    sae_model.eval()  # Set model to evaluation mode
    
   # Calculate total number of chunks and respect the num_chunks limit if provided
    total_chunks = (len(dataset) + dataset.chunk_size - 1) // dataset.chunk_size
    end_chunk = min(total_chunks, start_chunk + num_chunks) if num_chunks is not None else total_chunks
    
    # Calculate total samples and batches to process
    total_samples = min(len(dataset) - (start_chunk * dataset.chunk_size), 
                         dataset.chunk_size * (end_chunk - start_chunk))
    total_batches = (total_samples + batch_size - 1) // batch_size
    
    # Initialize metrics
    val_metrics = {'loss': 0.0, 'l2_loss': 0.0, 'l1_loss': 0.0, 'l0_loss': 0.0, 'nmse': 0.0}
    processed_batches = 0
    
    # Create progress bar for the entire validation
    pbar = tqdm(total=end_chunk, desc=f"Validating SAE")
    
    for chunk_idx in range(start_chunk, end_chunk):
        torch.cuda.empty_cache()
        
        chunk_start = chunk_idx * dataset.chunk_size
        chunk_end = min(chunk_start + dataset.chunk_size, len(dataset))
        chunk_indices = range(chunk_start, chunk_end)
        
        chunk_loader = DataLoader(
            torch.utils.data.Subset(dataset, chunk_indices),
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle for validation
            num_workers=0,
            pin_memory=False
        )
        
        # Process batches in this chunk
        for batch_idx, batch in enumerate(chunk_loader):
            batch = batch.to(device, non_blocking=True)
            
            # Skip batches with NaN values
            if torch.isnan(batch).any():
                print("NaN detected in input batch - skipping")
                continue
            
            # Forward pass
            with torch.no_grad():  # No need to track gradients for validation
                with torch.amp.autocast('cuda'):
                    loss, x_reconstruct, acts, l2_loss, nmse, l1_loss, l0_loss = sae_model(batch)
            
            # Skip if reconstruction has NaN values
            if torch.isnan(x_reconstruct).any():
                print("NaN detected in reconstruction - skipping")
                continue
            
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
                val_metrics[k] += v
            processed_batches += 1
            
            # Update progress bar with latest metrics
            pbar.set_postfix({k: f"{v/processed_batches:.4f}" for k, v in val_metrics.items()})
            pbar.update(1)
            
            # Clean up
            del loss, x_reconstruct, acts, l2_loss, nmse, l1_loss, l0_loss, batch
            torch.cuda.empty_cache()
        
        # End of chunk cleanup
        del chunk_loader
        torch.cuda.empty_cache()
    
    # Close progress bar
    pbar.close()
    
    # Calculate and return averages
    avg_metrics = {k: v/processed_batches for k, v in val_metrics.items()} if processed_batches > 0 else val_metrics
    
    # Print summary
    print(f"\nValidation Summary:")
    for k, v in avg_metrics.items():
        print(f"Average {k}: {v:.4f}")
    
    # Log to wandb if requested
    if wandb_log:
        wandb.log({
            'val_loss': avg_metrics['loss'],
            'val_l2_loss': avg_metrics['l2_loss'],
            'val_l1_loss': avg_metrics['l1_loss'],
            'val_l0_loss': avg_metrics['l0_loss'],
            'val_nmse': avg_metrics['nmse'],
        })
    
    return avg_metrics


def run_sae(sae_model: nn.Module, 
            dataset: Dataset, 
            batch_size: int, 
            device: str, 
            start_chunk: int = 0, 
            num_chunks: Optional[int] = None, 
            return_activations: bool = True, 
            return_reconstructions: bool = True) -> Dict[str, Any]:
    """
    Run a trained SAE model on a dataset and collect outputs.
    Similar to validate_sae but returns the model outputs instead of just metrics.
    
    Args:
        sae_model: The SAE model to run
        dataset: The dataset to process
        batch_size: Batch size for processing
        device: Device to run on
        start_chunk: Which chunk to start from
        num_chunks: How many chunks to process (None for all)
        return_activations: Whether to return the activations
        return_reconstructions: Whether to return the reconstructions
        
    Returns:
        Dictionary containing collected outputs and metrics
    """
    sae_model = sae_model.to(device)
    sae_model.eval()  # Set model to evaluation mode
    
    # Calculate total number of chunks and respect the num_chunks limit if provided
    total_chunks = (len(dataset) + dataset.chunk_size - 1) // dataset.chunk_size
    end_chunk = min(total_chunks, start_chunk + num_chunks) if num_chunks is not None else total_chunks
    
    # Calculate total samples to process
    total_samples = min(len(dataset) - (start_chunk * dataset.chunk_size), 
                         dataset.chunk_size * (end_chunk - start_chunk))
    
    # Initialize output containers
    outputs = {
        'metrics': {'loss': 0.0, 'l2_loss': 0.0, 'l1_loss': 0.0, 'l0_loss': 0.0, 'nmse': 0.0},
        'sample_indices': [],
    }
    
    if return_activations:
        outputs['activations'] = []
    
    if return_reconstructions:
        outputs['reconstructions'] = []
        outputs['inputs'] = []
    
    processed_batches = 0
    processed_samples = 0
    
    # Create progress bar for the entire run
    pbar = tqdm(total=end_chunk - start_chunk, desc=f"Running SAE")
    
    for chunk_idx in range(start_chunk, end_chunk):
        torch.cuda.empty_cache()
        
        chunk_start = chunk_idx * dataset.chunk_size
        chunk_end = min(chunk_start + dataset.chunk_size, len(dataset))
        chunk_indices = range(chunk_start, chunk_end)
        
        chunk_loader = DataLoader(
            torch.utils.data.Subset(dataset, chunk_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Process batches in this chunk
        for batch_idx, batch in enumerate(chunk_loader):
            batch = batch.to(device, non_blocking=True)
            
            # Skip batches with NaN values
            if torch.isnan(batch).any():
                print("NaN detected in input batch - skipping")
                continue
            
            # Forward pass
            with torch.no_grad():  # No need to track gradients
                with torch.amp.autocast('cuda'):
                    loss, x_reconstruct, acts, l2_loss, nmse, l1_loss, l0_loss = sae_model(batch)
            
            # Skip if reconstruction has NaN values
            if torch.isnan(x_reconstruct).any():
                print("NaN detected in reconstruction - skipping")
                continue
            
            # Record metrics
            batch_metrics = {
                'loss': loss.item(),
                'l2_loss': l2_loss.item(),
                'l1_loss': l1_loss.item(),
                'l0_loss': l0_loss.item(),
                'nmse': nmse.item()
            }
            
            # Update running metrics
            for k, v in batch_metrics.items():
                outputs['metrics'][k] += v
            
            # Store batch size for calculating averages later
            processed_batches += 1
            batch_size_actual = batch.shape[0]
            processed_samples += batch_size_actual
            
            # Store sample indices
            start_idx = chunk_start + batch_idx * batch_size
            batch_indices = list(range(start_idx, start_idx + batch_size_actual))
            outputs['sample_indices'].extend(batch_indices)
            
            # Store activations if requested
            if return_activations:
                outputs['activations'].append(acts.detach().cpu())
            
            # Store reconstructions and inputs if requested
            if return_reconstructions:
                outputs['reconstructions'].append(x_reconstruct.detach().cpu())
                outputs['inputs'].append(batch.detach().cpu())
            
            # Update progress bar with latest metrics
            pbar.set_postfix({k: f"{v/processed_batches:.4f}" for k, v in outputs['metrics'].items()})
        
        # End of chunk processing
        pbar.update(1)
        
        # Clean up
        del chunk_loader
        torch.cuda.empty_cache()
    
    # Close progress bar
    pbar.close()
    
    # Calculate average metrics
    if processed_batches > 0:
        outputs['metrics'] = {k: v/processed_batches for k, v in outputs['metrics'].items()}
    
    # Convert lists of tensors to single tensors
    if return_activations:
        outputs['activations'] = torch.cat(outputs['activations'], dim=0)
    
    if return_reconstructions:
        outputs['reconstructions'] = torch.cat(outputs['reconstructions'], dim=0)
        outputs['inputs'] = torch.cat(outputs['inputs'], dim=0)
    
    # Print summary
    print(f"\nRun Summary:")
    print(f"Processed {processed_samples} samples")
    for k, v in outputs['metrics'].items():
        print(f"Average {k}: {v:.4f}")
    
    return outputs

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


def extract_layer_activations_and_latents(model_nt, sae_model, tokens, layer_num, batch_size=32, device='cuda'):
    """
    Extract MLP activations and corresponding SAE latent representations.
    """
    # Validate layer number
    max_layers = len(model_nt.esm.encoder.layer)
    if layer_num >= max_layers:
        raise ValueError(f"Layer number {layer_num} is out of range. Model has {max_layers} layers.")

    # Get model dimensions
    d_mlp = model_nt.config.hidden_size
    
    # Calculate batch information
    total_tokens = tokens['input_ids'].shape[0] * tokens['input_ids'].shape[1]
    num_seqs = tokens['input_ids'].shape[0]
    num_batches = (num_seqs + batch_size - 1) // batch_size

    print(f"Total tokens: {total_tokens}, num_batches: {num_batches}")

    all_latents = []
    all_acts = []
    metrics = {
        'loss': 0.0,
        'l2_loss': 0.0,
        'nmse': 0.0,
        'l1_loss': 0.0,
        'true_l0': 0.0,
        'num_samples': 0
    }

    # Ensure models are in eval mode
    sae_model.eval()
    model_nt.eval()
    
    model_nt = model_nt.to(device)
    sae_model = sae_model.to(device)

    # Print model and input details for debugging
    print(f"SAE Model input expected shape: {sae_model.W_enc.shape}")
    print(f"d_mlp: {d_mlp}")

    # Process batches with progress bar
    try:
        for i in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_seqs)

            # Reshape tokens for current batch
            batch_input_ids = tokens['input_ids'][start_idx:end_idx].to(device)
            batch_attention_mask = tokens['attention_mask'][start_idx:end_idx].to(device)


            with torch.no_grad():
                with autocast():
                    # Get MLP activations
                    try:
                        mlp_act = get_layer_activations(
                            model_nt,
                            batch_input_ids,
                            batch_attention_mask,
                            layer_N=layer_num
                        )

                       
                        # Additional error checking
                        if mlp_act is None or len(mlp_act) == 0 or mlp_act[0].numel() == 0:
                            print(f"No or empty activations retrieved for batch {i}")
                            continue

                        mlp_act = mlp_act[0].reshape(-1, d_mlp)
                        
                      
                        # Check for empty tensors after reshaping
                        if mlp_act.numel() == 0:
                            print(f"Empty tensor after reshaping in batch {i}")
                            continue

                        
                        # Forward pass through SAE
                        loss, x_reconstruct, latents, l2_loss, nmse, l1_loss, true_l0 = sae_model(mlp_act)

                        # Check latents before appending
                        if latents.numel() == 0:
                            print(f"Empty latents in batch {i}")
                            continue
                            
                        all_latents.append(latents)
                        all_acts.append(mlp_act)
                        
                        # Accumulate metrics
                        metrics['loss'] += loss.item() * batch_size
                        metrics['l2_loss'] += l2_loss.item() * batch_size  
                        metrics['nmse'] += nmse.item() * batch_size
                        metrics['l1_loss'] += l1_loss.item() * batch_size
                        metrics['true_l0'] += true_l0.item() * batch_size
                        metrics['num_samples'] += batch_size

                    except Exception as e:
                        print(f"Error processing batch {i}: {e}")
                        continue

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    finally:
        # Check if we collected any activations
        if not all_acts or not all_latents:
            raise ValueError("No activations or latents were collected. Check your input data and model.")

        # Finalize metrics
        for key in metrics:
            if key != 'num_samples':
                metrics[key] = metrics[key] / metrics['num_samples'] if metrics['num_samples'] > 0 else 0

        # Combine results, move to cpu before
        all_acts = torch.cat(all_acts, dim=0).cpu()
        all_latents = [x.cpu() for x in all_latents]
        combined_latents = torch.cat(all_latents, dim=0).cpu()
        
        # Clean up
        torch.cuda.empty_cache()
        
    return all_acts, combined_latents, metrics







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