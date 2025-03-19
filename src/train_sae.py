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
