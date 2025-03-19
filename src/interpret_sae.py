# Description:

# This script contains functionality to run an SAE on some data and then 
# inspect which tokens most activate a given SAE latents

### First, tools to load data

import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from train_sae import get_layer_activations
import matplotlib.pyplot as plt



def load_and_process_annotations(file_path):
    """Load CSV and add 'valseq_' prefix to seq_id column if not already present."""
    df = pd.read_csv(file_path)
    df['seq_id'] = df['seq_id'].astype(str)
    # Add 'valseq_' prefix only if it's not already there
    df['seq_id'] = df['seq_id'].apply(lambda x: x if x.startswith('valseq_') else f'valseq_{x}')
    return df

def extract_and_tokenize_sequences(df_annotations, df_val, tokenizer_nt):
    """Extract sequence IDs, get corresponding sequences, and tokenize them."""
    # Extract and sort sequence IDs
    seq_ids = list(set(df_annotations['seq_id']))
    # More robust parsing of sequence IDs
    parsed_ids = []
    for seq_id in seq_ids:
        try:
            if 'valseq_' in seq_id:
                parsed_ids.append(int(seq_id.split('valseq_')[1]))
            else:
                parsed_ids.append(int(seq_id))
        except ValueError:
            print(f"Warning: Could not parse seq_id: {seq_id}")
            continue

    seq_ids = sorted(parsed_ids)

    # Get and tokenize sequences
    sequences = df_val['sequence'].iloc[seq_ids].tolist()
    tokens = tokenizer_nt(
        sequences,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

    return tokens, seq_ids



### Second, tools to run SAE on data


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
    num_batches = (total_tokens + batch_size - 1) // batch_size

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
            end_idx = min((i + 1) * batch_size, total_tokens)

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
                        if not mlp_act or len(mlp_act) == 0:
                            print(f"No activations retrieved for batch {i}")
                            continue

                        mlp_act = mlp_act[0].reshape(-1, d_mlp)
                        
                      
                        # Check if input matches SAE model's expected input
                        if mlp_act.shape[1] != sae_model.W_enc.shape[0]:
                            print(f"Dimension mismatch: mlp_act {mlp_act.shape}, W_enc {sae_model.W_enc.shape}")
                            continue

                        all_acts.append(mlp_act)

                        # Forward pass through SAE
                        loss, x_reconstruct, latents, l2_loss, nmse, l1_loss, true_l0 = sae_model(mlp_act)
                        all_latents.append(latents)
                        
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


def show_top_activating_tokens(token_df, latent_id, combined_latents):

    # we avoid modifying token_df directly as its very time-consuming to reload if we mess it up
    token_df_copy = token_df.copy() 

    # get the activation value for the N-th unit in the SAE for each input in batch
    hidden_act_feature_id = combined_latents[:, latent_id] # N = feature_id

    # add this to the dataframe
    token_df_copy[f"latent-{latent_id}-act"] = hidden_act_feature_id.cpu().detach().numpy()

    # sort to show the most activating tokens on top, add colours
    return token_df_copy.sort_values(f"latent-{latent_id}-act", ascending=False).head(50
                                                                               ).style.background_gradient("coolwarm")