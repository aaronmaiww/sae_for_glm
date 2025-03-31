import pandas as pd
import torch
import numpy as np
from collections import Counter
import torch
from tqdm import tqdm
import gc
from sklearn.metrics import f1_score, precision_score, recall_score
import re


def measure_monosemanticity_across_latents(
    latent_dict: dict, 
    combined_latents: torch.Tensor, 
    token_df: pd.DataFrame, 
    compute_metrics_across_thresholds: callable, 
    print_metrics: callable = None, 
    validation_set: int = 0
) -> pd.DataFrame:
    """
    Measures monosemanticity metrics for latent representations.
    
    Parameters:
    -----------
    latent_dict : dict
        Dictionary mapping latent IDs to semantic annotations.
    combined_latents : torch.Tensor
        Tensor containing activation values for all latent units.
    token_df : pd.DataFrame
        DataFrame containing token information.
    compute_metrics_across_thresholds : callable
        Function that computes performance metrics at different thresholds.
    print_metrics : callable, optional
        Function to print metrics during computation, by default None.
    validation_set : int, optional
        Identifier for the validation set being used, by default 0.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing results sorted by F1 score.
    """
    # Process each latent unit and collect results
    results_list = [
        process_single_latent(latent_id, annotation_entry, combined_latents, token_df, 
                             compute_metrics_across_thresholds, print_metrics, validation_set)
        for latent_id, annotation_entry in latent_dict.items()
    ]
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Sort results if the DataFrame is not empty
    f1_column = f'best_f1_val{validation_set}'
    if not results_df.empty and f1_column in results_df.columns:
        results_df = results_df.sort_values(by=f1_column, ascending=False).reset_index(drop=True)
    
    return results_df


def process_single_latent(
    latent_id: int,
    annotation_entry: list,
    combined_latents: torch.Tensor,
    token_df: pd.DataFrame,
    compute_metrics_across_thresholds: callable,
    print_metrics: callable = None,
    validation_set: int = 0
) -> dict:
    """Process a single latent unit and return its metrics."""
    
    # Default values
    best_f1 = 0.0
    best_threshold = 0.0
    error_msg = None

    try:
        # Extract activation values for this specific latent unit
        activation_values = combined_latents[:, latent_id]
        
        # Create dataframe with activations
        token_df_copy = token_df.copy()
        token_df_copy[f"latent-{latent_id}-act"] = activation_values.cpu().detach().numpy()
        
      
        print("compute metrics...")
        # Compute metrics

        try:
            print("About to call compute_metrics...")
            results = compute_metrics_across_thresholds(
                token_df_copy, 
                annotation_entry, 
                latent_id, 
                thresholds=None, 
                modified_recall=True
            )
            print("compute_metrics returned successfully")
        except Exception as e:
            print(f"Exception in compute_metrics: {e}")
   
        
        if print_metrics:
            print_metrics(results)
        
        # Get best result based on F1 score
        print(f"DEBUG: Calling _get_best_result with results length: {len(results)}")
        best_result = _get_best_result(results, latent_id, annotation_entry)
        print(f"DEBUG: _get_best_result returned: {best_result}")
        if best_result:
            best_threshold, best_f1 = best_result[0], best_result[3]
            
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing latent {latent_id}: {error_msg}")
    
    # Prepare result dictionary
    result = {
        'latent_id': latent_id,
        'annotation': str(annotation_entry),
        f'best_f1_val{validation_set}': best_f1,
        'threshold': best_threshold
    }
    
    if error_msg:
        result['error'] = error_msg
        
    return result

def compute_metrics_across_thresholds(token_df: pd.DataFrame,
                                      annotation: list, 
                                      latent_id: int,
                                      thresholds: list, 
                                      modified_recall: bool = True) -> list:
    """
    Computes precision, recall, and F1 scores across different activation thresholds.

    Args:
        token_df: DataFrame with token data
        annotation: list
        latent_id: ID of the latent being analyzed
        thresholds: List of activation thresholds to evaluate
        modified_recall: Whether to use the modified recall method

    Returns:
        List of tuples (threshold, precision, recall, f1)
    """
    # Preprocess data
    print(f"Starting compute_metrics with annotation: {annotation}")


    if modified_recall:
        try:
            print("About to call preprocess function...")
            modified_df = preprocess_annotation_data_for_modrecall(token_df, annotation, latent_id)
            print("Preprocess function returned successfully")
            
            # Verify the dataframe
            print(f"modified_df type: {type(modified_df)}")
            if isinstance(modified_df, pd.DataFrame):
                print(f"modified_df shape: {modified_df.shape}")
                print(f"modified_df columns: {modified_df.columns}")
                if f"latent-{latent_id}-act" not in modified_df.columns:
                    print(f"WARNING: Activation column missing from modified_df")
            else:
                print(f"ERROR: modified_df is not a DataFrame")
                
        except Exception as e:
            print(f"Exception in preprocessing: {str(e)}")
            print("Falling back to original dataframe")
            modified_df = token_df.copy()
    else:
        modified_df = token_df.copy()

    print("Generate Thresholds...")
    if thresholds is None:
        if f"latent-{latent_id}-act" not in token_df.columns or token_df.empty:
            print(f"WARNING: Column latent-{latent_id}-act is missing or empty.")
            return []
          
      
        max_act = round(max(token_df[f"latent-{latent_id}-act"]))
        print(f"Max activation: {max_act}, Activation Stats:\n{token_df[f'latent-{latent_id}-act'].describe()}")

        # Ensure at least one threshold exists
        thresholds = range(0, max(1, max_act))

    
    results = []
    for threshold in thresholds:
        # Generate prediction masks
        pred_precision = (token_df[f"latent-{latent_id}-act"] > threshold).astype(int)
        pred_recall = (modified_df[f"latent-{latent_id}-act"] > threshold).astype(int)

        # Generate ground truth masks
        true_precision = token_df['token_annotations'].apply(lambda x: 1 if annotation in x else 0)
        true_recall = modified_df['token_annotations'].apply(lambda x: 1 if annotation in x else 0)

        # Compute metrics
        precision = precision_score(true_precision, pred_precision, zero_division=0)
        recall = recall_score(true_recall, pred_recall, zero_division=0)

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append((threshold, precision, recall, f1))

    return results


def preprocess_annotation_data_for_modrecall(token_df: pd.DataFrame, 
                                             annotation: list, 
                                             latent_id: int) -> pd.DataFrame:
    """
    Preprocess annotation data to extract the highest activation tokens for annotated regions
    and combine them with all non-annotated tokens.
    """
    print("Preprocess data...")
    print("Annotation:", annotation, "type:", type(annotation))
    
    if isinstance(annotation, list):
        # Check what type the token_annotations are
        sample_type = type(token_df['token_annotations'].iloc[0])
        print(f"token_annotations data type: {sample_type}")
        
        # Handle the case where token_annotations contains lists
        if isinstance(token_df['token_annotations'].iloc[0], list):
            # Direct list comparison
            has_annotation = token_df['token_annotations'].apply(
                lambda x: any(ann in x for ann in annotation)
            )
        else:
            # Try string pattern matching as before
            pattern = '|'.join(map(re.escape, annotation))
            print("Searching pattern:", pattern, "of type:", type(pattern))
            has_annotation = token_df['token_annotations'].str.contains(pattern, regex=True)
        
        not_has_annotation = ~has_annotation
    else:
        raise ValueError("Annotation must be a list of strings.")

    print("Annotation mask created. True count:", has_annotation.sum())
    
    # Get highest activation tokens for annotated regions
    # Important assumption here: an annotated region occurs only once per sequence and so we only take the highest activation token for each sequence
    high_act_tokens = (
        token_df[has_annotation]
        .groupby('seq_id')
        .apply(lambda x: x.nlargest(1, f"latent-{latent_id}-act"))
    )
    remaining_tokens = token_df[not_has_annotation]

    # Combine dataframes
    combined_df = pd.concat([high_act_tokens, remaining_tokens])

    # Right before returning
    print("About to return DataFrame")
    print(f"Final DataFrame shape: {combined_df.shape}")
    print(f"Final DataFrame column check: {f'latent-{latent_id}-act' in combined_df.columns}")
    
    print("Returning combined df...")
    
    # Try adding explicit check of the result
    try:
        result = combined_df.copy()
        print("Successfully created copy of result")
        return result
    except Exception as e:
        print(f"Error creating copy: {e}")
        # Fall back to original
        return token_df.copy()

def print_metrics(results):
    """Prints formatted metrics for each threshold."""
    for threshold, precision, recall, f1 in results:
        print(f"F1 score for threshold {threshold}: {f1:.3f}, "
              f"Precision: {precision:.3f}, Recall: {recall:.3f}")
        print("-" * 50)


def analyze_latents_fast(combined_latents: torch.Tensor, 
                         token_df: pd.DataFrame,
                         sae_model: torch.nn.Module,
                         top_n: int = 20,
                         min_threshold: int = 10, 
                         batch_size: int = 1000) -> dict:
    """
    Analyze latent units to identify what they detect, using batching for memory efficiency.
    
    Parameters:
    -----------
    combined_latents: Activation values for all latent units
    token_df: DataFrame containing tokens and their annotations
    sae_model: The sparse autoencoder model
    top_n: Number of top activations to consider per latent
    min_threshold: Minimum count for an annotation to be considered common
    batch_size: Number of latents to process at once
    
    Returns:
    --------
    Dictionary mapping latent IDs to their detected annotation sets
    """
    n_latents = sae_model.d_hidden
    latent_dict = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pre-process annotations and tokens once to avoid repeated access
    annotations_list = [safe_get_annotations(ann) for ann in token_df['token_annotations']]
    tokens_array = token_df['tokens'].values
    
    # Excluded annotations that should be ignored
    excluded_annotations = {'special token: <cls>', 'special token: <pad>'}

    # Process latents in batches to save memory
    for batch_start in tqdm(range(0, n_latents, batch_size)):
        batch_end = min(batch_start + batch_size, n_latents)
        batch_latents = process_latent_batch(combined_latents, batch_start, batch_end, device)
        
        # Find top activating tokens for each latent in batch
        top_k_indices, top_k_values = get_top_activations(batch_latents, top_n)
        
        # Analyze each latent in the current batch
        for i, latent_id in enumerate(range(batch_start, batch_end)):
            analyze_single_latent(
                latent_id, i, top_k_indices, top_k_values, 
                annotations_list, tokens_array, excluded_annotations,
                min_threshold, latent_dict
            )
            
        # Free memory
        cleanup_memory(batch_latents, top_k_indices, top_k_values)

    return latent_dict


def process_latent_batch(combined_latents, batch_start, batch_end, device):
    """Extract and process a batch of latents."""
    batch_latents = combined_latents[:, batch_start:batch_end].to(device)
    return batch_latents.cpu().detach().numpy()


def get_top_activations(batch_latents, top_n):
    """Find indices and values of top activating tokens for each latent."""
    # Get the indices of the top_n activations for each latent
    top_n_indices = np.argsort(-batch_latents, axis=0)[:top_n, :]

    # Retrieve the corresponding activation values
    col_indices = np.arange(batch_latents.shape[1])
    top_n_values = batch_latents[top_n_indices, col_indices]

    return top_n_indices, top_n_values


def analyze_single_latent(latent_id, batch_idx, top_k_indices, top_k_values, 
                         annotations_list, tokens_array, excluded_annotations,
                         min_threshold, latent_dict):
    """Analyze a single latent unit and update results dictionary."""
    # Skip latents with zero activations
    if np.any(top_k_values[:, batch_idx] == 0):
        return
        
    # Get annotations for top activating tokens
    top_annotations = [annotations_list[idx] for idx in top_k_indices[:, batch_idx]]
    
    # Count occurrences of each annotation
    annotation_counts = Counter([
        ann for ann_list in top_annotations
        for ann in ann_list
        if ann not in excluded_annotations
    ])
    
    # Find annotations that appear frequently
    common_annotations = {
        ann for ann, count in annotation_counts.items()
        if count >= min_threshold
    }
    
    # Store and display results if meaningful annotations found
    if common_annotations:
        latent_dict[latent_id] = common_annotations
        print_latent_analysis(
            latent_id, common_annotations, 
            tokens_array[top_k_indices[:, batch_idx]],
            top_annotations, 
            top_k_values[:, batch_idx]
        )


def print_latent_analysis(latent_id, common_annotations, top_tokens, top_annotations, top_activations):
    """Print analysis results for a latent unit."""
    print(f"\nLatent {latent_id} appears to detect: {common_annotations}")
    print("Top 20 activating tokens and their annotations:")
    
    for token, anns, act in zip(top_tokens, top_annotations, top_activations):
        print(f"Token: {token}, Annotations: {anns}, Activation: {act:.3f}")


def cleanup_memory(batch_latents, top_k_indices, top_k_values):
    """Free memory after processing a batch."""
    del batch_latents
    del top_k_indices
    del top_k_values
    torch.cuda.empty_cache()
    gc.collect()

def safe_get_annotations(ann_entry):
    """Helper function to safely process annotations"""
    if isinstance(ann_entry, str):
        try:
            return eval(ann_entry)
        except:
            return []
    return ann_entry


"""
def _parse_annotation_entry(entry):
    if isinstance(entry, set):
        # Convert set to list
        annotation = list(entry)
        thresholds = None
    elif isinstance(entry, list):
        # Use list directly
        annotation = entry
        # Check if third element exists and is numeric for threshold
        thresholds = [entry[2]] if len(entry) > 2 and isinstance(entry[2], (int, float)) else None
    else:
        # Wrap single item in list
        annotation = [entry] if entry else ["unknown"]
        thresholds = None
        
    return annotation, thresholds
"""

def _get_best_result(results, latent_id, annotation):
    """
    Helper function to extract the best result based on F1 score.
    
    Parameters:
    -----------
    results : list
        List of result tuples from compute_metrics_across_thresholds.
    latent_id : int or str
        ID of the latent unit.
    annotation : str
        Annotation of the latent unit.
        
    Returns:
    --------
    tuple or None
        The result tuple with the highest F1 score, or None if results is empty.
    """
    if not results:
        print(f"Warning: No results found for latent {latent_id} with annotation {annotation}")
        return None
    
    # Find the result with the highest F1 score (assumed to be at index 3)
    return max(results, key=lambda x: x[3])

def save_latent_results(results_df, output_path='latent_annotation_results.csv'):

    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return output_path

# turn results_df into a dict
def results_df_to_dict(results_df):
    """Convert results DataFrame to a dictionary."""
    return results_df.set_index('latent_id').T.to_dict('list')



if __name__ == '__main__':
    
    # Load the latent dictionary
    latent_dict = pd.read_csv("/home/maiwald/Downloads/sae_for_glm/annotated_seqs/plasmidpretrainedNT_SAElatent_results_val012.csv")
    latent_dict = latent_dict[['latent_id', 'annotation']].set_index('latent_id')['annotation'].to_dict()
    print(latent_dict)

    # Load the token df pkl file
    token_df = pd.read_pickle("/home/maiwald/Downloads/sae_for_glm/annotated_seqs/token_df_1k_ss0_standardized.pkl")

    # Generate random activations with similar mean/variance
    fake_activations = np.random.normal(loc=2, scale=1, size=(len(token_df), 10))
    fake_activations = torch.tensor(fake_activations)

    # Get particular single latent, annotaiton pair
    latent_id = 2
    annotation = latent_dict[latent_id]

    results = process_single_latent(latent_id, 
                            [annotation], 
                            fake_activations, 
                            token_df, 
                            compute_metrics_across_thresholds, 
                            print_metrics, 
                            0)
