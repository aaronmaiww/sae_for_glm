import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def calculate_all_probabilities(dataset, annotation='PuroR'):
    # Create one-time annotation mask
    annotation_mask = dataset['token_annotations'].str.contains(annotation)

    # Group by tokens and calculate probabilities in one go
    # Count total occurrences of each token
    token_counts = dataset.groupby('tokens').size()

    # Count occurrences where both token and annotation are present
    token_annotation_counts = dataset[annotation_mask].groupby('tokens').size()

    # Calculate probabilities using vectorized operations
    probabilities = token_annotation_counts.divide(token_counts, fill_value=0)

    # Convert to dictionary
    return probabilities.to_dict()

# Calculate activation statistics using groupby
def calculate_activation_stats(df, column):
    # Group by tokens and calculate mean and max in one operation
    activation_stats = df.groupby('tokens')[column].agg(['mean', 'max'])

    # Convert to dictionaries
    avg_act_dict = activation_stats['mean'].to_dict()
    max_act_dict = activation_stats['max'].to_dict()

    return avg_act_dict, max_act_dict

def plot_correlation(results_df):

    # Plot correlation between 'P(annotation | type)' and 'Avg Activation'
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['P(annotation | type)'], results_df['Avg Activation'])
    plt.xlabel("P(annotation | type)")
    plt.ylabel("Avg Activation")
    plt.grid(True)
    plt.show()

    # Optional: Add correlation coefficient to the plot
    correlation = results_df['P(annotation | type)'].corr(results_df['Avg Activation'])
    print(f"Correlation coefficient: {correlation:.4f}")



    # Plot correlation between 'P(annotation | type)' and 'Max Activation'
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['P(annotation | type)'], results_df['Max Activation'])
    plt.xlabel("P(annotation | type)")
    plt.ylabel("Max Activation")
    plt.grid(True)
    plt.show()

    # Optional: Add correlation coefficient to the plot
    correlation = results_df['P(annotation | type)'].corr(results_df['Max Activation'])
    print(f"Correlation coefficient: {correlation:.4f}")




if __name__ == "__main__":
    # Load the dataset
    token_df = pd.read_csv('data/tokens.csv')

    # Define latent ID
    latent_id = 0

    # Make a copy of the DataFrame
    token_df_copy = token_df.copy()

    # Define latent column
    latent_col = f"latent-{latent_id}-act"

    # Get activation statistics
    avg_act_dict, max_act_dict = calculate_activation_stats(token_df_copy, latent_col)
    dict_type_p = calculate_all_probabilities(token_df_copy, annotation='PuroR')

    # Combine all results into a DataFrame for visualization
    results_df = pd.DataFrame({
        'P(annotation | type)': pd.Series(dict_type_p),
        'Avg Activation': pd.Series(avg_act_dict),
        'Max Activation': pd.Series(max_act_dict)
    })

    # Display the combined results
    plot_correlation(results_df)

