import biopython
from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML  # Add this import
from tqdm import tqdm
import time

# Create a FASTA file from the sequences
def create_fasta_from_df(df, output_file):
    with open(output_file, 'w') as f:
        for idx, row in df.iterrows():
            # Get the sequence and remove spaces and '|'
            seq = row['context'].replace(' ', '').replace('|', '')

            # Write in FASTA format with sequence ID and the sequence
            f.write(f">sequence_{idx}\n{seq}\n")

def blast_sequence(seq, database="nr"):
    try:
        # Run BLAST search
        result_handle = NCBIWWW.qblast(
            "blastn",                     # nucleotide BLAST
            database,                     # nucleotide database
            seq,
            expect=e_threshold,                 # E-value threshold
            hitlist_size=n_hits                # Number of hits to return
        )
        return result_handle
    except Exception as e:
        print(f"Error during BLAST: {e}")
        return None

def analyze_blast_results(blast_record):
    hiv_related = False
    for alignment in blast_record.alignments:
        if any(term.lower() in alignment.title.lower()
               for term in ['hiv', 'lentivirus', 'immunodeficiency virus']):
            hiv_related = True
            break
    return hiv_related


if __name__ == "__main__":
    # Create the FASTA file
    output_file = "sequences.fasta"
    top_N = 100

    # Sort and store the result, then take top N rows
    sorted_df = token_df_copy.sort_values(f"latent-{latent_id}-act", ascending=False)
    top_sequences = sorted_df.head(top_N)

    create_fasta_from_df(top_sequences, output_file)

    # Verify the file contents
    with open(output_file, 'r') as f:
        print("First few sequences in the FASTA file:")
        print(f.read().strip()[:500])  # Print first 500 characters as preview

    # Config for BLAST search
    Entrez.email = "maiwald.aaron@outlook.de"
    n_hits = 30
    e_threshold = 1e-10

    # Assuming your sequences are in a FASTA file
    sequences = []  # Store your sequences here
    hiv_matches = 0

    # Read your sequences (modify this part based on how your sequences are stored)
    with open('sequences.fasta', 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            sequences.append(str(record.seq))

    # Process each sequence
    for i, seq in enumerate(tqdm(sequences)):

        print(f"Processing sequence {i+1}/{len(sequences)}")
        result_handle = blast_sequence(seq)

        if result_handle:
            print("Parsing BLAST results...")
            # Parse BLAST results
            blast_records = NCBIXML.parse(result_handle)

            for blast_record in blast_records:
                if analyze_blast_results(blast_record):
                    hiv_matches += 1
                    print("HIV/lentivirus match found!")


        # NCBI recommends waiting between requests
        time.sleep(3)

    # Calculate and print results
    match_percentage = (hiv_matches / len(sequences)) * 100
    print(f"\nResults:")
    print(f"Total sequences: {len(sequences)}")
    print(f"HIV/lentivirus matches: {hiv_matches}")
    print(f"Percentage of matches: {match_percentage:.2f}%")
