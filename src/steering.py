def modified_forward(
    self,
    input_ids: torch.LongTensor,
    injection_layer: int,
    tokenizer,
    sae_model, # Pass the SAE model as an argument
    latent_id:int, # Pass the latent ID to steer
    act_increase:int, # Pass the activation increase value
    top_k: int = 5

) -> torch.Tensor: # Return logits
    """
    Modified forward pass that adds a steering vector (generated from SAE)
    to the residual stream at a specific layer.
    """
    # Get embeddings (will inherit device from input_ids)
    hidden_states = self.get_input_embeddings()(input_ids)

    # Forward through each layer
    for i, layer_module in enumerate(self.esm.encoder.layer):
        hidden_states = layer_module(hidden_states)[0]
        # Add steering vector at specified layer
        if i == injection_layer:
            print(f"Generating and adding steering vector at layer: {injection_layer}")
            # Generate the steering vector based on the current hidden_states
            vec_steer = get_steering_vec(sae_model, latent_id, act_increase, hidden_states.float())
            hidden_states =  vec_steer

    # Get logits
    logits = self.lm_head(hidden_states)
    return logits


def inject_vector_into_model(
    model: PreTrainedModel,
    input_ids: torch.LongTensor,
    tokenizer,
    injection_layer: int,
    mask_positions_batch: list[list[int]],
    sae_model=None,
    latent_id=None,
    act_increase=None,
    top_k: int = 5,
    print_table: bool = True
) -> tuple[list[list[tuple[str]]], list[list[float]], list[dict]]:

    original_forward = model.forward
    model.forward = modified_forward.__get__(model)

    batch_tokens = []
    batch_probs = []

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            injection_layer=injection_layer,
            tokenizer=tokenizer,
            top_k=top_k,
            sae_model=sae_model,
            latent_id= latent_id,
            act_increase=act_increase
        )
        print(f"Shape of logits from model: {logits.shape}")


    # Process each sequence in the batch
    for batch_idx, sequence_logits in enumerate(logits):

        results = get_top_k_predictions(
            sequence_logits,
            tokenizer,
            [mask_positions_batch[batch_idx]],
            top_k=top_k
        )

        if results:  # Check if results is not empty
            print(f"Results for sequence {batch_idx}: {results}")
            tokens, probs = results[0], results[1]
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        else:
            batch_tokens.append([])
            batch_probs.append([])

    model.forward = original_forward
    return batch_tokens, batch_probs


def get_steering_vec(sae_model: torch.nn.Module,
                     latent_id: int, 
                     act_increase: int, 
                     original_act: torch.Tensor):

    """
    add doc

    """
    # original_act might have shape (1, seq_len, hidden_size)
    if len(original_act.shape) == 3:
        original_act = original_act.squeeze(0).float() 

    # Now vec has shape (seq_len, hidden_size) which should be acceptable to the SAE
    _, _, sae_latent_act,_,_,_,_ = sae_model(original_act)

    sae_latent_act[:, latent_id] += act_increase

    vec_steer = sae_latent_act @ sae_model.W_dec + sae_model.b_dec
    # vec_steer will have shape (seq_len, hidden_size)

    return vec_steer.unsqueeze(0) # Add back the batch dimension to match hidden_states


def get_top_k_predictions(logits: torch.Tensor, tokenizer, mask_position: int, top_k: int = 5) -> tuple[list[str], list[float]]:
    """
    Get top-k predictions for a specific masked position.

    Args:
        logits: Tensor of shape (sequence_length, vocab_size)
        tokenizer: Tokenizer for decoding predictions
        mask_position: Position of the masked token to predict
        top_k: Number of top predictions to return
    """
    # check logit shape
    if len(logits.shape) != 2:
        raise ValueError("Logits tensor should have shape (sequence_length, vocab_size)")

    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)

    # Get top-k predictions for masked token position
    top_k_probs, top_k_indices = probs[mask_position].topk(top_k)

    print(f"Shape of top_k_indices: {top_k_indices.shape}")
    print(f"Shape of top_k_probs: {top_k_probs.shape}")

    if len(top_k_indices.shape) == 2:
        # getting rid of unnecessary batch dim
        top_k_indices = top_k_indices.squeeze()
        top_k_probs = top_k_probs.squeeze()

    # Move to CPU for tokenizer decoding
    top_k_indices = top_k_indices.cpu()
    top_k_probs = top_k_probs.cpu()

    # Decode tokens
    tokens = [tokenizer.decode(idx.item()) for idx in top_k_indices]
    probs = top_k_probs.tolist()

    return tokens, probs



if __name__ == "__main__":
    
    input_seqs = df_train['sequence'].sample(n=1)
    tokenised_seqs = tokenizer_nt(input_seqs.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = tokenised_seqs['input_ids']
    attention_mask = tokenised_seqs['attention_mask']

    rand_mask_pos_batch = [2]

    top_k_tokens, top_k_probs = inject_vector_into_model(
        model=model_nt.to('cuda'),
        input_ids=input_ids.to('cuda'),
        injection_layer=11, # remember that layer idx starts at 0
        top_k=10,
        print_table=True,
        tokenizer=tokenizer_nt,
        mask_positions_batch=rand_mask_pos_batch,
        sae_model=sae_L12,
        latent_id=2222, # remember: starts at 0 here. and at 1 on dashboard
        act_increase=13
        )

    original_input = [tokenizer_nt.decode(ids) for ids in input_ids]

    # Print the top-k tokens along the probability assigned to them
    print_predictions_table(tokens=top_k_tokens[0], probabilities=top_k_probs[0], input_text=original_input[0], injection_layer=12)