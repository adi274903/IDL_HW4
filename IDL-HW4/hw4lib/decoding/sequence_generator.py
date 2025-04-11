import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # TODO: Implement greedy search
        batch_size = x.size(0)
        sequences = x.clone()
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        

        
        for _ in range(self.max_length - x.size(1)):
        # Check if all sequences have finished
          if finished.all():
              break
          
        # Get logits from the score function
          next_token_logits = self.score_fn(sequences)  # (batch_size, vocab_size)
          
          # Apply repetition penalty
          next_token_logits = self._apply_repeat_penalty(next_token_logits, sequences, repeat_penalty)
          
          # Apply temperature
          next_token_logits = next_token_logits / temperature
          
          # Get log probabilities
          log_probs = torch.log_softmax(next_token_logits, dim=-1)
          
          # Greedy selection: choose the token with highest probability
          next_token_scores, next_tokens = torch.max(log_probs, dim=-1)  # (batch_size,)
          
          # Update scores only for unfinished sequences
          scores = torch.where(finished, scores, scores + next_token_scores)
          
          # Append next tokens to the sequences
          next_tokens = next_tokens.unsqueeze(-1)  # (batch_size, 1)
          sequences = torch.cat([sequences, next_tokens], dim=1)  # (batch_size, seq_len+1)
          
          # Check if any sequence has reached EOS
          is_eos = (next_tokens.squeeze(-1) == self.tokenizer.eos_id)
          finished = finished | is_eos
      
        return sequences, scores


    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling (use 1.0 for standard beam search)
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length), sorted by score
             - scores is of shape (batch_size, beam_width), sorted log probabilities
        """
        # --- Input Validation ---
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if repeat_penalty < 1.0:
             raise ValueError("repeat_penalty must be >= 1.0")

        x = x.to(self.device)
        batch_size = x.size(0)
        start_seq_len = x.size(1)

        # --- Initialization ---
        # Expand input sequences to beam width
        # Shape: (batch_size, beam_width, start_seq_len)
        sequences = x.unsqueeze(1).expand(-1, beam_width, -1)

        # Initial scores: 0 for all beams (log probability space)
        # Shape: (batch_size, beam_width)
        scores = torch.zeros(batch_size, beam_width, dtype=torch.float, device=self.device)

        # Keep track of finished beams
        # Shape: (batch_size, beam_width)
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=self.device)

        # --- Beam Search Loop ---
        for step in range(self.max_length - start_seq_len):
            current_seq_len = sequences.size(-1)

            # Check stopping condition
            if finished.all(dim=1).all():
                 break

            # --- Get Log Probs for Next Token ---

            # Handle the first step separately (works with the original score_fn expectation)
            if step == 0:
                 with torch.no_grad():
                      # Original input shape: (batch_size, start_seq_len)
                      next_token_logits = self.score_fn(x) # Returns (batch_size, vocab_size)

                 # Apply penalty/temp before topk
                 if repeat_penalty != 1.0:
                      next_token_logits = self._apply_repeat_penalty(next_token_logits, x, repeat_penalty)
                 if temperature != 1.0:
                      next_token_logits = next_token_logits / temperature

                 log_probs = torch.log_softmax(next_token_logits.float(), dim=-1)

                 # Get top-k scores and tokens for initial beams
                 # Shape: (batch_size, beam_width)
                 # Ensure scores are float, tokens are long
                 scores, tokens = torch.topk(log_probs, beam_width, dim=-1, largest=True, sorted=True)
                 tokens = tokens.long()

                 # Update sequences by appending the first set of beam tokens
                 # Shape: (batch_size, beam_width, start_seq_len + 1)
                 sequences = torch.cat([sequences, tokens.unsqueeze(-1)], dim=-1)

                 # Update finished status for the first time
                 finished = (tokens == self.tokenizer.eos_id) # Shape: (batch_size, beam_width)
                 # Continue to next iteration
                 continue # Skip the rest of the loop for step 0

            # --- ELSE block (step > 0) ---
            # Here, we must call score_fn carefully to match its expectation

            all_logits_list = [] # To store logits for each beam position
            with torch.no_grad(): # Disable grads for inference calls
                for j in range(beam_width): # Loop through each beam position
                    # Extract sequences for the j-th beam across all batch items
                    # Shape: (batch_size, current_seq_len)
                    sequences_for_beam_j = sequences[:, j, :]

                    # Call score_fn with the expected shape (batch_size, seq_len)
                    # score_fn should return (batch_size, vocab_size)
                    logits_for_beam_j = self.score_fn(sequences_for_beam_j)

                    # Store the result, adding a beam dimension back
                    # Shape: (batch_size, 1, vocab_size)
                    all_logits_list.append(logits_for_beam_j.unsqueeze(1))

            # Concatenate results along the beam dimension
            # Shape: (batch_size, beam_width, vocab_size)
            next_token_logits = torch.cat(all_logits_list, dim=1)

            # --- Apply Penalties & Temperature (Now happens *after* getting all logits) ---
            # Apply repetition penalty (works on the 3D tensor)
            if repeat_penalty != 1.0:
                 next_token_logits = self._apply_repeat_penalty(next_token_logits, sequences, repeat_penalty)

            # Apply temperature
            if temperature != 1.0:
                temp_mask = (~finished).unsqueeze(-1).expand_as(next_token_logits)
                scaled_logits = next_token_logits / temperature
                next_token_logits = torch.where(temp_mask, scaled_logits, next_token_logits)

            # Get log probabilities
            log_probs = torch.log_softmax(next_token_logits.float(), dim=-1)

            # --- Mask Finished Beams ---
            finished_mask = finished.unsqueeze(-1).expand_as(log_probs)
            large_neg = -1e9
            log_probs = torch.where(finished_mask, torch.full_like(log_probs, large_neg), log_probs)
            log_probs[:, :, self.tokenizer.eos_id] = torch.where(finished, torch.zeros_like(scores), log_probs[:, :, self.tokenizer.eos_id])

            # --- Calculate Candidate Scores ---
            candidate_scores = scores.unsqueeze(-1) + log_probs

            # --- Select Top-K Candidates Across Beams ---
            flat_candidate_scores = candidate_scores.view(batch_size, -1)
            top_scores, top_indices = torch.topk(flat_candidate_scores, beam_width, dim=-1, largest=True, sorted=True)

            # --- Determine Origin Beam and Token ID ---
            beam_idx = (top_indices // self.vocab_size).long()
            token_idx = (top_indices % self.vocab_size).long()

            # --- Gather Information for New Beams ---
            parent_sequences = torch.gather(sequences, 1, beam_idx.unsqueeze(-1).expand(-1, -1, current_seq_len))
            parent_finished = torch.gather(finished, 1, beam_idx)

            # --- Update Beams for Next Iteration ---
            sequences = torch.cat([parent_sequences, token_idx.unsqueeze(-1)], dim=-1)
            scores = top_scores
            is_eos = (token_idx == self.eos_id)
            finished = parent_finished | is_eos


        # --- Final Sorting ---
        final_sequences_list = []
        final_scores_list = []
        for batch_idx in range(batch_size):
            sorted_indices = torch.argsort(scores[batch_idx], descending=True)
            final_sequences_list.append(sequences[batch_idx][sorted_indices])
            final_scores_list.append(scores[batch_idx][sorted_indices])

        sequences = torch.stack(final_sequences_list)
        scores = torch.stack(final_scores_list)

        return sequences, scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break

            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            # Check if any sequence has reached EOS 
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]