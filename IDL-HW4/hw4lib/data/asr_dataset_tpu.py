# paste-4.txt (Modified ASRDataset)

from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer

# ... (Keep imports and class docstring) ...

class ASRDatasetTPU(Dataset):
    def __init__(
            self,
            partition:Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config:dict,
            tokenizer:H4Tokenizer,
            isTrainPartition:bool,
            global_stats:Optional[Tuple[torch.Tensor, torch.Tensor]]=None
    ):
        self.config    = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        self.fbank_dir = os.path.join(config['root'], partition, 'fbank')
        self.fbank_files = sorted([os.path.join(self.fbank_dir, f) for f in os.listdir(self.fbank_dir) if f.endswith('.npy')])

        subset_size = config.get('subset_size', len(self.fbank_files))
        self.fbank_files = self.fbank_files[:subset_size]
        self.length = len(self.fbank_files)

        # --- Store file paths, DO NOT load data here ---
        self.text_files = []
        if self.partition != "test-clean":
            self.text_dir = os.path.join(config['root'], partition, 'text')
            self.text_files = sorted([os.path.join(self.text_dir, f) for f in os.listdir(self.text_dir) if f.endswith('.npy')])
            self.text_files = self.text_files[:subset_size]
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")

        # --- REMOVE lists storing full data ---
        # self.feats, self.transcripts_shifted, self.transcripts_golden = [], [], []

        # --- Calculate stats on the fly or pass them in ---
        # Need to handle global_mvn differently if not pre-calculated and passed via global_stats
        self.global_mean = None
        self.global_std = None
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            elif isTrainPartition:
                # Option 1: Pre-calculate and save/load stats separately.
                # Option 2: Implement streaming calculation (complex).
                # For now, raise error if not provided for training.
                raise ValueError("global_stats must be provided or calculated separately for training set with global_mvn")
            else:
                 raise ValueError("global_stats must be provided for dev/test sets with global_mvn")

        # --- Keep track of max lengths - Requires iterating file info or loading metadata ---
        # Option 1: Precompute and save/load max lengths separately
        # Option 2: Approximate or set a large fixed value (simpler but less efficient padding)
        # Option 3: Iterate through files once to get lengths (slower init)
        # For simplicity, let's assume they might be set in config or use defaults:
        self.feat_max_len = config.get('feat_max_len', 3000) # Example default
        self.text_max_len = config.get('text_max_len', 500)  # Example default
        # If precise max lengths are needed for padding later, you'll need to compute them.

        # --- Remove char/token counting from init (can be done separately) ---
        # self.total_chars = 0
        # self.total_tokens = 0
        # self.avg_chars_per_token = 0 # Can calculate later if needed

        # --- Initialize SpecAugment ---
        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )
        print(f"ASRDataset for {partition} initialized with {self.length} samples (loading on demand).") # Confirm init finish


    # --- get_avg_chars_per_token: Remove or calculate differently if needed ---
    # def get_avg_chars_per_token(self):
    #     return self.avg_chars_per_token

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get a single sample from the dataset by loading data from disk.
        """
        # --- Load feature file for this index ---
        feat_path = self.fbank_files[idx]
        try:
            feat = np.load(feat_path)
            feat = feat[:self.config['num_feats'], :] # Truncate features
            feat = torch.FloatTensor(feat)
        except Exception as e:
            print(f"Error loading feature file {feat_path} at index {idx}: {e}")
            # Return dummy data or raise error to avoid crashing dataloader workers later
            # Returning dummy data might hide issues but prevent hangs in simple cases
            dummy_feat = torch.zeros((self.config['num_feats'], 10)) # Arbitrary small size
            # Depending on strictness, either raise e or return dummy data with None transcripts
            # raise e
            return dummy_feat, None, None


        # Apply normalization
        if self.config['norm'] == 'global_mvn':
            if self.global_mean is not None and self.global_std is not None:
                feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
            else:
                 # This shouldn't happen if __init__ checks are done correctly
                 print(f"Warning: global_mean/std not available for global_mvn at index {idx}")
        elif self.config['norm'] == 'cepstral':
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)
        # elif self.config['norm'] == 'none': pass

        # --- Load transcript file for this index (if not test set) ---
        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            if idx < len(self.text_files):
                text_path = self.text_files[idx]
                try:
                    transcript = ''.join(np.load(text_path).tolist())
                    tokenized = self.tokenizer.encode(transcript)
                    shifted_transcript = torch.LongTensor([self.sos_token] + tokenized)
                    golden_transcript = torch.LongTensor(tokenized + [self.eos_token])
                except Exception as e:
                    print(f"Error loading transcript file {text_path} at index {idx}: {e}")
                    # Return feat but None transcripts if text fails
                    return feat, None, None
            else:
                # Should not happen if lengths match, but as safety
                print(f"Warning: Index {idx} out of bounds for text files.")
                return feat, None, None

        return feat, shifted_transcript, golden_transcript

    # --- collate_fn remains largely the same ---
    # It receives data already loaded by __getitem__
    def collate_fn(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        # Filter out samples where loading might have failed (returned None transcripts)
        batch = [item for item in batch if item[1] is not None and item[2] is not None]
        if not batch: # If all items in the batch failed loading
            # Need to return tensors of expected shapes/types, maybe size 0 or 1 dummy batch?
            # This depends on how the training loop handles potential empty batches.
            # Returning dummy batch of size 1:
             dummy_feat = torch.zeros((1, 10, self.config['num_feats']))
             dummy_len = torch.ones(1, dtype=torch.long) * 10
             if self.partition != "test-clean":
                  dummy_shifted = torch.ones((1, 5), dtype=torch.long) * self.pad_token
                  dummy_golden = torch.ones((1, 5), dtype=torch.long) * self.pad_token
                  dummy_transcript_len = torch.ones(1, dtype=torch.long) * 5
                  return dummy_feat, dummy_shifted, dummy_golden, dummy_len, dummy_transcript_len
             else:
                  return dummy_feat, None, None, dummy_len, None


        # Features (B x T x F)
        # Transpose feature tensors before padding: (T x F) -> (T x F)
        # Note: __getitem__ returns (F x T), transpose here or in __getitem__?
        # Let's assume __getitem__ returns (F x T), so transpose needed here.
        # batch_feats = [item[0].transpose(0, 1) for item in batch] # If __getitem__ returns (F, T)
        # If __getitem__ returns (F, T) and collate needs (B, T, F), then:
        batch_feats_transposed = [item[0].transpose(0, 1) for item in batch] # List of (T, F) tensors
        feat_lengths = torch.LongTensor([feat.shape[0] for feat in batch_feats_transposed])
        padded_feats = pad_sequence(batch_feats_transposed, batch_first=True, padding_value=0) # (B, T_max, F)

        # Transcripts
        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            batch_shifted = [item[1] for item in batch] # List of (T_text,) tensors
            batch_golden = [item[2] for item in batch]
            # Ensure transcript_lengths matches the filtered batch
            transcript_lengths = torch.LongTensor([len(t) for t in batch_shifted])
            padded_shifted = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token) # (B, T_text_max)
            padded_golden = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)

        # SpecAugment
        if self.config["specaug"] and self.isTrainPartition:
            # Permute to (B x F x T) for torchaudio transforms
            padded_feats = padded_feats.permute(0, 2, 1)
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    padded_feats = self.freq_mask(padded_feats)
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    padded_feats = self.time_mask(padded_feats)
            # Permute back to (B x T x F)
            padded_feats = padded_feats.permute(0, 2, 1)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
