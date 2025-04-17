# Filename: asr_dataset_tpu.py (or similar)

from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
import warnings

# Assuming H4Tokenizer is importable, adjust path if necessary
from .tokenizer import H4Tokenizer
# Placeholder if tokenizer is passed directly:
# class H4Tokenizer: # Placeholder
#     def __init__(self):
#         self.eos_id = 2
#         self.sos_id = 1
#         self.pad_id = 0
#     def encode(self, text):
#         # Dummy encode for placeholder
#         return [ord(c) % 100 for c in text[:10]]


'''
Specification:
The ASRDataset class provides data loading and processing for ASR (Automatic Speech Recognition):

1. Data Organization:
   - Handles dataset partitions (train-clean-100, dev-clean, test-clean)
   - Features stored as .npy files in fbank directory
   - Transcripts stored as .npy files in text directory
   - Loads data on-demand in __getitem__ to conserve memory.

2. Feature Processing:
   - Loads log mel filterbank features from .npy files.
   - Supports multiple normalization strategies:
     * global_mvn: Global mean and variance normalization. Can calculate stats during training set init.
     * cepstral: Per-utterance mean and variance normalization.
     * none: No normalization.
   - Applies SpecAugment data augmentation during training.

3. Transcript Processing:
   - Creates shifted (SOS-prefixed) and golden (EOS-suffixed) versions.
   - Handles tokenization using H4Tokenizer.

4. Batch Preparation:
   - Pads features and transcripts to batch-uniform lengths via collate_fn.
   - Provides lengths for potential packed sequence processing elsewhere.
'''

class ASRDatasetTPU(Dataset): # Renamed class for clarity
    def __init__(
            self,
            partition:Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config:dict,
            tokenizer:H4Tokenizer,
            isTrainPartition:bool,
            global_stats:Optional[Tuple[torch.Tensor, torch.Tensor]]=None
    ):
        """
        Initialize the ASRDataset for ASR training/validation/testing, loading data on demand.

        Args:
            partition (str): Dataset partition ('train-clean-100', 'dev-clean', or 'test-clean')
            config (dict): Configuration dictionary containing dataset settings (root, norm, num_feats, specaug etc.)
            tokenizer (H4Tokenizer): Tokenizer for encoding/decoding text
            isTrainPartition (bool): Whether this is the training partition.
            global_stats (tuple, optional): (mean, std) computed from training set.
                                          If None and using 'global_mvn' for the training set,
                                          stats will be computed during this init.
                                          Required for dev/test sets if using 'global_mvn'.
        """
        self.config    = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # Set up data paths
        self.fbank_dir = os.path.join(config['root'], partition, 'fbank')
        try:
            self.fbank_files = sorted([os.path.join(self.fbank_dir, f) for f in os.listdir(self.fbank_dir) if f.endswith('.npy')])
        except FileNotFoundError:
             raise FileNotFoundError(f"Feature directory not found: {self.fbank_dir}")

        subset_size = config.get('subset_size', len(self.fbank_files))
        self.fbank_files = self.fbank_files[:subset_size]
        self.length = len(self.fbank_files)
        if self.length == 0:
            warnings.warn(f"Warning: No feature files found or selected for partition {partition}.")

        # Store file paths, DO NOT load data here
        self.text_files = []
        if self.partition != "test-clean":
            self.text_dir = os.path.join(config['root'], partition, 'text')
            try:
                self.text_files = sorted([os.path.join(self.text_dir, f) for f in os.listdir(self.text_dir) if f.endswith('.npy')])
            except FileNotFoundError:
                raise FileNotFoundError(f"Text directory not found: {self.text_dir}")
            self.text_files = self.text_files[:subset_size]
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError(f"Number of feature ({len(self.fbank_files)}) and transcript ({len(self.text_files)}) files must match for {partition}")

        # Initialize stats attributes
        self.global_mean = None
        self.global_std = None

        # Handle global_mvn statistics
        if self.config.get('norm', 'none') == 'global_mvn':
            if global_stats is not None:
                # Stats are provided, use them
                self.global_mean, self.global_std = global_stats
                if not isinstance(self.global_mean, torch.Tensor) or not isinstance(self.global_std, torch.Tensor):
                     raise TypeError("global_stats must be a tuple of two torch.Tensors (mean, std)")
                print(f"Using provided global stats for {partition}.")
            elif isTrainPartition:
                # Training partition AND stats are NOT provided: Calculate them now.
                print(f"Calculating global stats from training data ({partition})...")
                # Initialize Welford's algorithm accumulators
                count = 0
                # Use float64 for accumulators to maintain precision
                mean_acc = torch.zeros(self.config['num_feats'], dtype=torch.float64)
                M2 = torch.zeros(self.config['num_feats'], dtype=torch.float64)

                # Iterate through all training feature files *just* to calculate stats
                for f_path in tqdm(self.fbank_files, desc="Calculating Stats", unit="file"):
                    try:
                        feat = np.load(f_path)
                        feat = feat[:self.config['num_feats'], :] # Truncate features dim if necessary
                        if feat.shape[1] == 0: # Skip empty features
                            continue
                        # Use float64 for calculation precision
                        feat_tensor = torch.from_numpy(feat).to(dtype=torch.float64) # (num_feats, time)

                        # Welford's online algorithm update
                        batch_count = feat_tensor.shape[1] # Number of time steps in this file
                        delta = feat_tensor - mean_acc.unsqueeze(1) # Shape: (num_feats, time)
                        mean_acc += delta.sum(dim=1) / (count + batch_count) # Update mean incorporating new data points
                        delta2 = feat_tensor - mean_acc.unsqueeze(1) # Use updated mean
                        M2 += (delta * delta2).sum(dim=1) # Update sum of squares of differences
                        count += batch_count # Update total count of time steps processed

                    except Exception as e:
                        print(f"\nWarning: Error processing file {f_path} during stats calculation: {e}. Skipping file.")

                # Compute final variance and standard deviation
                if count > 1:
                    variance = M2 / (count - 1) # Use count-1 for unbiased sample variance
                    self.global_std = torch.sqrt(variance.clamp_(min=1e-8)).float() # clamp_ inplace
                    self.global_mean = mean_acc.float()
                    print("Global stats calculation complete.")
                    print(f"  Calculated Mean shape: {self.global_mean.shape}, Std shape: {self.global_std.shape}")
                else:
                    self.global_mean = None # Keep as None if calculation failed
                    self.global_std = None
                    warnings.warn(f"Warning: Not enough data points ({count}) processed to calculate reliable global stats for {partition}. Global stats set to None.")

            else:
                # Dev/Test partition AND stats are NOT provided: Raise error
                 raise ValueError("global_stats must be provided for dev/test sets when using global_mvn")
        # --- End global_mvn handling ---

        # Approximation for max lengths (or load pre-computed values) - NOT tracked during on-demand loading
        # These might be needed by the model, pass via config if required.
        self.feat_max_len = config.get('feat_max_len', -1) # Indicate not tracked unless provided
        self.text_max_len = config.get('text_max_len', -1) # Indicate not tracked unless provided

        # Initialize SpecAugment transforms (if config enables it)
        self.time_mask = None
        self.freq_mask = None
        if config.get("specaug", False):
             specaug_conf = config.get("specaug_conf", {})
             self.time_mask = tat.TimeMasking(
                 time_mask_param=specaug_conf.get('time_mask_width_range', 80), # Provide default
                 iid_masks=True
             )
             self.freq_mask = tat.FrequencyMasking(
                 freq_mask_param=specaug_conf.get('freq_mask_width_range', 27), # Provide default
                 iid_masks=True
             )

        print(f"ASRDatasetTPU for {partition} initialized ({self.length} samples, on-demand loading).")

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get a single sample from the dataset by loading data from disk.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (features, shifted_transcript, golden_transcript) where:
                - features: FloatTensor of shape (num_feats, time)
                - shifted_transcript: LongTensor (time) or None if test set/error
                - golden_transcript: LongTensor (time) or None if test set/error
                Returns (None, None, None) on failure to allow collate_fn to filter.
        """
        # Load feature file for this index
        feat_path = self.fbank_files[idx]
        try:
            feat = np.load(feat_path)
            feat = feat[:self.config['num_feats'], :] # Truncate features dim
            if feat.shape[1] == 0: # Handle empty features
                 print(f"Warning: Empty feature file encountered: {feat_path}. Skipping index {idx}.")
                 return None, None, None # Signal failure for collate_fn
            feat = torch.FloatTensor(feat) # (num_feats, time)
        except Exception as e:
            print(f"Error loading feature file {feat_path} at index {idx}: {e}")
            return None, None, None # Signal failure for collate_fn

        # Apply normalization
        norm_type = self.config.get('norm', 'none')
        if norm_type == 'global_mvn':
            # Check if stats were successfully loaded/computed
            if self.global_mean is None or self.global_std is None:
                 warnings.warn(f"Global stats not available for normalization, but global_mvn requested for {self.partition}. Skipping normalization for sample {idx}.", RuntimeWarning)
            else:
                 feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif norm_type == 'cepstral':
            feat_mean = feat.mean(dim=1, keepdim=True)
            feat_std = feat.std(dim=1, keepdim=True)
            feat = (feat - feat_mean) / (feat_std + 1e-8)
        # elif norm_type == 'none': pass

        # Load transcript file for this index (if not test set)
        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            if idx < len(self.text_files):
                text_path = self.text_files[idx]
                try:
                    # Load list of chars, join to string
                    transcript_chars = np.load(text_path, allow_pickle=True).tolist()
                    transcript = ''.join(transcript_chars)

                    tokenized = self.tokenizer.encode(transcript)
                    if not tokenized: # Handle empty transcripts after tokenization
                         print(f"Warning: Empty transcript after tokenization for file {text_path} at index {idx}. Skipping.")
                         return None, None, None # Signal failure

                    shifted_transcript = torch.LongTensor([self.sos_token] + tokenized)
                    golden_transcript = torch.LongTensor(tokenized + [self.eos_token])
                except Exception as e:
                    print(f"Error loading/processing transcript file {text_path} at index {idx}: {e}")
                    return None, None, None # Signal failure
            else:
                # Should not happen if lengths match in __init__, but safety check
                print(f"Warning: Index {idx} out of bounds for text files.")
                return None, None, None # Signal failure

        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Collate and pad a batch of samples, handle failed loads, apply SpecAugment.

        Args:
            batch (list): List of samples from __getitem__. Samples can be None or contain None transcripts on load failure.

        Returns:
            tuple: (padded_features, padded_shifted, padded_golden, feat_lengths, transcript_lengths)
                   Returns tensors even for empty batches after filtering, ensuring type consistency.
        """
        # Filter out samples where loading failed in __getitem__
        # A valid item is not None and has non-None transcripts (unless test set)
        if self.partition != "test-clean":
            valid_batch = [item for item in batch if item is not None and item[1] is not None and item[2] is not None]
        else:
            valid_batch = [item for item in batch if item is not None] # Test set only needs features

        if not valid_batch:
             # Handle empty batch case - return appropriately shaped zero tensors
             print("Warning: collate_fn creating dummy batch due to all samples failing load.")
             # Determine shapes based on config (use small defaults if not available)
             num_feats = self.config.get('num_feats', 80)
             # Return Batch size 1, Time 1, Feat dim / Text len 1
             dummy_feat = torch.zeros((1, 1, num_feats))
             dummy_len = torch.ones(1, dtype=torch.long)
             if self.partition != "test-clean":
                  dummy_shifted = torch.ones((1, 1), dtype=torch.long) * self.pad_token
                  dummy_golden = torch.ones((1, 1), dtype=torch.long) * self.pad_token
                  dummy_transcript_len = torch.ones(1, dtype=torch.long)
                  return dummy_feat, dummy_shifted, dummy_golden, dummy_len, dummy_transcript_len
             else:
                  return dummy_feat, None, None, dummy_len, None

        # --- Process Valid Batch ---
        # Features (B x T x F)
        # Assume __getitem__ returns (F, T), need to transpose before padding
        try:
             batch_feats_transposed = [item[0].transpose(0, 1) for item in valid_batch] # List of (T, F)
             feat_lengths = torch.LongTensor([feat.shape[0] for feat in batch_feats_transposed])
             padded_feats = pad_sequence(batch_feats_transposed, batch_first=True, padding_value=0.0) # Use float padding
        except Exception as e:
             print(f"Error during feature padding: {e}")
             # Handle error - maybe return dummy batch? Or raise?
             # Let's try to return a dummy batch to avoid crashing the trainer immediately
             num_feats = self.config.get('num_feats', 80)
             dummy_feat = torch.zeros((1, 1, num_feats))
             dummy_len = torch.ones(1, dtype=torch.long)
             dummy_shifted = torch.ones((1, 1), dtype=torch.long) * self.pad_token if self.partition != 'test-clean' else None
             dummy_golden = torch.ones((1, 1), dtype=torch.long) * self.pad_token if self.partition != 'test-clean' else None
             dummy_transcript_len = torch.ones(1, dtype=torch.long) if self.partition != 'test-clean' else None
             return dummy_feat, dummy_shifted, dummy_golden, dummy_len, dummy_transcript_len


        # Transcripts
        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            try:
                batch_shifted = [item[1] for item in valid_batch]
                batch_golden = [item[2] for item in valid_batch]
                # Calculate lengths *before* padding
                transcript_lengths = torch.LongTensor([len(t) for t in batch_shifted])
                padded_shifted = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token)
                padded_golden = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)
            except Exception as e:
                print(f"Error during transcript padding: {e}")
                # Set transcript related outputs to None or dummy values if padding fails
                padded_shifted, padded_golden, transcript_lengths = None, None, None
                # Potentially return the features that *did* pad correctly? Or fail the batch?
                # For simplicity, let's allow returning features even if transcripts fail padding here.
                # The training loop needs to handle None transcripts if this occurs.

        # SpecAugment (Apply only on training partition AFTER padding)
        if self.config.get("specaug", False) and self.isTrainPartition and self.time_mask and self.freq_mask:
            # Permute to (B x F x T) for torchaudio transforms
            padded_feats_permuted = padded_feats.permute(0, 2, 1)

            specaug_conf = self.config.get("specaug_conf", {})
            if specaug_conf.get("apply_freq_mask", False):
                for _ in range(specaug_conf.get("num_freq_mask", 1)):
                    padded_feats_permuted = self.freq_mask(padded_feats_permuted)
            if specaug_conf.get("apply_time_mask", False):
                for _ in range(specaug_conf.get("num_time_mask", 1)):
                    # Ensure time_mask is applied correctly even if lengths vary?
                    # TimeMasking typically works on the padded tensor directly.
                    padded_feats_permuted = self.time_mask(padded_feats_permuted) # Check if needs lengths? Docs say no.

            # Permute back to (B x T x F)
            padded_feats = padded_feats_permuted.permute(0, 2, 1)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
