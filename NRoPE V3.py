# ===========| Imports |======================================#
import torch
from typing import *
import math as m
import time as t

# ===========| NroPE V3 Settings Class |======================================#
class NRoPE_Settings:
    BASE_FREQUENCY = 10e3
    LRP = 17  # Learning Range Percentage
    ARM = {"EMBEDDING":2 / 3, "ATTENTION":2.0}  # Angle Range Multiplier
    MIN_L = 1 - LRP / 100
    SDT_L = (BASE_FREQUENCY - MIN_L * BASE_FREQUENCY)*2
    def FREQUENCY_OPTIMIZATION(self, param): return (actv.sigmoid(param) * NRoPE_Settings.SDT_L) + NRoPE_Settings.MIN_L*NRoPE_Settings.BASE_FREQUENCY
    
# ===========| PWQ NroPE V3 Class |=====================================#
class PWQ_NRoPE_V3:
    def __init__(self, embedding_array: torch.Tensor, neutralize_angle: bool = True,angle_neutralization_range: Union[int, float] = 2 * torch.pi,base_frequency: Union[int, float] = 10e3):
        """
        Neutralized Rotary Positional Encoding, consiting of neutralizing the angle between 0 and ANR parameter, and 
        making the base frequency an adaptive, learnable parameter, 
        however limiting it to (actv.sigmoid(param) * NRoPE_Settings.SDT_L) + NRoPE_Settings.MIN_L*NRoPE_Settings.BASE_FREQUENCY
        """
        self.embedded = embedding_array
        self.batch_size = self.embedded.shape[0]
        self.seq_len = self.embedded.shape[1]
        self.embedding_dim = self.embedded.shape[2]

        self.pair_count = self.embedding_dim // 2
        self.neutralize_angle = neutralize_angle
        self.base_frequency = base_frequency
        self.anr = angle_neutralization_range

        self._rotate()

    def _rotate(self):
        device = self.embedded.device

        # 1. Create Frequency and Position tensors
        POS = torch.arange(self.seq_len, device=device).to(self.embedded.dtype)
        PAIRS = torch.arange(self.pair_count, device=device).to(self.embedded.dtype)

        # 2. Compute Frequencies: base_freq ^ (-2i / dim)
        # Note: self.embedding_dim is used as the 'd' in the formula
        Freqs = self.base_frequency ** (-2 * PAIRS / self.embedding_dim)

        # 3. Compute Angles [seq_len, pair_count]
        Angles = torch.outer(POS, Freqs)
        if self.neutralize_angle:
            Angles = Angles % self.anr

        # 4. Prepare for Broadcasting: [1, seq_len, pair_count]
        # We need the 1 at the start so it applies identically to every item in the Batch
        sin = torch.sin(Angles).unsqueeze(0)
        cos = torch.cos(Angles).unsqueeze(0)

        # 5. Reshape embedded into pairs: [batch, seq_len, pair_count, 2]
        embed_pairs = self.embedded.view(self.batch_size, self.seq_len, self.pair_count, 2)

        x1 = embed_pairs[..., 0]
        x2 = embed_pairs[..., 1]

        # 6. Apply Rotation Matrix
        # [x1, x2] * [cos, -sin] = [x1*cos - x2*sin, x1*sin + x2*cos]
        # [x1, x2] * [sin,  cos]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # 7. Stack and Reshape back to [batch, seq_len, embedding_dim]
        self.embedded = torch.stack([rotated_x1, rotated_x2], dim=-1).view(self.batch_size, self.seq_len,self.embedding_dim)
