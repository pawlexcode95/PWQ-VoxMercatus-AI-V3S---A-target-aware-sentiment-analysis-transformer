# ===========| Imports |=====================================#
import os
import torch
import pandas as pd
def set_determinism(seed:int)->None:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.unicode.east_asian_width', False)
set_determinism(42)
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer as autoken
from typing import *
import os
from dotenv import load_dotenv
from PWQ_PON_V3 import PWQ_PON_V_3_0 as pwq_ponv3
import math as m
import csv
from PWQ_OBD_V2 import PWQ_OBD_V2 as pwq_obdv2
from PWQ_VM_AI_Dataset_V3S import PWQ_VM_AI_Dataset_V3S as pwqvmai_dataset_v3s
from PWQ_NRoPE_V3 import PWQ_NRoPE_V3 as pwq_nropev3
import pandas as pd
from tqdm import tqdm
from PWQ_VM_AI__Hypothesis_Test import PWQ_VM_AI_Hypothesis_Proof as pwqvmai_hyp
# ===========| Global Variables |======================================#
load_dotenv(r"C:\Users\pawli\OneDrive\Dokumenty\PWQ-VM-AI\API Data\HuggingFace API Data\.env")
class Global:
    eps: float = 1e-5
    tokenizer: autoken = autoken.from_pretrained("bert-base-uncased", token=os.getenv("ACCESS_TOKEN"))
    device: torch = torch.device("cuda")
    @staticmethod
    def tokenize(input: str, add_special_tokens:bool=True) -> torch.Tensor: return Global.tokenizer(input, return_tensors="pt", add_special_tokens=add_special_tokens)['input_ids']
    def fibonacci_sequence(self, iteration: int):
        start: int = 1
        for i in range(iteration):
            start += 1
            start = start
# ===========| Activation Methods Class |=====================================#
class actv:
    @staticmethod
    def softmax(x: torch.Tensor, dim=0) -> torch.Tensor:
        exp_x: torch.Tensor = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values.detach())
        return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    @staticmethod
    def softplus(x: torch.Tensor, threshold:Union[int,float]=20) -> torch.Tensor:
        return torch.where(x>threshold,x,torch.log1p(torch.exp(x)))
    @staticmethod
    def mish(x: torch.Tensor, threshold:Union[int,float]=20) -> torch.Tensor:
        return x * torch.tanh(torch.where(x>threshold,x,torch.log1p(torch.exp(x))))
    @staticmethod
    def pmish(x: torch.Tensor, a: nn.Parameter, threshold:Union[int,float]=5):
        return x * a* torch.tanh(torch.where(x*a>threshold,x*a,torch.log1p(torch.exp(x*a))))
    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0)
    @staticmethod
    def leaky_relu(x: torch.Tensor,a: Union[int, float]=0.01) -> torch.Tensor:
        return torch.where(x>0, x, x*a)
    @staticmethod
    def elu(x:torch.Tensor, alpha:float=1.0) -> torch.Tensor:
        return torch.where(x>0, x, alpha*(torch.exp(x)-1))
    @staticmethod
    def selu(x: torch.Tensor, gamma:float=1.0507009873554804934193349852946, alpha:float=1.6732632423543772848170429916717) -> torch.Tensor:
        return gamma*actv.elu(x, alpha=alpha)
    @staticmethod
    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0,1 / (1 + torch.exp(-x)),torch.exp(x) / (1 + torch.exp(x)))
    @staticmethod
    def tanh(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)
    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        return x * actv.sigmoid(x)
    @staticmethod
    def gelu(x: torch.Tensor, gamma:float=0.044715) -> torch.Tensor:
        return 1/2 * x * (1+actv.tanh(torch.sqrt(2/torch.pi) * (x * gamma * x ** 3)))
    @staticmethod
    def maxout(x: torch.Tensor) -> torch.Tensor:
        ...
    @staticmethod
    def bent_identity(x: torch.Tensor) -> torch.Tensor:
        return x + (torch.sqrt(x**2+1)-1)/2
    @staticmethod
    def arctan(x: torch.Tensor) -> torch.Tensor:
        return torch.arctan(x)
    @staticmethod
    def sinlu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sin(x)
    @staticmethod
    def trelu(x: torch.Tensor, t:float=0.02) -> torch.Tensor:
        return torch.where(x>t,x, 0)
    @staticmethod
    def gaussian(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x**2))
    @staticmethod
    def hardswish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.clamp(x+3, 0, 6) / 6

# ===========| Settings of PWQ-VM-AI |======================================#
class Settings:
    V_SIZE: int = Global.tokenizer.vocab_size  # The amount of words the AI can learn
    E_DIM: int = 128  # The count of numbers that are describing each word # PWQVMAIV3R -> 512
    HEADS_DIM: int = 8  # The amount # PWQVMAIV3R -> 32
    LAYERS: int = 4  # Amount of transformer layers # PWQVMAIV3R -> 10
    ACTIVATION: actv = actv.pmish  # Activation utilized in FFN of every layer
    FFN_EXP_R: int = 2  # Expansion factor of every FFN layer # PWQVMAIV3R -> 3
    MAX_TOK_LENGTH: int = 256  # Max token length per sentence # PWQVMAIV3R -> 1024
    TEMP: float = 1.25
    SENTIMENTS: List[str] = ["<POS>", "<NEU>", "<NEG>"]  # Index 0 -> Positive | Index 1 -> Neutral | Index 2 -> Negative
    POS: str = SENTIMENTS[0]
    NEU: str = SENTIMENTS[1]
    NEG: str = SENTIMENTS[2]
    #PWQVMAI_DATASET: pwqvmai_dataset = pwqvmai_dataset(r"C:\Users\pawli\OneDrive\Dokumenty\PWQ-VM-AI\PWQ-VM-AI Training Dataset V.2.0\Avatar 3 Files 1.3k Coms\Avatar_3_AI_Refined.csv")
    #SENTENCES: List[str] = list("This iphone is very good product, I think its better than samsung ones.")#list(PWQVMAI_DATASET.dataset.keys())
    #GT_SENTIMENTS: str = dict(["<POS>", "<NEG>"])#[f"<{sentiment[:3].upper()}>" for sentiment in PWQVMAI_DATASET.dataset.values()]
    #TARGET: str = list(["iphone", "samsung"])
    DATASET: List[Dict[str, Dict[str, str]]]  = pwqvmai_dataset_v3s(sentiments=SENTIMENTS, max_sample_count=None).DATASET

# ===========| Mode Configuration of PWQ-VM-AI |======================================#
class Config:
    def __init__(self, mode:str="train", weights_filepath:str=None, epochs:int=None, input_fp:str=None) -> None:
        self.mode = mode
        self.weights_filepath = weights_filepath
        self.epochs = epochs
        self.input_fp = input_fp

class NRoPE_Settings:
    BASE_FREQUENCY = 10e3
    LRP = 17  # Learning Range Percentage
    ARM = {"EMBEDDING":2 / 3, "ATTENTION":2.0}  # Angle Range Multiplier
    MIN_L = 1 - LRP / 100
    SDT_L = (BASE_FREQUENCY - MIN_L * BASE_FREQUENCY)*2
    def FREQUENCY_OPTIMIZATION(self, param): return (actv.sigmoid(param) * NRoPE_Settings.SDT_L) + NRoPE_Settings.MIN_L*NRoPE_Settings.BASE_FREQUENCY

class Dropout_Settings:
    MIN_KR = 0.05
    MAX_KR = 0.50
    STD_KR = (MIN_KR + MAX_KR)/2
    def DROPOUT_KR_OPTIMIZATION(self, param): return (actv.sigmoid(param) * Dropout_Settings.STD_KR) + Dropout_Settings.MIN_KR
    def _get_dropout_scalar_equil(self, rate): return 1 / (1 - rate)

# ===========| Error Classes of PWQ-VM-AI |======================================#
class PWQ_Invalid_Input(NameError):
    def __init__(self, msg):
        super().__init__(f"{msg}")

# ===========| GT Vector Construction |==============================#
class GT_Vector_Construction:
    def __init__(self, batch_idx: int,sentence:str ,target: str) -> None:
        super(GT_Vector_Construction, self).__init__()
        GT = str([v for k, v in Settings.DATASET[batch_idx][sentence].items() if k.lower() == target.lower()][0])
        Index = Settings.SENTIMENTS.index(GT)
        self.GT_Vector = [0.0] * Settings.SENTIMENTS.__len__()
        self.GT_Vector_Dict = {Settings.SENTIMENTS[i]:0.0 for i in range(Settings.SENTIMENTS.__len__())}
        self.GT_Vector_Dict[GT] = 1.0
        self.GT_Vector[Index] = 1.0

# ===========| Initialization |======================================#
class PWQ_Initialization():
    def __init__(self, w: torch.Size, L: int, debug_prints:bool=False) -> None:
        super(PWQ_Initialization, self).__init__()
        """
        After weeks of research PWQ-VM-AI engineered a initializer that outperforms Kaiming by 94.8%, and Orthogonal
        by 86.3%, reaching a mean loss of 3.01549e-02 (0.0301549) and a minimum loss reaching 6.031449e-07 (0.0000006031449).
        It was engineered based off of orthogonal, by slightly breaking/perturbating the perfect orthogonal symmetry of the
        QR decomposition, adding a normally distributed matrix with tiny numbers to still maintain the approximate symmetry
        of 90 deg vectors, yet breaking its perfection. 
        """
        self.Q, self.P = pwq_ponv3.pon(w, L, bias_return=True, perturb_bias=True, debug_prints=debug_prints, mean=0.5, std=1.0) # torch.nn.Parameter() objects

    def init(self) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
        return self.Q, self.P

# ===========| Linear Projection Class |======================================#
class PWQ_Linear_Projection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, debug_prints:bool=False, mode:str=None) -> None:
        super(PWQ_Linear_Projection, self).__init__()
        self.out_dim, self.in_dim = out_dim, in_dim
        self.mode = mode
        self.lp_weight, self.lp_bias = PWQ_Initialization(torch.Size([in_dim, out_dim]) , Settings.LAYERS, debug_prints=debug_prints).init() # Weight: dim = R^[in, out] | Bias: dim = R^[out]
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.matmul(input, self.lp_weight) + self.lp_bias

# ===========| Layer Normalization Class|==================================#
class PWQ_Layer_Norm(nn.Module):
    def __init__(self, in_dim: int):
        super(PWQ_Layer_Norm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(in_dim)) # Gamma is full of ones (1) | "Start at scale 1x"
        self.beta = nn.Parameter(torch.zeros(in_dim)) # Beta is full of zeros (0) | "Start at shift +0"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(input, dim=-1, keepdim=True)
        """
        Keepdim=True -> When using ops like mean,sum,var etc. The dim=x collapses. Keepdim=True -> Dimension stays
        mean = torch.mean(tensor.shape[1, 64, 128], dim=1) => keepdim=True -> mean.shape = [1, 1, 128] | keepdim=False -> mean.shape = [1, 128]
        unbiased=False -> refers to Bessel's Correction in Statistics. unbiased=True -> divide by n-1 | unbiased=False -> divide by n | In ML -> always use unbiased=False
        """
        variance = torch.var(input, dim=-1, unbiased=False, keepdim=True)
        return ( ( input - mean ) / torch.sqrt( variance + Global.eps ) ) * self.gamma + self.beta

# ===========| Multi-Head Self-Attention |======================================#
class PWQ_MultiHead_SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, heads_dim: int, mode:str=None) -> None:
        super(PWQ_MultiHead_SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim # Total values per word
        self.heads_dim = heads_dim # Values per head (per questioner of q - query)
        self.heads = embedding_dim // heads_dim # number of heads (count of questioners of q - query)
        """
        In theory this should be of shape [embed_dim, heads_dim] BUT in practice we'd need embed_dim/heads_dim amount of these weights.
        Instead what we do is we create a single big matrix of weights with shape of [embed_dim, embed_dim], which works out equally.
        """
        self.query = PWQ_Linear_Projection(embedding_dim, embedding_dim, mode=mode)
        self.key = PWQ_Linear_Projection(embedding_dim, embedding_dim, mode=mode)
        self.value = PWQ_Linear_Projection(embedding_dim, embedding_dim, mode=mode)
        self.output_lp = PWQ_Linear_Projection(embedding_dim, embedding_dim, mode=mode)
        self.nrope_frequency = torch.nn.Parameter(torch.tensor([0.0], device=Global.device))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = input.shape
        q = self.query(input).view(batch, tokens, self.heads, self.heads_dim).permute(0, 2, 1, 3).contiguous() # Shape - [batch, heads, tokens, heads_dim]
        k = self.key(input).view(batch, tokens, self.heads, self.heads_dim).permute(0, 2, 1, 3).contiguous() # Shape - [batch, heads, tokens, heads_dim]
        v = self.value(input).view(batch, tokens, self.heads, self.heads_dim).permute(0, 2, 1, 3).contiguous() # Shape - [batch, heads, tokens, heads_dim]
        """
        The process of the PWQ Disentangled NRoPE, first implemented in Microsoft's ReBERTa model, tweaked and adjusted for
        PWQVMAI model and PWQNRoPEV3 concept. In PWQVMAIV3a, I introduced NRoPE procedure over attention's query and key tokens,
        since after research, it appeared that applying it solely over token embedding tensor was silently 'deleting' model's 
        knowledge about positioning of word vectors. Applying it over both attention and token embedding, ensures no data is lost,
        and further NRoPE's rotationary positional matrix is directly injected into query and key. The diffrence is ARM parameter. 
        In token embedding we ensure ARM or Angle Range Multiplier, stays below 2/3, meaning the maximum rotation a distance bet-
        ween 2 vectors can't mathematically cross 2/3π radians. However in attention we allow for 2π radians as maximum, ensuring
        we enrich the data more, where it truly matters. 
        """
        # Split query and key into content and spatial position chunks
        q_content, q_spatial = torch.chunk(q, 2, -1)
        k_content, k_spatial = torch.chunk(k, 2, -1)
        # Reshape for NRoPEV3 rotation
        batch_heads = batch * self.heads
        q_spatial_flatten = q_spatial.view(batch_heads, tokens, -1)
        k_spatial_flatten = k_spatial.view(batch_heads, tokens, -1)
        # Apply NRoPEV3 with ANR set to 2π, over query and key
        self.optimized_frequency = NRoPE_Settings().FREQUENCY_OPTIMIZATION(self.nrope_frequency)
        q_spatial_nrope = pwq_nropev3(
            q_spatial_flatten,
            neutralize_angle=True,
            angle_neutralization_range=NRoPE_Settings.ARM["ATTENTION"]*torch.pi,
            base_frequency=self.optimized_frequency
        ).embedded
        k_spatial_nrope = pwq_nropev3(
            k_spatial_flatten,
            neutralize_angle=True,
            angle_neutralization_range=NRoPE_Settings.ARM["ATTENTION"] * torch.pi,
            base_frequency=self.optimized_frequency
        ).embedded
        # Reshape back into its original form
        q_spatial = q_spatial_nrope.view(batch, self.heads, tokens, -1)
        k_spatial = k_spatial_nrope.view(batch, self.heads, tokens, -1)
        # Concatenate content tensors and spatial position tensors back together.
        q = torch.cat([q_content, q_spatial], dim=-1)
        k = torch.cat([k_content, k_spatial], dim=-1)
        # In attention forward, print frequency
        # print(
        #    f"nrope_freq_base: {self.nrope_frequency.item():.4f}, optimized: {optimized_frequency.item():.4f}")
        """
        (A x B) * (B x C) = (A x C)
        Since in score we care about the relationship of the token dimension, we want the last 2 dimensions to be [_,_,tokens,tokens]
        Thus, in this example multiplying q with k, we need to transpose the last 2 dims of k, so that we multiply a - [_,_,tokens,heads_dim] by b - [_,_,heads_dim,tokens]
        Therefore, matmul will output this shape: -> [batch, heads, tokens, tokens]
        NOTE: We dont have to use .contiguous() after transposal of k, since matmul is an operation that's capable of handling non-contiguous tensors. 
        """
        score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.heads_dim, dtype=torch.float32)) # Shape - [batch, heads, q_tokens, k_tokens]
        """
        The reason why we apply softmax on the last dimension is because we want to get the probability of each key (result) corresponing to each query (question),
        and since after matmul the last dimension got transmitted from k and the second to last got transmitted from q, we apply it on the last dimension (-1).
        """
        att_weights = actv.softmax(score, dim=-1) # Shape - [batch, heads, q_tokens, k_tokens]
        output_token = torch.matmul(att_weights, v) # Shape - [batch, heads, q_tokens, heads_dim]
        output = output_token.permute(0, 2, 1, -1).contiguous()  # Shape - [batch, q_tokens, heads, heads_dim]
        output = output.view(batch, tokens, self.embedding_dim) # Shape - [batch, q_tokens, embedding_dim]
        """
        We return the linear projected version of output, or a weighted sum of each head's result, since we want to mix data from every head. 
        Without this, the data of each head stays separated, thus the logic of network breaks.
        The reason is, we have merged two distinct dimensions into a single one, therefore we need to make sure they're properly mixed. Thus 
        we utilize linear projection and a set of weight and bias. 
        """
        return self.output_lp(output) # Shape - [batch, tokens, embedding_dim]

# ===========| Feed Forward Network - Dual Layer MLP |======================================#
class PWQ_FFN_MLP(nn.Module):
    def __init__(self, embedding_dim: int, expansion_factor: int = 4, activation: actv = actv.pmish, mode:str=None) -> None:
        super(PWQ_FFN_MLP, self).__init__()
        """
        Here in FFN, we typically use a 2-layer MLP, wherein first layer expands the dimension of the input, and second layer reduces it back to original.
        In PWQ-VM-AI, PWQ_Linear_Projection actually acts as a weighted sum/linear projection which consists of multiplying the input by a weight and adding bias. 
        Therefore when we call EXPLay1 later in forward, it acts as the whole layer (matmul by weight and add bias). 
        """
        self.embedding_dim = embedding_dim
        self.r = expansion_factor
        # Stage 1: Massive expansion
        self.EXPLay1 = PWQ_Linear_Projection(embedding_dim, 2*expansion_factor*embedding_dim, mode=mode)  # 512→4096 (2r)
        self.REDLay2 = PWQ_Linear_Projection(2*expansion_factor*embedding_dim, expansion_factor*embedding_dim, mode=mode)  # 4096→2048 (r)
        self.EXPLay3 = PWQ_Linear_Projection(expansion_factor*embedding_dim, 2*expansion_factor*embedding_dim, mode=mode)  # 2048→4096 (2r again)
        self.REDLay4 = PWQ_Linear_Projection(2*expansion_factor*embedding_dim, embedding_dim, mode=mode)  # 4096→512

        self.activation = activation
        if self.activation == actv.pmish:
            self.a_large = nn.Parameter(torch.clamp(torch.randn(2 * self.r * embedding_dim), min=-1, max=1))
            self.a_small = nn.Parameter(torch.clamp(torch.randn(self.r * embedding_dim), min=-1, max=1))
            self.a_default = nn.Parameter(torch.clamp(torch.randn(embedding_dim), min=-1, max=1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Stage 1: 512 → 4096 → 2048
        x = self.EXPLay1(input)
        x = self._activate(x)
        x = self.REDLay2(x)
        x = self.EXPLay3(x)
        x = self._activate(x)
        x = self.REDLay4(x)

        return x

    def _activate(self, act_input: torch.Tensor) -> torch.Tensor:
        if self.activation == actv.pmish:
            if act_input.shape[-1] == self.r * self.embedding_dim:
                return self.activation(act_input, a=self.a_small)
            elif act_input.shape[-1] == 2 * self.r * self.embedding_dim:
                return self.activation(act_input, a=self.a_large)
            else:
                return self.activation(act_input, a=self.a_default)
        if torch.isnan(act_input).any():
            print("NaN input to activation")
        return self.activation(act_input)  # Default case

# ===========| VM-AI Transformer Block |===================================#
class PWQ_VM_AI_Transformer_Block(nn.Module):
    def __init__(self, embedding_dim: int, heads_dim: int, layers: int ,expansion_factor: int=4, activation:actv=actv.pmish, mode:str=None):
        super(PWQ_VM_AI_Transformer_Block, self).__init__()
        self.block_instance_list: nn.ModuleList[nn.ModuleDict[str, nn.Module]] = nn.ModuleList([])
        """
        We're looping through layers, and creating 4 instances of blocks for each layer, and appending their dictionary to
        a ModuleList. Why not use normal List & Dict? Because these work on CPU, and ModuleList/Dict both track the instances
        saved on selected device. 
        """
        for lay in range(layers):
            layer_block: nn.ModuleDict[str, nn.Module] = nn.ModuleDict({
                'attention': PWQ_MultiHead_SelfAttention(embedding_dim, heads_dim, mode=mode),
                'ln_att': PWQ_Layer_Norm(embedding_dim),
                'dropout_att':PWQ_AdpDropout(mode=mode),
                'ffn': PWQ_FFN_MLP(embedding_dim, expansion_factor=expansion_factor, activation=activation, mode=mode),
                'ln_ffn': PWQ_Layer_Norm(embedding_dim),
                'dropout_ffn': PWQ_AdpDropout(mode=mode),
            })
            self.block_instance_list.append(layer_block)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(Global.device)
        for idx, layer in enumerate(self.block_instance_list):
            """
            Meanwhile, in forward() we only call each instance, and reassign input for every layer. At the end we return 
            the output tensor of the very final layer. 
            """
            attention = input + layer['dropout_att'](layer['attention'](layer['ln_att'](input)))
            input = attention + layer['dropout_ffn'](layer['ffn'](layer['ln_ffn'](attention)))
        return input
# ===========| Noise Injection Methods |=========================================#
class PWQ_NoiseInjection(nn.Module):
    def __init__(self, std:float=0.012) -> None:
        super(PWQ_NoiseInjection, self).__init__()
        self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + torch.randn_like(input)*self.std

# ===========| Dropout Layer Methods |=========================================#
class PWQ_AdpDropout(nn.Module):
    def __init__(self, mode:str)-> None:
        super(PWQ_AdpDropout, self).__init__()
        self.mode = mode
        self.kill_rate = torch.nn.Parameter(torch.tensor([(Dropout_Settings.MAX_KR+Dropout_Settings.MIN_KR)/2]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.mode.lower() != "train":
            return input
        self.optimizer_kr = Dropout_Settings().DROPOUT_KR_OPTIMIZATION(self.kill_rate)
        mask = torch.rand_like(input) > self.optimizer_kr
        deactivation_count = (mask == 0).sum().item()
        total_count = mask.numel()
        real_rate = deactivation_count / total_count
        self.scalar = Dropout_Settings()._get_dropout_scalar_equil(real_rate)
        return input * mask * self.scalar

# ===========| Target-Awareness Methods |=========================================#
class PWQ_Target_Awareness:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _get_target_idx_range(target_tokenized: torch.Tensor, sentence_tokenized: torch.Tensor) -> Optional[Tuple[int, int]]:
        if sentence_tokenized.dim() > 1:
            sentence = sentence_tokenized[0]
        else:
            sentence = sentence_tokenized
        if target_tokenized.dim() > 1:
            target = target_tokenized[0]
        else:
            target = target_tokenized
        target_tok_len = len(target)
        sentence = sentence[1:-1]
        windows = sentence.unfold(0, target_tok_len, 1)
        appearance_match = ((windows == target).all(dim=1)).nonzero(as_tuple=True)[0] if target_tok_len > 1 else (windows == target).nonzero(as_tuple=True)[0]
        if len(appearance_match) > 0:
            start_t = appearance_match[0].item()
            end_t = start_t + target_tok_len
            return start_t, end_t
        return

# ===========| Loss Functions |=========================================#
class Loss_Calculation:
    def __init__(self) -> None:
        super(Loss_Calculation, self).__init__()

    @staticmethod
    def Cross_Entropy(pred_logits: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        """
        pred_logits: Shape [Batch, 3] (The raw numbers before softmax)
        target_indices: Shape [Batch] (The correct IDs: 0, 1, or 2)
        """
        # Reduction='mean' averages the loss across the 16 samples in your batch
        return torch.nn.functional.cross_entropy(pred_logits, target_indices, reduction='mean')

# ==========| Batched Training Methods |=================================#
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class PWQ_SentimentDataset(Dataset):
    def __init__(self, raw_dataset):
        self.samples = []
        for bidx, cur_dict_set in enumerate(raw_dataset):
            sentence = str(next(iter(cur_dict_set)))
            for target, sentiment in cur_dict_set[sentence].items():
                self.samples.append({
                    'sentence': sentence,
                    'target': target,
                    'sentiment': sentiment
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def pwq_collate_fn(batch):
    sentences = [Global.tokenize(s['sentence'])[0][:Settings.MAX_TOK_LENGTH] for s in batch]
    targets = [Global.tokenize(s['target'], add_special_tokens=False)[0] for s in batch]
    sentiments = [Settings.SENTIMENTS.index(s['sentiment']) for s in batch]

    # Pad sequences so they all match the longest one in this batch
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=Global.tokenizer.pad_token_id)

    # We need ranges for each target
    ranges = []
    for t_tok, s_tok in zip(targets, sentences):
        r = PWQ_Target_Awareness._get_target_idx_range(t_tok, s_tok)
        ranges.append(r if r else (0, 1))  # Fallback

    return padded_sentences.to(Global.device), torch.tensor(ranges).to(Global.device), torch.tensor(sentiments).to(
        Global.device)

# ===========| Main PWQ VM AI Class |===================================#
class PWQ_VM_AI_V3S(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, heads_dim: int, layers: int, activation: actv = actv.pmish, ffn_expansion_r: int = 4 ,out_dim: int = 3, max_tok_length:int=1024, temp:float=2.5, mode:str=None) -> None:
        super(PWQ_VM_AI_V3S, self).__init__()
        self.to('cuda')
        """
        Both embeddings are defined here as nn.Parameter() because, the network will eventually learn what values to assign to what word,
        and what shifts in values to apply towards every position. The token embedding has dim 0 set to vocab size, since we want it to 
        learn to map all words within the tokenizer, whereas the positional embedding has dim 0 set to max_tok_length since, we only want
        it to learn map positions in a certain range. Both have dim 1 set to embedding dim, since that's what VM-AI chose for count of 
        vectors assigned within every word. 
        """
        self.tok_embedding = nn.Parameter(torch.randn(vocabulary_size, embedding_dim) * 0.015)
        self.tok_embed_noise = PWQ_NoiseInjection(std=3e-2)
        self.pos_embedding = nn.Parameter(torch.randn(max_tok_length, embedding_dim) * 5e-4)
        self.pos_embed_noise = PWQ_NoiseInjection(std=5e-3)
        self.vm_ai_transformer_block = PWQ_VM_AI_Transformer_Block(embedding_dim, heads_dim, layers, expansion_factor=ffn_expansion_r, activation=activation, mode=mode)
        self.postLayerNorm = PWQ_Layer_Norm(embedding_dim)
        self.linearProjection = PWQ_Linear_Projection(embedding_dim, out_dim, mode=mode)
        self.temp = temp
        self.nrope_frequency = torch.nn.Parameter(torch.tensor([0.0], device=Global.device))
        self.adp_dropout__embed = PWQ_AdpDropout(mode=mode)
        self.mode = mode

    def forward(self, input: torch.Tensor, target_idx_range: Optional[Tuple[int, int]]) -> Optional[Tuple[Dict, torch.tensor, torch.tensor]]:
        batch, tokens = input.shape
        if target_idx_range is None:
            raise ValueError("Forward received None for target_idx_range. Check your DataLoader.")
        """
        Here we index the input for token embedding and token count for positonal one, we use our Parameters, as look-up tables
        to search for correct values for words, and correct shifts for positions, and we add both search results together, thus
        both, initializing each matrix for every word, and shifting it based on its position within a sentence. 
        """
        # Step 1 - Compute positional indices and map them to position embedding matrix
        pos_indices = torch.arange(tokens, device=Global.device)
        positions = self.pos_embedding[pos_indices].unsqueeze(0)
        # Step 2 - Apply NRoPE over token embedding matrix with ARM set to 2/3π
        optimized_frequency = NRoPE_Settings().FREQUENCY_OPTIMIZATION(self.nrope_frequency)
        pwq_nrope = pwq_nropev3(self.tok_embedding[input], neutralize_angle=True,
                                base_frequency=optimized_frequency,
                                angle_neutralization_range=NRoPE_Settings.ARM["EMBEDDING"] * torch.pi)
        # Step 3 - Add NRoPE'd token embedding towards positional embedding to create a matrix which contains both the tokenized content of each word but also its neutralized
        # rotated positional token
        embedded = pwq_nrope.embedded + positions
        # Step 4 - Pass the double-data matrix through adaptive dropout layer
        embedded_dropped_out = self.adp_dropout__embed(embedded)
        # Step 5 - Run the dropout layer output matrix, through the transformer block (attention head + FFN)
        transformer_output = self.vm_ai_transformer_block(embedded_dropped_out)
        # Step 6 - Apply Post-LN School on transformer block output
        postnorm_final = self.postLayerNorm(transformer_output)
        # Step 7 - Figure out which indices of tokenized words are our target
        mask = torch.zeros(batch, tokens, device=Global.device) # Create an empty mask
        if isinstance(target_idx_range, tuple): # If its a tuple simply convert the span of target indices to 1.0
            START_T, END_T = target_idx_range
            mask[0, START_T:END_T] = 1.0
        else:
            for i in range(batch): # Else if in batched training, loop through batches converting indices to integer values, then convert to 1.0
                start = target_idx_range[i, 0].item() if torch.is_tensor(target_idx_range[i, 0]) else target_idx_range[i, 0]
                end = target_idx_range[i, 1].item() if torch.is_tensor(target_idx_range[i, 1]) else target_idx_range[i, 1]
                mask[i, start:end] = 1.0
        # Step 8 - Apply the mask to zero out non-target tokens
        target_vectors = postnorm_final * mask.unsqueeze(-1)
        # Step 9 -
        target_token_counts = mask.sum(dim=1, keepdim=True).clamp(min=1) # Count total target words
        globalAveragePooled = target_vectors.sum(dim=1) / target_token_counts # Compute average value over target words, excluding non-target tokens
        # Step 10 - Apply linear projection layer to convert values into logits, and divide by temp for smoother / scricter values
        linearProjected = self.linearProjection(globalAveragePooled) / self.temp
        # Step 11 - Apply Softmax activation on logits to get probabilities
        softmaxed = actv.softmax(linearProjected, dim=-1)
        # Step 12 - Prepare readable output and return necessary values
        softmax_dict = {
            Settings.SENTIMENTS[j]: round(softmaxed[0][j].item(), 8)
            for j in range(len(Settings.SENTIMENTS))
        }
        return softmax_dict, softmaxed, linearProjected


# ======| Training Loop |===============================================================================================
def Batched_Training(model: PWQ_VM_AI_V3S, epochs: int, weight_fp: str, lr: float = 3e-5, batch_size: int = 16):
    dataset = PWQ_SentimentDataset(Settings.DATASET)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=pwq_collate_fn, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    start_epoch = 0
    if weight_fp and os.path.exists(weight_fp):
        checkpoint = torch.load(weight_fp, map_location=Global.device)
        load_info = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state restored successfully.")
        except ValueError:
            print("⚠️ Optimizer group mismatch! Starting with a fresh optimizer (Weights are still loaded).")
        start_epoch = checkpoint.get('epoch', 0)
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        for b_idx, (src, ranges, labels) in enumerate(dataloader):

            optimizer.zero_grad()
            output_dict, output_softmax, output_logits = model(src, ranges)
            logits_tensor = output_logits
            loss = torch.nn.functional.cross_entropy(logits_tensor, labels)
            loss.backward()

            for name, param in model.named_parameters():
                if 'nrope_frequency' in name and param.grad is not None:
                    param.grad.data.clamp_(-0.01, 0.01)
                if param.grad is not None and torch.isnan(param.grad).any():
                    param.grad = None  # Zero out corrupted gradients

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits_tensor, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if b_idx % 10 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch {b_idx} | Loss: {loss.item():.4f} | Acc: {correct / total_samples * 100:.2f}%")

        save_path = fr"C:\Users\pawli\OneDrive\Dokumenty\PWQ-VM-AI\PWQ-VM-AI Weight Saves\V3 Batched\PWQ_V3S_E{epoch}_A{correct / total_samples * 100:.2f}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

# =======| Inference Run |==============================================================================================
def Inference_Run(model: PWQ_VM_AI_V3S, weight_fp: str, inf_mode: str, input_file_path: str=None) -> None:
    Settings.TEMP = 3.25

    if not os.path.exists(weight_fp):
        raise PWQ_Invalid_Input("Invalid config.weights_filepath input value!")
    checkpoint = torch.load(weight_fp, weights_only=True)
    load_info = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    with torch.no_grad():
        model.nrope_frequency.fill_(0.5)
        for block in model.vm_ai_transformer_block.block_instance_list:
            block['attention'].nrope_frequency.fill_(0.5)

    model.eval()

    if inf_mode.lower() == "manual":
        print("\n" + "=" * 50)
        print("Enter a sentence and target to analyze (or 'quit' to exit)")
        print("=" * 50)
        while True:
            sentence = input("\nSentence: ").strip()
            if sentence.lower() == 'quit':
                break

            target = input("\nTarget Word: ").strip()
            sentence_tok = Global.tokenize(sentence)
            target_tok = Global.tokenize(target, add_special_tokens=False)
            idx_range = PWQ_Target_Awareness._get_target_idx_range(target_tok, sentence_tok[0])
            if idx_range is None:
                print(f"❌ Target '{target}' not found in sentence")
                continue

            print(f"Sentence Tokenized shape: {sentence_tok.shape}")
            print(f"Target range: {idx_range}")
            print(f"Target Tokenized shape: {target_tok.shape}")

            with torch.no_grad():
                output_dict, output_softmax, _ = model(sentence_tok, idx_range)

            print(f"\n📝 Sentence: {sentence}")
            print(f"🎯 Target: '{target}'")
            print("📊 Sentiment:", max(output_dict, key=output_dict.get))
            print("📈 Probabilities:")
            for sentiment, prob in output_dict.items():
                print(f"   {sentiment}: {prob * 100:.2f}%")

            pwq_obdv2(
                positive_count=1 if max(output_dict, key=output_dict.get) == "<POS>" else 0,
                negative_count=1 if max(output_dict, key=output_dict.get) == "<NEG>" else 0,
                neutral_count=1 if max(output_dict, key=output_dict.get) == "<NEU>" else 0,
                title=f"{target.capitalize()} - Audience Response",
                gradient_colors=('#c50145', 'white', '#058b3f'),
                label_colors={
                    'neg': '#c50145',
                    'neu': 'gray',
                    'pos': '#058b3f',
                    'strong_neg': '#48001a',
                    'strong_pos': '#02461f'
                }
            )

    elif inf_mode.lower() == "multiple":
        df = pd.read_csv(input_file_path)
        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="SENTIMENT PREDICTION..."):
            comment = row["COMMENT"]
            target = row["TARGET"]
            clean_comment = pwqvmai_hyp.Text_Cleaner(comment)
            sentence_tok = Global.tokenize(clean_comment)
            max_len = Settings.MAX_TOK_LENGTH
            target_tok = Global.tokenize(target, add_special_tokens=False)
            idx_range = PWQ_Target_Awareness._get_target_idx_range(target_tok, sentence_tok[0])
            if sentence_tok.shape[1] > max_len:
                sentence_tok = sentence_tok[:, :max_len]
                if idx_range and idx_range[1] > max_len:
                    idx_range = (idx_range[0], min(idx_range[1], max_len))

            if idx_range is None:
                results.append({
                    'comment': comment,
                    'target': target,
                    'sentiment': 'ERROR',
                    'pos_prob': 0,
                    'neu_prob': 0,
                    'neg_prob': 0,
                    'confidence':0,
                    'final_decision': None,
                    'note': 'target_not_found'
                })
                continue

            with torch.no_grad():
                output_dict, output_softmax, _ = model(sentence_tok, idx_range)

            results.append({
                'comment': comment,
                'target': target,
                'sentiment': max(output_dict, key=output_dict.get),
                'pos_prob': round(output_dict['<POS>']*100, 4),
                'neu_prob': round(output_dict['<NEU>']*100, 4),
                'neg_prob': round(output_dict['<NEG>']*100, 4),
                'confidence': max(output_dict.values()),
                'final_decision':max(output_dict, key=output_dict.get),
                'note': 'success'
            })
            results_df = pd.DataFrame(results)

        results_df = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print(f"🎯 PWQ-VM-AI V3S INFERENCE REPORT - {len(results_df)} SAMPLES")
        print("=" * 80)

        for idx, row in results_df.iterrows():
            if row['sentiment'] == '<POS>':
                emoji = "✅"
            elif row['sentiment'] == '<NEU>':
                emoji = "⚖️"
            elif row['sentiment'] == '<NEG>':
                emoji = "❌"
            else:
                emoji = "❓"

            print(f"\n[{idx:03d}] {emoji} SENTIMENT: {row['sentiment']} (Conf: {row['confidence'] * 100:.1f}%)")
            print(f"      💬 COMMENT: {row['comment']}..." if len(row['comment']) > 100 else f"      💬 COMMENT: {row['comment']}")
            print(f"      🎯 TARGET:  {row['target']}")
            print(f"      📊 PROBS:   POS: {row['pos_prob']:.4f} | NEU: {row['neu_prob']:.4f} | NEG: {row['neg_prob']:.4f}")
            print("-" * 50)

        pos_count = sum(1 for r in results if r['sentiment'] == '<POS>')
        neu_count = sum(1 for r in results if r['sentiment'] == '<NEU>')
        neg_count = sum(1 for r in results if r['sentiment'] == '<NEG>')
        total = len(results)

        print("\n" + "=" * 60)
        print("📊 SENTIMENT DISTRIBUTION")
        print("=" * 60)
        print(f"POS: {pos_count:3d} ({pos_count / total * 100:5.1f}%) {'█' * int(pos_count / total * 20)}")
        print(f"NEU: {neu_count:3d} ({neu_count / total * 100:5.1f}%) {'█' * int(neu_count / total * 20)}")
        print(f"NEG: {neg_count:3d} ({neg_count / total * 100:5.1f}%) {'█' * int(neg_count / total * 20)}")
        print(f"📊 TOTAL:    {total} comments")

        pwq_obdv2(
        positive_count=pos_count,
        negative_count=neg_count,
        neutral_count=neu_count,
        title=f"{target.capitalize()} - Audience Response",
        gradient_colors=('#c50145', 'white', '#058b3f'),
        label_colors={
        'neg': '#c50145',
        'neu': 'gray',
        'pos': '#058b3f',
        'strong_neg': '#48001a',
        'strong_pos': '#02461f'
        }
    )


def _PWQ_VM_AI_V3S_Exe(config: Config) -> None:
    load_mode = "train"
    pwq_vm_ai_model: PWQ_VM_AI_V3S = PWQ_VM_AI_V3S(
        vocabulary_size=Settings.V_SIZE,
        heads_dim=Settings.HEADS_DIM,
        embedding_dim=Settings.E_DIM,
        layers=Settings.LAYERS,
        activation=Settings.ACTIVATION,
        ffn_expansion_r=Settings.FFN_EXP_R,
        out_dim=Settings.SENTIMENTS.__len__(),
        max_tok_length=Settings.MAX_TOK_LENGTH,
        temp=Settings.TEMP, mode=config.mode
    ).to(Global.device)
    if config.mode == "train":
        Batched_Training(pwq_vm_ai_model, config.epochs, config.weights_filepath, batch_size=64)
    elif config.mode == "inference":
        Inference_Run(pwq_vm_ai_model, config.weights_filepath,"manual" ,input_file_path=config.input_fp)
    else:
        raise PWQ_Invalid_Input("Invalid config.mode input value!")

if __name__ == "__main__":
    config = Config(
        mode="inference",
        weights_filepath=r"C:\Users\pawli\OneDrive\Dokumenty\PWQ-VM-AI\PWQ-VM-AI Weight Saves\V3 Batched\PWQ_V3S_E376_A97.87.pt",
        epochs=380,
        input_fp=r"C:\Users\pawli\OneDrive\Dokumenty\PWQ-VM-AI\Hypothesis Comment Files\AVATAR 3\Avatar 3V3.csv")
    _PWQ_VM_AI_V3S_Exe(config)
