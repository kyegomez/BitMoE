import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from bitnet import BitFeedForward, BitLinear
from dotenv import load_dotenv
from loguru import logger
from torch import Tensor, nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Load the environment variables
load_dotenv()

# Set the seed for reproducibility
torch.manual_seed(0)


# Device configuration
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()

    # Print the number of GPUs
    logger.info(f"Number of GPUs available: {num_gpus}")



# Network setup
def network_setup(rank, world_size, backend="nccl"):
    """
    Initialize the network.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        backend (str, optional): The backend to be used for the distributed communication. Defaults to "nccl".

    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    # Initialize the process group
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    logger.info(f"Initialized the process group: {rank}")

    # Set the device
    torch.cuda.set_device(rank)
    
    
def cleanup():
    logger.info("Cleaned up the process group")
    dist.destroy_process_group()
    


class Gate(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
    ):
        """
        GatingMechanism is a class that represents the gating mechanism in a mixture of experts model.

        Args:
            dim (int): The input dimension.
            num_experts (int): The number of experts in the mixture.

        """
        super().__init__()
        self.gate = BitLinear(dim, num_experts)

    def forward(self, x: Tensor):
        """
        Forward pass of the gating mechanism.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the gating mechanism.

        """
        return F.softmax(self.gate(x), dim=-1)


class BitMoE(nn.Module):
    """
    Simple Mixture of Experts (MoE) model.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward network.
        output_dim (int): Output dimension.
        num_experts (int): Number of experts in the MoE.
        mult (int, optional): Multiplier for the hidden dimension. Defaults to 4.
    """

    def __init__(
        self,
        dim,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        mult: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.mult = mult
        
        if torch.cuda.is_available():
            network_setup(rank=0, world_size=num_gpus)
            
            self.experts = nn.ModuleList(
                [
                    FSDP(BitFeedForward(dim, mult))
                    for _ in range(num_experts)
                ]
            )

            self.gate = FSDP(Gate(dim, num_experts))
            
            logger.info("Using FSDP")
            
            
        else: 
            self.experts = nn.ModuleList(
                [
                    BitFeedForward(dim, mult)
                    for _ in range(num_experts)
                ]
            )

            self.gate = Gate(dim, num_experts)
            
            # Log
            logger.info("Using BitFeedForward")

    def forward(self, x: Tensor):
        """
        Forward pass of the SimpleMoE model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        gating_scores = self.gate(x)

        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )

        output = torch.sum(
            gating_scores.unsqueeze(2) * expert_outputs, dim=-1
        )

        return output
