"""
Core Distillation Pipeline for Agent Models

Supports multi-GPU training on the cluster with LoRA fine-tuning
for efficient specialization of base models.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import logging

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset as HFDataset

logger = logging.getLogger(__name__)


class AgentTier(Enum):
    """Agent hierarchy tiers with associated model size targets"""
    TRIUMVIRATE = "triumvirate"  # ~3-7B params - consensus/coordination
    PALADIN = "paladin"          # ~1-3B params - defensive
    CHAOS = "chaos"              # ~1-3B params - adversarial
    SWARM_UNIT = "swarm_unit"    # ~100M-500M params - basic coordination


@dataclass
class DistillationConfig:
    """Configuration for model distillation"""
    
    # Model selection
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    agent_tier: AgentTier = AgentTier.TRIUMVIRATE
    
    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training configuration
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1
    
    # Hardware
    use_bf16: bool = True
    use_4bit_quantization: bool = False  # QLoRA
    
    # Paths
    output_dir: str = "./distilled_models"
    data_path: str = "./training_data"
    
    # Recommended base models per tier
    RECOMMENDED_MODELS: Dict[AgentTier, List[str]] = field(default_factory=lambda: {
        AgentTier.TRIUMVIRATE: [
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Llama-3.1-8B-Instruct",
        ],
        AgentTier.PALADIN: [
            "Qwen/Qwen2.5-3B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
            "google/gemma-2-2b-it",
        ],
        AgentTier.CHAOS: [
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "google/gemma-2-2b-it",
        ],
        AgentTier.SWARM_UNIT: [
            "microsoft/Phi-3-mini-4k-instruct",  # Or custom Mamba/SSM
            "Qwen/Qwen2.5-1.5B-Instruct",
        ],
    })


class AgentTrainingDataset(Dataset):
    """Dataset for agent model training"""
    
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 2048,
        agent_tier: AgentTier = AgentTier.TRIUMVIRATE,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.agent_tier = agent_tier
        self.examples = self._load_data(data_path)
        
    def _load_data(self, data_path: Path) -> List[Dict[str, Any]]:
        """Load training examples from JSONL files"""
        examples = []
        data_file = Path(data_path) / f"{self.agent_tier.value}_training.jsonl"
        
        if not data_file.exists():
            logger.warning(f"No training data found at {data_file}")
            return examples
            
        with open(data_file, "r") as f:
            for line in f:
                examples.append(json.loads(line))
                
        logger.info(f"Loaded {len(examples)} training examples for {self.agent_tier.value}")
        return examples
    
    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format example into training prompt"""
        # Task-specific formatting based on agent tier
        if self.agent_tier == AgentTier.TRIUMVIRATE:
            return self._format_triumvirate(example)
        elif self.agent_tier == AgentTier.PALADIN:
            return self._format_paladin(example)
        elif self.agent_tier == AgentTier.CHAOS:
            return self._format_chaos(example)
        else:
            return self._format_swarm_unit(example)
    
    def _format_triumvirate(self, example: Dict[str, Any]) -> str:
        """Format for consensus/coordination training"""
        return f"""<|system|>
You are a triumvirate coordinator agent in a neuro-inspired swarm system.
Your role is to analyze swarm state and determine optimal coordination decisions.
Consider all agent perspectives and reach consensus through deliberation.
</|system|>

<|user|>
## Swarm State
{example.get('swarm_state', '')}

## Agent Reports
{example.get('agent_reports', '')}

## Task
{example.get('task', 'Determine optimal swarm coordination')}
</|user|>

<|assistant|>
{example.get('response', '')}
</|assistant|>"""

    def _format_paladin(self, example: Dict[str, Any]) -> str:
        """Format for defensive agent training"""
        return f"""<|system|>
You are a Paladin defensive agent protecting critical infrastructure.
Analyze incoming signals for anomalies and potential threats.
Respond with threat assessment and recommended defensive actions.
</|system|>

<|user|>
## Network State
{example.get('network_state', '')}

## Incoming Signals
{example.get('signals', '')}

## Historical Baseline
{example.get('baseline', '')}
</|user|>

<|assistant|>
{example.get('response', '')}
</|assistant|>"""

    def _format_chaos(self, example: Dict[str, Any]) -> str:
        """Format for adversarial agent training"""
        return f"""<|system|>
You are a Chaos agent in red team simulation.
Your role is to probe defenses and identify vulnerabilities through creative attack strategies.
Think like an adversary while operating within ethical boundaries of security testing.
</|system|>

<|user|>
## Target System State
{example.get('system_state', '')}

## Known Defenses
{example.get('defenses', '')}

## Objective
{example.get('objective', 'Identify potential attack vectors')}
</|user|>

<|assistant|>
{example.get('response', '')}
</|assistant|>"""

    def _format_swarm_unit(self, example: Dict[str, Any]) -> str:
        """Format for basic swarm unit training"""
        return f"""<|system|>
You are a swarm unit agent. Process local observations and coordinate with neighbors.
</|system|>

<|user|>
Observation: {example.get('observation', '')}
Neighbor States: {example.get('neighbor_states', '')}
</|user|>

<|assistant|>
{example.get('response', '')}
</|assistant|>"""
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        text = self._format_example(example)
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze(),
        }


class AgentDistiller:
    """
    Main distillation pipeline for training specialized agent models.
    
    Supports:
    - LoRA fine-tuning for efficient training
    - QLoRA (4-bit) for memory-constrained setups
    - Multi-GPU training via DeepSpeed/FSDP
    """
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading base model: {self.config.base_model}")
        
        # Quantization config for QLoRA
        quantization_config = None
        if self.config.use_4bit_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Model setup complete")
        
    def prepare_dataset(self) -> HFDataset:
        """Load and prepare training dataset"""
        dataset = AgentTrainingDataset(
            data_path=Path(self.config.data_path),
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
            agent_tier=self.config.agent_tier,
        )
        
        # Convert to HuggingFace Dataset for SFTTrainer
        def format_for_sft(example):
            return {"text": dataset._format_example(example)}
        
        hf_dataset = HFDataset.from_list(dataset.examples)
        hf_dataset = hf_dataset.map(format_for_sft)
        
        return hf_dataset
    
    def train(self, dataset: Optional[HFDataset] = None) -> str:
        """
        Run the training loop.
        
        Returns:
            Path to the trained model checkpoint
        """
        if self.model is None:
            self.setup()
            
        if dataset is None:
            dataset = self.prepare_dataset()
            
        output_dir = Path(self.config.output_dir) / self.config.agent_tier.value
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            bf16=self.config.use_bf16,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="tensorboard",
            # Multi-GPU settings
            ddp_find_unused_parameters=False,
            dataloader_num_workers=4,
            # Memory optimization
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
        )
        
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save final checkpoint
        final_path = output_dir / "final"
        self.trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        
        logger.info(f"Training complete. Model saved to {final_path}")
        return str(final_path)
    
    def merge_and_save(self, output_path: str) -> str:
        """
        Merge LoRA weights into base model for deployment.
        
        Args:
            output_path: Where to save the merged model
            
        Returns:
            Path to merged model
        """
        if self.model is None:
            raise ValueError("No model loaded. Run setup() first.")
            
        logger.info("Merging LoRA weights...")
        merged_model = self.model.merge_and_unload()
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        merged_model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        
        logger.info(f"Merged model saved to {output_path}")
        return str(output_path)


def train_agent_model(
    agent_tier: AgentTier,
    base_model: Optional[str] = None,
    data_path: str = "./training_data",
    output_dir: str = "./distilled_models",
    **kwargs,
) -> str:
    """
    Convenience function to train an agent model.
    
    Args:
        agent_tier: Which agent tier to train
        base_model: Override default base model selection
        data_path: Path to training data
        output_dir: Where to save trained model
        **kwargs: Additional config overrides
        
    Returns:
        Path to trained model
    """
    config = DistillationConfig(
        agent_tier=agent_tier,
        data_path=data_path,
        output_dir=output_dir,
        **kwargs,
    )
    
    # Use recommended model if not specified
    if base_model is None:
        recommended = config.RECOMMENDED_MODELS[agent_tier]
        base_model = recommended[0]
        logger.info(f"Using recommended model for {agent_tier.value}: {base_model}")
    
    config.base_model = base_model
    
    distiller = AgentDistiller(config)
    model_path = distiller.train()
    
    # Optionally merge for deployment
    merged_path = Path(output_dir) / f"{agent_tier.value}_merged"
    distiller.merge_and_save(str(merged_path))
    
    return str(merged_path)
