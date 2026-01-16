# Model Distillation Roadmap for Neuro-Swarm Agents

> **Date:** January 2025  
> **Status:** Planning  
> **Goal:** Create locally-deployed, specialized models for each agent type - eliminating remote API dependency for critical infrastructure defense.

---

## Executive Summary

Remote model APIs are unsuitable for swarm coordination:
- **Latency kills coordination** - milliseconds matter for emergent behavior
- **Critical infrastructure can't depend on internet connectivity**
- **Cost scales poorly** with thousands of agents

**Solution:** Distill specialized, lightweight models trained on your simulation data, deployed locally.

---

## Agent Capability Hierarchy

```
┌─────────────────────────────────────────────┐
│  TRIUMVIRATE (Consensus/Coordination)       │  ← Most capable (~3-7B params)
│  - Multi-agent reasoning                    │
│  - Strategic decision-making                │
│  - State space navigation                   │
│  - Manifold navigation decisions            │
├─────────────────────────────────────────────┤
│  PALADIN (Defensive)                        │  ← Specialized (~1-3B params)
│  - Anomaly detection                        │
│  - Pattern recognition                      │
│  - Threat classification                    │
│  - Infrastructure protection                │
├─────────────────────────────────────────────┤
│  CHAOS (Adversarial)                        │  ← Specialized (~1-3B params)
│  - Attack generation                        │
│  - Exploitation creativity                  │
│  - Evasion tactics                          │
│  - Red team scenarios                       │
├─────────────────────────────────────────────┤
│  SWARM UNITS (Basic agents)                 │  ← Minimal (~100M-500M params)
│  - Stimulus-response                        │    OR pure SSM (non-LLM)
│  - Local coordination                       │
│  - Stigmergic communication                 │
│  - Energy-based state transitions           │
└─────────────────────────────────────────────┘
```

---

## Hardware Context

**Training Cluster:**
- 256GB RAM
- 32-core Threadripper
- 2× Titan GPUs (~48GB VRAM total)
- k3s orchestration
- Google Cloud credits available for overflow

**Deployment Target:**
- Edge devices / embedded systems
- Must run without internet connectivity
- Sub-100ms inference latency required

---

## Base Model Selection

| Agent Type | Recommended Base | Params | Why | Quantized Size |
|------------|------------------|--------|-----|----------------|
| Triumvirate | Qwen2.5-7B | 7B | Best reasoning at trainable size | ~4GB (INT4) |
| Triumvirate (alt) | Mistral-7B-v0.3 | 7B | Strong instruction following | ~4GB (INT4) |
| Paladin | Qwen2.5-3B | 3B | Good pattern recognition | ~2GB (INT4) |
| Paladin (alt) | Phi-3-mini | 3.8B | Efficient, strong at classification | ~2GB (INT4) |
| Chaos | Llama-3.2-3B | 3B | Creative generation | ~2GB (INT4) |
| Swarm Units | Custom Mamba SSM | 100-500M | Sub-LLM, pure state-space | ~200MB |
| Swarm Units (alt) | Phi-3-mini (INT4) | 3.8B | If language needed | ~2GB |

### Key Architectural Decision

**Swarm units may not need LLMs at all.**

For basic agents doing stigmergic communication and local coordination, a Mamba SSM (actual state-space model, not language model) is likely more appropriate:
- Aligns with theoretical framework (state space navigation)
- Sub-millisecond inference
- Scales to thousands of agents
- Lower memory footprint

**Recommendation:** Triumvirate uses distilled LLM for reasoning. Swarm units use pure SSM for state-to-action mapping.

---

## Phase 1: Data Generation Pipeline

### Task Distributions by Agent Type

```python
"""
Training data generation from teacher model + simulation
Run this against your existing neuro-swarm environment
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class TrainingExample:
    input_state: Dict[str, Any]
    output_action: Dict[str, Any]
    reasoning: str  # Chain-of-thought from teacher
    task_type: str
    agent_type: str


class AgentTaskGenerator:
    """Generate training examples from teacher model + simulation"""
    
    def __init__(self, teacher_model, simulation_env):
        self.teacher = teacher_model  # Claude API or large local model
        self.env = simulation_env      # Your NeuroSwarm environment
    
    def generate_triumvirate_examples(self, n_episodes: int) -> List[TrainingExample]:
        """
        Triumvirate tasks:
        - Given swarm state → optimal coordination decision
        - Given conflicting signals → consensus resolution
        - Given threat landscape → strategic allocation
        """
        examples = []
        
        for episode in self.env.run_episodes(n_episodes):
            # Capture swarm-level decision points
            for decision_point in episode.get_coordination_decisions():
                state = decision_point.swarm_state
                
                prompt = f"""You are coordinating a defensive swarm. 
                
Current swarm state:
- Active agents: {state.agent_count}
- Threat level: {state.threat_assessment}
- Resource allocation: {state.resources}
- Agent positions: {state.positions}

What coordination action should the triumvirate take?
Think step by step, then provide the action."""

                teacher_response = self.teacher.complete(prompt)
                
                examples.append(TrainingExample(
                    input_state=state.serialize(),
                    output_action=self._parse_action(teacher_response),
                    reasoning=teacher_response,
                    task_type='coordination',
                    agent_type='triumvirate'
                ))
        
        return examples
    
    def generate_paladin_examples(self, n_scenarios: int) -> List[TrainingExample]:
        """
        Paladin tasks:
        - Given sensor data → threat classification
        - Given network state → anomaly detection
        - Given attack pattern → defensive response
        """
        examples = []
        
        for scenario in self.env.generate_defense_scenarios(n_scenarios):
            observation = scenario.sensor_data
            
            prompt = f"""You are a defensive agent protecting critical infrastructure.

Sensor observation:
{json.dumps(observation, indent=2)}

Classify the threat level and recommend defensive action.
Categories: [benign, suspicious, active_threat, critical]"""

            teacher_response = self.teacher.complete(prompt)
            
            examples.append(TrainingExample(
                input_state=observation,
                output_action=self._parse_classification(teacher_response),
                reasoning=teacher_response,
                task_type='threat_classification',
                agent_type='paladin'
            ))
        
        return examples
    
    def generate_chaos_examples(self, n_scenarios: int) -> List[TrainingExample]:
        """
        Chaos tasks:
        - Given target system → attack vector generation
        - Given defenses → evasion strategy
        - Given constraints → creative exploitation
        """
        examples = []
        
        for scenario in self.env.generate_attack_scenarios(n_scenarios):
            target = scenario.target_description
            constraints = scenario.constraints
            
            prompt = f"""You are a red team agent testing defenses.

Target system:
{json.dumps(target, indent=2)}

Constraints (ethical boundaries):
{json.dumps(constraints, indent=2)}

Generate a realistic attack vector that would test the defenses.
Focus on creativity and realism for training defensive systems."""

            teacher_response = self.teacher.complete(prompt)
            
            examples.append(TrainingExample(
                input_state={'target': target, 'constraints': constraints},
                output_action=self._parse_attack_vector(teacher_response),
                reasoning=teacher_response,
                task_type='attack_generation',
                agent_type='chaos'
            ))
        
        return examples
    
    def _parse_action(self, response: str) -> Dict:
        # Extract structured action from teacher response
        # Implementation depends on your action space
        pass
    
    def _parse_classification(self, response: str) -> Dict:
        pass
    
    def _parse_attack_vector(self, response: str) -> Dict:
        pass


def save_dataset(examples: List[TrainingExample], path: str):
    """Save in format suitable for SFTTrainer"""
    formatted = []
    for ex in examples:
        formatted.append({
            'text': f"""### Input
{json.dumps(ex.input_state)}

### Reasoning
{ex.reasoning}

### Action
{json.dumps(ex.output_action)}""",
            'task_type': ex.task_type,
            'agent_type': ex.agent_type
        })
    
    with open(path, 'w') as f:
        json.dump(formatted, f, indent=2)
```

### Data Volume Targets

| Agent Type | Examples Needed | Reasoning |
|------------|-----------------|-----------|
| Triumvirate | 50,000+ | Complex reasoning, needs diversity |
| Paladin | 20,000+ | Classification task, less diversity needed |
| Chaos | 30,000+ | Creative task, needs variety |
| Swarm Units | 100,000+ | Simple behaviors, high volume low complexity |

---

## Phase 2: Training Infrastructure

### LoRA Fine-Tuning Setup

```python
"""
Training pipeline for agent model distillation
Optimized for 2x Titan GPU setup
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset


class AgentDistiller:
    """Distillation pipeline for agent models"""
    
    def __init__(
        self, 
        base_model_name: str, 
        agent_type: str,
        use_4bit: bool = True  # QLoRA for memory efficiency
    ):
        self.agent_type = agent_type
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config for QLoRA
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Spreads across 2 Titans
            attn_implementation="flash_attention_2",  # If supported
        )
        
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        # Higher rank = more capability, more memory
        lora_rank = self._get_lora_rank_for_agent()
        
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj"       # MLP
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, self.lora_config)
        self._print_trainable_params()
    
    def _get_lora_rank_for_agent(self) -> int:
        """Higher rank for more complex agents"""
        ranks = {
            'triumvirate': 128,  # Maximum capability
            'paladin': 64,       # Good classification
            'chaos': 64,         # Creative generation
            'swarm': 32,         # Simple behaviors
        }
        return ranks.get(self.agent_type, 64)
    
    def _print_trainable_params(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def train(
        self, 
        dataset_path: str, 
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation: int = 4
    ):
        """Run training with optimal settings for dual Titan setup"""
        
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            # Effective batch size = 4 * 4 * 2 GPUs = 32
            
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            
            bf16=True,
            
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=3,
            
            # Optimization
            optim="paged_adamw_32bit",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            
            # Multi-GPU
            ddp_find_unused_parameters=False,
            
            # Reporting
            report_to="tensorboard",
        )
        
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            max_seq_length=2048,
            dataset_text_field="text",
            packing=True,  # Efficient sequence packing
        )
        
        trainer.train()
        
        # Save LoRA weights
        trainer.save_model(output_dir)
        
        return output_dir
    
    def merge_and_export(self, lora_path: str, output_path: str):
        """Merge LoRA weights back into base model"""
        from peft import PeftModel
        
        # Load base model in full precision for merging
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model.config._name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Load and merge LoRA
        model = PeftModel.from_pretrained(base_model, lora_path)
        merged_model = model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        return output_path


# Usage
if __name__ == "__main__":
    # Train Triumvirate
    distiller = AgentDistiller(
        base_model_name="Qwen/Qwen2.5-7B-Instruct",
        agent_type="triumvirate",
        use_4bit=True
    )
    
    distiller.train(
        dataset_path="data/triumvirate_training.json",
        output_dir="models/triumvirate_lora",
        num_epochs=3
    )
    
    distiller.merge_and_export(
        lora_path="models/triumvirate_lora",
        output_path="models/triumvirate_merged"
    )
```

---

## Phase 3: Quantization for Deployment

### INT4 Quantization Pipeline

```python
"""
Quantize trained models for edge deployment
Target: ~4x memory reduction with minimal quality loss
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch


def quantize_for_deployment(
    model_path: str, 
    output_path: str,
    calibration_data_path: str,
    bits: int = 4
):
    """
    Quantize trained model for edge deployment
    
    Args:
        model_path: Path to merged model
        output_path: Where to save quantized model
        calibration_data_path: Subset of training data for calibration
        bits: Quantization bits (4 recommended)
    
    Returns:
        Path to quantized model
    """
    
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,  # Balance between quality and speed
        desc_act=True,   # Better quality, slightly slower
        damp_percent=0.1,
    )
    
    # Load model for quantization
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Prepare calibration data
    calibration_dataset = prepare_calibration_data(
        calibration_data_path, 
        tokenizer,
        n_samples=256  # Typically 128-512 samples sufficient
    )
    
    # Quantize
    model.quantize(calibration_dataset)
    
    # Save
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Quantized model saved to {output_path}")
    print(f"Size reduction: ~{get_size_reduction(model_path, output_path):.1f}x")
    
    return output_path


def prepare_calibration_data(data_path: str, tokenizer, n_samples: int = 256):
    """Prepare calibration dataset for quantization"""
    import json
    
    with open(data_path) as f:
        data = json.load(f)
    
    # Sample representative examples
    samples = data[:n_samples]
    
    calibration_data = []
    for sample in samples:
        tokens = tokenizer(
            sample['text'],
            return_tensors='pt',
            truncation=True,
            max_length=2048
        )
        calibration_data.append(tokens['input_ids'])
    
    return calibration_data


def get_size_reduction(original_path: str, quantized_path: str) -> float:
    """Calculate size reduction ratio"""
    import os
    
    def get_dir_size(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total
    
    original_size = get_dir_size(original_path)
    quantized_size = get_dir_size(quantized_path)
    
    return original_size / quantized_size


# Alternative: Use llama.cpp for GGUF format (better for CPU inference)
def convert_to_gguf(model_path: str, output_path: str, quantization: str = "q4_k_m"):
    """
    Convert to GGUF format for llama.cpp inference
    Better for CPU-only deployment
    """
    import subprocess
    
    # Requires llama.cpp installed
    cmd = [
        "python", "llama.cpp/convert.py",
        model_path,
        "--outfile", f"{output_path}/model.gguf",
        "--outtype", quantization
    ]
    
    subprocess.run(cmd, check=True)
    return f"{output_path}/model.gguf"
```

### Quantization Options Comparison

| Method | Size (7B) | Speed | Quality | Best For |
|--------|-----------|-------|---------|----------|
| FP16 | ~14GB | Baseline | 100% | Training only |
| INT8 | ~7GB | 1.5x | ~99% | GPU inference |
| INT4 (GPTQ) | ~4GB | 2x | ~97% | GPU edge deployment |
| Q4_K_M (GGUF) | ~4GB | 1.5x | ~97% | CPU inference |
| Q2_K (GGUF) | ~2.5GB | 2x | ~90% | Extreme compression |

---

## Phase 4: Integration with NeuroAgent

### Model Loading in Agent Architecture

```python
"""
Integration of distilled models into NeuroAgent architecture
"""

from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class AgentBrain:
    """
    Neural decision-making component for agents
    Wraps distilled model for inference
    """
    
    def __init__(
        self,
        model_path: str,
        agent_type: str,
        device: str = "cuda",
        max_new_tokens: int = 256
    ):
        self.agent_type = agent_type
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        # Load quantized model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Warm up
        self._warmup()
    
    def _warmup(self):
        """Warm up model for consistent latency"""
        dummy_input = self.tokenizer("warmup", return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model.generate(**dummy_input, max_new_tokens=1)
    
    @torch.inference_mode()
    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision given current state
        
        Args:
            state: Current agent/swarm state
            
        Returns:
            Action dictionary
        """
        prompt = self._format_prompt(state)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=1792  # Leave room for generation
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Deterministic for consistency
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return self._parse_action(response)
    
    def _format_prompt(self, state: Dict) -> str:
        """Format state into prompt for model"""
        # Agent-type specific formatting
        templates = {
            'triumvirate': self._triumvirate_template,
            'paladin': self._paladin_template,
            'chaos': self._chaos_template,
        }
        return templates[self.agent_type](state)
    
    def _triumvirate_template(self, state: Dict) -> str:
        return f"""### Swarm State
{state}

### Coordination Decision
"""
    
    def _paladin_template(self, state: Dict) -> str:
        return f"""### Observation
{state}

### Threat Assessment
"""
    
    def _chaos_template(self, state: Dict) -> str:
        return f"""### Target
{state}

### Attack Vector
"""
    
    def _parse_action(self, response: str) -> Dict[str, Any]:
        """Parse model output into action dict"""
        # Implementation depends on your action space
        # Could use structured output parsing
        return {'raw_response': response}


class NeuroAgentWithBrain:
    """
    Extended NeuroAgent with distilled model brain
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        model_path: Optional[str] = None,
        # ... other NeuroAgent params
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        
        # Initialize brain if model provided
        self.brain = None
        if model_path:
            self.brain = AgentBrain(model_path, agent_type)
        
        # ... rest of NeuroAgent initialization
    
    def think(self, state: Dict) -> Dict:
        """High-level decision making using brain"""
        if self.brain is None:
            raise RuntimeError("Agent has no brain loaded")
        
        return self.brain.decide(state)
```

---

## Phase 5: Swarm Unit Architecture (SSM-Based)

### Pure State-Space Model for Swarm Units

```python
"""
Lightweight SSM-based swarm units
No language model - pure state-to-action mapping
Aligns with theoretical framework of state space navigation
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class SwarmUnitBrain(nn.Module):
    """
    Minimal neural architecture for swarm units
    Uses Mamba SSM for efficient sequence modeling
    
    Design principles:
    - Simplicity generates complexity
    - Balance over power
    - The medium is the message (stigmergic communication)
    """
    
    def __init__(
        self,
        state_dim: int = 64,      # Internal state dimension
        obs_dim: int = 32,        # Observation dimension
        action_dim: int = 8,      # Action dimension
        d_model: int = 128,       # Model dimension
        n_layers: int = 4,        # Number of Mamba layers
        neighbor_limit: int = 7,  # Starling-inspired neighbor limit
    ):
        super().__init__()
        
        self.neighbor_limit = neighbor_limit
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Neighbor state encoder (stigmergic input)
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Mamba SSM layers for temporal dynamics
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            for _ in range(n_layers)
        ])
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, action_dim),
        )
        
        # Internal state (persistent across steps)
        self.register_buffer('internal_state', torch.zeros(1, d_model))
    
    def forward(
        self, 
        observation: torch.Tensor,
        neighbor_states: torch.Tensor,  # [batch, max_neighbors, state_dim]
        neighbor_mask: torch.Tensor,    # [batch, max_neighbors] - valid neighbors
    ) -> torch.Tensor:
        """
        Process observation and neighbor states to produce action
        
        Biological inspiration:
        - Limited neighbor attention (7 neighbors like starlings)
        - Stigmergic communication through shared state
        - Energy-based dynamics (implicit in SSM gating)
        """
        batch_size = observation.shape[0]
        
        # Encode observation
        obs_encoded = self.obs_encoder(observation)  # [batch, d_model]
        
        # Encode and aggregate neighbor states
        neighbor_encoded = self.neighbor_encoder(neighbor_states)  # [batch, n, d_model]
        
        # Masked mean pooling over valid neighbors
        neighbor_mask = neighbor_mask.unsqueeze(-1)  # [batch, n, 1]
        neighbor_sum = (neighbor_encoded * neighbor_mask).sum(dim=1)
        neighbor_count = neighbor_mask.sum(dim=1).clamp(min=1)
        neighbor_agg = neighbor_sum / neighbor_count  # [batch, d_model]
        
        # Combine observation and neighbor signal
        x = obs_encoded + neighbor_agg  # [batch, d_model]
        x = x.unsqueeze(1)  # [batch, 1, d_model] for Mamba
        
        # Process through Mamba layers
        for mamba in self.mamba_layers:
            x = mamba(x) + x  # Residual connection
        
        x = x.squeeze(1)  # [batch, d_model]
        
        # Produce action
        action = self.action_head(x)
        
        return action
    
    def get_state_for_neighbors(self) -> torch.Tensor:
        """
        Export internal state for neighbor communication
        This is the stigmergic signal
        """
        return self.internal_state.clone()


class SwarmUnitTrainer:
    """
    Training pipeline for swarm unit SSM
    Uses imitation learning from simulation
    """
    
    def __init__(self, model: SwarmUnitBrain, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()  # For continuous actions
    
    def train_step(
        self,
        observations: torch.Tensor,
        neighbor_states: torch.Tensor,
        neighbor_masks: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        pred_actions = self.model(observations, neighbor_states, neighbor_masks)
        loss = self.loss_fn(pred_actions, target_actions)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## Execution Timeline

### Week 1-2: Data Generation
- [ ] Set up teacher model access (Claude API or local 70B)
- [ ] Instrument simulation for decision point capture
- [ ] Generate 50K triumvirate examples
- [ ] Generate 20K paladin examples
- [ ] Generate 30K chaos examples
- [ ] Generate 100K swarm unit trajectories

### Week 3-4: Triumvirate Training
- [ ] Set up training infrastructure
- [ ] Train Qwen2.5-7B with LoRA
- [ ] Validate on held-out simulation episodes
- [ ] Iterate on data/hyperparameters

### Week 5-6: Specialized Agent Training
- [ ] Train Paladin model (Qwen2.5-3B)
- [ ] Train Chaos model (Llama-3.2-3B)
- [ ] Train Swarm Unit SSM
- [ ] Cross-validation across scenarios

### Week 7-8: Quantization & Integration
- [ ] INT4 quantize all LLM models
- [ ] Export swarm unit to ONNX
- [ ] Integrate into NeuroAgent architecture
- [ ] Benchmark latency and quality
- [ ] Full swarm simulation validation

### Week 9+: Iteration & Scale
- [ ] Identify failure modes
- [ ] Collect hard examples for retraining
- [ ] Scale to 1000+ agent simulations
- [ ] Prepare publication results

---

## Key Metrics to Track

| Metric | Target | Measurement |
|--------|--------|-------------|
| Triumvirate decision latency | <100ms | Time from state to action |
| Swarm unit latency | <10ms | Individual agent cycle time |
| Coordination accuracy | >90% | Match teacher decisions |
| Threat detection F1 | >0.95 | Paladin classification |
| Memory per agent | <2GB | Quantized model size |
| Swarm emergence quality | Qualitative | Murmuration patterns |

---

## Open Questions

1. **Swarm unit architecture**: Pure SSM vs tiny LLM vs hybrid?
2. **Communication protocol**: How do agents share state for neighbor encoding?
3. **Training curriculum**: Should we train on progressively harder scenarios?
4. **Model versioning**: How to update deployed models safely?
5. **Failure modes**: What happens when model confidence is low?

---

## References

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)
