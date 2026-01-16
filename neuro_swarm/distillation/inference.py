"""
Agent Inference Engine

Provides fast inference for distilled agent models, integrating with
the NeuroAgent architecture for real-time swarm coordination.

Key features:
- Batch inference for multiple agents
- Caching for repeated patterns
- Integration with existing NeuroAgent state/action interfaces
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from abc import ABC, abstractmethod
import logging
import time
from functools import lru_cache

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine"""
    
    # Model settings
    model_path: str = ""
    device: str = "auto"
    torch_dtype: str = "bfloat16"
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Batching
    max_batch_size: int = 8
    
    # Caching
    use_cache: bool = True
    cache_size: int = 1000
    
    # Performance
    use_flash_attention: bool = True
    compile_model: bool = False  # torch.compile (experimental)


class AgentBrain(ABC):
    """
    Abstract base class for agent decision-making modules.
    
    Subclasses implement specific agent tier behaviors.
    """
    
    @abstractmethod
    def decide(
        self,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a decision based on observation and context.
        
        Args:
            observation: Current observation from environment
            context: Additional context (neighbor states, etc.)
            
        Returns:
            Decision dictionary with action and metadata
        """
        pass
    
    @abstractmethod
    def batch_decide(
        self,
        observations: List[Dict[str, Any]],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Batch decision-making for multiple agents"""
        pass


class DistilledAgentBrain(AgentBrain):
    """
    Brain module using a distilled LLM for decision-making.
    
    Used for Triumvirate, Paladin, and Chaos agents.
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        agent_tier: str = "triumvirate",
    ):
        self.config = config
        self.agent_tier = agent_tier
        self.model = None
        self.tokenizer = None
        self._setup()
    
    def _setup(self):
        """Load model and tokenizer"""
        logger.info(f"Loading {self.agent_tier} brain from {self.config.model_path}")
        
        # Determine dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
        
        # Model kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": self.config.device,
            "trust_remote_code": True,
        }
        
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **model_kwargs,
        )
        
        # Optional: compile for faster inference (PyTorch 2.0+)
        if self.config.compile_model:
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Brain ready on {self.model.device}")
    
    def _format_prompt(
        self,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format observation into model prompt"""
        if self.agent_tier == "triumvirate":
            return self._format_triumvirate_prompt(observation, context)
        elif self.agent_tier == "paladin":
            return self._format_paladin_prompt(observation, context)
        elif self.agent_tier == "chaos":
            return self._format_chaos_prompt(observation, context)
        else:
            return self._format_swarm_unit_prompt(observation, context)
    
    def _format_triumvirate_prompt(
        self,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        return f"""<|system|>
You are a triumvirate coordinator. Analyze and coordinate.
</|system|>

<|user|>
## State
{self._serialize_observation(observation)}

## Context
{self._serialize_context(context)}

Decide: consensus action for the swarm.
</|user|>

<|assistant|>
"""
    
    def _format_paladin_prompt(
        self,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        return f"""<|system|>
You are a Paladin. Detect threats and defend.
</|system|>

<|user|>
## Observation
{self._serialize_observation(observation)}

Assess: threat level, anomalies, action.
</|user|>

<|assistant|>
"""
    
    def _format_chaos_prompt(
        self,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        return f"""<|system|>
You are a Chaos agent. Identify vulnerabilities.
</|system|>

<|user|>
## Target
{self._serialize_observation(observation)}

Plan: attack vector and evasion.
</|user|>

<|assistant|>
"""
    
    def _format_swarm_unit_prompt(
        self,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        return f"""<|user|>
State: {self._serialize_observation(observation)}
Neighbors: {self._serialize_context(context)}
</|user|>

<|assistant|>
ACTION:"""
    
    def _serialize_observation(self, obs: Dict[str, Any]) -> str:
        """Convert observation dict to string"""
        if obs is None:
            return "None"
        
        lines = []
        for key, value in obs.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 10:
                    value = f"[{len(value)} items]"
                else:
                    value = [round(v, 3) if isinstance(v, float) else v for v in value]
            elif isinstance(value, float):
                value = round(value, 3)
            lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def _serialize_context(self, ctx: Optional[Dict[str, Any]]) -> str:
        """Convert context dict to string"""
        if ctx is None:
            return "None"
        return self._serialize_observation(ctx)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response into structured decision"""
        decision = {
            "raw_response": response,
            "action": None,
            "direction": None,
            "intensity": None,
            "confidence": 0.5,
        }
        
        # Parse based on agent tier
        response_lower = response.lower()
        
        # Try to extract action
        if "action:" in response_lower:
            action_line = response_lower.split("action:")[1].split("\n")[0].strip()
            if "move" in action_line:
                decision["action"] = "move"
            elif "rest" in action_line:
                decision["action"] = "rest"
            elif "deposit" in action_line:
                decision["action"] = "deposit"
            elif "follow" in action_line:
                decision["action"] = "follow"
            elif "attack" in action_line:
                decision["action"] = "attack"
            elif "defend" in action_line:
                decision["action"] = "defend"
            elif "coordinate" in action_line:
                decision["action"] = "coordinate"
        
        # Try to extract direction
        if "direction:" in response_lower:
            try:
                dir_str = response_lower.split("direction:")[1].split("\n")[0]
                # Parse [x, y] format
                dir_str = dir_str.replace("[", "").replace("]", "").strip()
                parts = dir_str.split(",")
                if len(parts) >= 2:
                    decision["direction"] = [float(parts[0]), float(parts[1])]
            except:
                pass
        
        # Try to extract intensity
        if "intensity:" in response_lower:
            try:
                int_str = response_lower.split("intensity:")[1].split("\n")[0].strip()
                decision["intensity"] = float(int_str)
            except:
                pass
        
        # Try to extract confidence
        if "confidence:" in response_lower:
            try:
                conf_str = response_lower.split("confidence:")[1].split("\n")[0].strip()
                decision["confidence"] = float(conf_str)
            except:
                pass
        
        # Threat level for paladins
        if "threat_level:" in response_lower or "threat level:" in response_lower:
            for level in ["critical", "high", "medium", "low", "none"]:
                if level in response_lower:
                    decision["threat_level"] = level
                    break
        
        return decision
    
    @lru_cache(maxsize=1000)
    def _cached_inference(self, prompt_hash: str, prompt: str) -> str:
        """Cached inference for repeated prompts"""
        return self._generate(prompt)
    
    def _generate(self, prompt: str) -> str:
        """Run inference"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - self.config.max_new_tokens,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response
    
    def decide(
        self,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a single decision"""
        prompt = self._format_prompt(observation, context)
        
        if self.config.use_cache:
            import hashlib
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            response = self._cached_inference(prompt_hash, prompt)
        else:
            response = self._generate(prompt)
        
        return self._parse_response(response)
    
    def batch_decide(
        self,
        observations: List[Dict[str, Any]],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batch decision-making for multiple agents.
        
        More efficient than individual calls due to batched inference.
        """
        if contexts is None:
            contexts = [None] * len(observations)
        
        # Format all prompts
        prompts = [
            self._format_prompt(obs, ctx)
            for obs, ctx in zip(observations, contexts)
        ]
        
        # Batch inference
        decisions = []
        for i in range(0, len(prompts), self.config.max_batch_size):
            batch_prompts = prompts[i:i + self.config.max_batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048 - self.config.max_new_tokens,
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode each output
            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(
                    output[inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                decisions.append(self._parse_response(response))
        
        return decisions


class MambaAgentBrain(AgentBrain):
    """
    Brain module using a Mamba SSM for fast local coordination.
    
    This is for swarm units that need sub-millisecond inference.
    Uses actual state-space model architecture (not an LLM).
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        state_dim: int = 32,
        hidden_dim: int = 64,
        action_dim: int = 4,  # [move, rest, deposit, follow]
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.model = self._build_model()
        
        if model_path:
            self._load_weights(model_path)
    
    def _build_model(self) -> torch.nn.Module:
        """Build a minimal state-space model for fast inference"""
        try:
            from mamba_ssm import Mamba
            
            class MambaDecisionModel(torch.nn.Module):
                def __init__(self, state_dim, hidden_dim, action_dim):
                    super().__init__()
                    self.encoder = torch.nn.Linear(state_dim, hidden_dim)
                    self.mamba = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
                    self.action_head = torch.nn.Linear(hidden_dim, action_dim)
                    self.direction_head = torch.nn.Linear(hidden_dim, 2)
                    self.intensity_head = torch.nn.Linear(hidden_dim, 1)
                
                def forward(self, x):
                    # x: [batch, seq_len, state_dim]
                    h = self.encoder(x)
                    h = self.mamba(h)
                    h = h[:, -1, :]  # Take last timestep
                    
                    action_logits = self.action_head(h)
                    direction = torch.tanh(self.direction_head(h))
                    intensity = torch.sigmoid(self.intensity_head(h))
                    
                    return action_logits, direction, intensity
            
            return MambaDecisionModel(self.state_dim, self.hidden_dim, self.action_dim)
            
        except ImportError:
            logger.warning("mamba_ssm not installed, using simple MLP fallback")
            return self._build_mlp_fallback()
    
    def _build_mlp_fallback(self) -> torch.nn.Module:
        """Simple MLP when Mamba isn't available"""
        class MLPDecisionModel(torch.nn.Module):
            def __init__(self, state_dim, hidden_dim, action_dim):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                )
                self.action_head = torch.nn.Linear(hidden_dim, action_dim)
                self.direction_head = torch.nn.Linear(hidden_dim, 2)
                self.intensity_head = torch.nn.Linear(hidden_dim, 1)
            
            def forward(self, x):
                if len(x.shape) == 3:
                    x = x[:, -1, :]  # Take last timestep
                h = self.net(x)
                
                action_logits = self.action_head(h)
                direction = torch.tanh(self.direction_head(h))
                intensity = torch.sigmoid(self.intensity_head(h))
                
                return action_logits, direction, intensity
        
        return MLPDecisionModel(self.state_dim, self.hidden_dim, self.action_dim)
    
    def _load_weights(self, path: str):
        """Load trained weights"""
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded Mamba brain weights from {path}")
    
    def _observation_to_tensor(
        self,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Convert observation to tensor"""
        features = []
        
        # Agent's own state
        features.extend(observation.get("position", [0, 0]))
        features.extend(observation.get("velocity", [0, 0]))
        features.append(observation.get("energy", 0.5))
        features.append(observation.get("phase", 0))
        
        # Neighbor features (pad/truncate to fixed size)
        if context and "neighbors" in context:
            neighbors = context["neighbors"][:7]  # Max 7 neighbors (starling rule)
            for n in neighbors:
                features.extend(n.get("relative_position", [0, 0]))
                features.append(n.get("energy", 0.5))
            # Pad if fewer than 7 neighbors
            for _ in range(7 - len(neighbors)):
                features.extend([0, 0, 0])
        else:
            features.extend([0] * 21)  # 7 neighbors * 3 features
        
        # Pad to state_dim
        while len(features) < self.state_dim:
            features.append(0)
        
        return torch.tensor(features[:self.state_dim], dtype=torch.float32)
    
    def decide(
        self,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a fast decision"""
        x = self._observation_to_tensor(observation, context)
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        
        self.model.eval()
        with torch.no_grad():
            action_logits, direction, intensity = self.model(x)
        
        action_idx = torch.argmax(action_logits, dim=-1).item()
        action_map = {0: "move", 1: "rest", 2: "deposit", 3: "follow"}
        
        return {
            "action": action_map.get(action_idx, "move"),
            "direction": direction.squeeze().tolist(),
            "intensity": intensity.squeeze().item(),
            "confidence": torch.softmax(action_logits, dim=-1).max().item(),
        }
    
    def batch_decide(
        self,
        observations: List[Dict[str, Any]],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Fast batch inference"""
        if contexts is None:
            contexts = [None] * len(observations)
        
        # Stack all observations
        tensors = [
            self._observation_to_tensor(obs, ctx)
            for obs, ctx in zip(observations, contexts)
        ]
        x = torch.stack(tensors).unsqueeze(1)  # [batch, 1, state_dim]
        
        self.model.eval()
        with torch.no_grad():
            action_logits, directions, intensities = self.model(x)
        
        action_map = {0: "move", 1: "rest", 2: "deposit", 3: "follow"}
        action_indices = torch.argmax(action_logits, dim=-1)
        confidences = torch.softmax(action_logits, dim=-1).max(dim=-1).values
        
        decisions = []
        for i in range(len(observations)):
            decisions.append({
                "action": action_map.get(action_indices[i].item(), "move"),
                "direction": directions[i].tolist(),
                "intensity": intensities[i].item(),
                "confidence": confidences[i].item(),
            })
        
        return decisions


class AgentInferenceEngine:
    """
    Main inference engine that manages brains for all agent tiers.
    
    Provides a unified interface for the swarm simulation.
    """
    
    def __init__(
        self,
        model_paths: Dict[str, str],
        configs: Optional[Dict[str, InferenceConfig]] = None,
    ):
        """
        Initialize inference engine with models for each tier.
        
        Args:
            model_paths: Dict mapping tier name to model path
            configs: Optional configs per tier
        """
        self.brains: Dict[str, AgentBrain] = {}
        
        for tier, path in model_paths.items():
            config = (configs or {}).get(tier, InferenceConfig(model_path=path))
            config.model_path = path
            
            if tier == "swarm_unit":
                # Use fast Mamba brain for swarm units
                self.brains[tier] = MambaAgentBrain(model_path=path)
            else:
                # Use LLM brain for higher tiers
                self.brains[tier] = DistilledAgentBrain(config, agent_tier=tier)
        
        logger.info(f"Inference engine ready with tiers: {list(self.brains.keys())}")
    
    def decide(
        self,
        tier: str,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get decision for a single agent"""
        if tier not in self.brains:
            raise ValueError(f"Unknown tier: {tier}")
        
        return self.brains[tier].decide(observation, context)
    
    def batch_decide(
        self,
        tier: str,
        observations: List[Dict[str, Any]],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Get decisions for multiple agents"""
        if tier not in self.brains:
            raise ValueError(f"Unknown tier: {tier}")
        
        return self.brains[tier].batch_decide(observations, contexts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        return {
            "active_tiers": list(self.brains.keys()),
            "memory_usage_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        }
