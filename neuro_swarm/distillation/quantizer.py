"""
Model Quantization for Edge Deployment

Provides INT4/INT8 quantization to reduce model size and enable
fast inference on edge devices or constrained environments.

Supports:
- GPTQ quantization (best quality, requires calibration)
- AWQ quantization (faster, good quality)
- bitsandbytes quantization (easiest, good for inference)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import logging
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    
    # Quantization method
    method: str = "gptq"  # 'gptq', 'awq', or 'bnb'
    
    # Bit width
    bits: int = 4  # 4 or 8
    
    # GPTQ-specific
    group_size: int = 128
    desc_act: bool = True  # Activation order (improves quality)
    use_exllama: bool = True  # Fast inference kernel
    
    # AWQ-specific
    zero_point: bool = True
    fuse_layers: bool = True
    
    # Calibration
    calibration_samples: int = 128
    calibration_seq_length: int = 2048
    
    # Output
    output_dir: str = "./quantized_models"


class ModelQuantizer:
    """
    Quantize trained models for efficient deployment.
    
    After training with LoRA and merging, use this to create
    INT4/INT8 versions that run faster with less memory.
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def quantize_gptq(
        self,
        model_path: str,
        calibration_data: Optional[List[str]] = None,
        output_name: Optional[str] = None,
    ) -> str:
        """
        Quantize using GPTQ (best quality).
        
        Args:
            model_path: Path to merged model
            calibration_data: List of text samples for calibration
            output_name: Name for output directory
            
        Returns:
            Path to quantized model
        """
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise ImportError("Install auto-gptq: pip install auto-gptq")
        
        logger.info(f"Loading model from {model_path} for GPTQ quantization")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Prepare calibration data
        if calibration_data is None:
            calibration_data = self._get_default_calibration_data()
        
        calibration_dataset = self._prepare_calibration_dataset(
            calibration_data, tokenizer
        )
        
        # Configure quantization
        quantize_config = BaseQuantizeConfig(
            bits=self.config.bits,
            group_size=self.config.group_size,
            desc_act=self.config.desc_act,
        )
        
        # Load model for quantization
        model = AutoGPTQForCausalLM.from_pretrained(
            model_path,
            quantize_config=quantize_config,
        )
        
        # Quantize
        logger.info("Running GPTQ quantization (this may take a while)...")
        model.quantize(calibration_dataset)
        
        # Save
        output_name = output_name or Path(model_path).name + f"_gptq_{self.config.bits}bit"
        output_path = self.output_dir / output_name
        
        model.save_quantized(str(output_path), use_safetensors=True)
        tokenizer.save_pretrained(str(output_path))
        
        # Save config
        self._save_quantization_info(output_path, "gptq")
        
        logger.info(f"GPTQ quantized model saved to {output_path}")
        return str(output_path)
    
    def quantize_awq(
        self,
        model_path: str,
        calibration_data: Optional[List[str]] = None,
        output_name: Optional[str] = None,
    ) -> str:
        """
        Quantize using AWQ (faster than GPTQ, good quality).
        
        Args:
            model_path: Path to merged model
            calibration_data: List of text samples for calibration
            output_name: Name for output directory
            
        Returns:
            Path to quantized model
        """
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError("Install awq: pip install autoawq")
        
        logger.info(f"Loading model from {model_path} for AWQ quantization")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Prepare calibration data
        if calibration_data is None:
            calibration_data = self._get_default_calibration_data()
        
        # Load model for quantization
        model = AutoAWQForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        # Quantize
        logger.info("Running AWQ quantization...")
        quant_config = {
            "zero_point": self.config.zero_point,
            "q_group_size": self.config.group_size,
            "w_bit": self.config.bits,
        }
        
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calibration_data[:self.config.calibration_samples],
        )
        
        # Save
        output_name = output_name or Path(model_path).name + f"_awq_{self.config.bits}bit"
        output_path = self.output_dir / output_name
        
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        # Optionally fuse layers for faster inference
        if self.config.fuse_layers:
            logger.info("Fusing layers for optimized inference...")
            model.fuse_layers()
        
        self._save_quantization_info(output_path, "awq")
        
        logger.info(f"AWQ quantized model saved to {output_path}")
        return str(output_path)
    
    def quantize_bnb(
        self,
        model_path: str,
        output_name: Optional[str] = None,
    ) -> str:
        """
        Prepare model for bitsandbytes inference (no pre-quantization needed).
        
        This creates a config that loads the model in quantized form at inference time.
        Useful when you want the original weights but quantized inference.
        
        Args:
            model_path: Path to merged model
            output_name: Name for output directory
            
        Returns:
            Path to prepared model
        """
        from transformers import BitsAndBytesConfig
        
        logger.info(f"Preparing model from {model_path} for BnB inference")
        
        # Create BnB config
        if self.config.bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load with quantization config
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Save (note: this saves the quantized state)
        output_name = output_name or Path(model_path).name + f"_bnb_{self.config.bits}bit"
        output_path = self.output_dir / output_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        # Save BnB config for loading
        with open(output_path / "bnb_config.json", "w") as f:
            json.dump({
                "bits": self.config.bits,
                "quant_type": "nf4" if self.config.bits == 4 else "int8",
            }, f)
        
        self._save_quantization_info(output_path, "bnb")
        
        logger.info(f"BnB prepared model saved to {output_path}")
        return str(output_path)
    
    def quantize(
        self,
        model_path: str,
        calibration_data: Optional[List[str]] = None,
        output_name: Optional[str] = None,
    ) -> str:
        """
        Quantize using the configured method.
        
        Args:
            model_path: Path to merged model
            calibration_data: List of text samples for calibration
            output_name: Name for output directory
            
        Returns:
            Path to quantized model
        """
        if self.config.method == "gptq":
            return self.quantize_gptq(model_path, calibration_data, output_name)
        elif self.config.method == "awq":
            return self.quantize_awq(model_path, calibration_data, output_name)
        elif self.config.method == "bnb":
            return self.quantize_bnb(model_path, output_name)
        else:
            raise ValueError(f"Unknown quantization method: {self.config.method}")
    
    def _prepare_calibration_dataset(
        self,
        texts: List[str],
        tokenizer,
    ) -> List[Dict[str, torch.Tensor]]:
        """Prepare calibration data for GPTQ"""
        dataset = []
        
        for text in texts[:self.config.calibration_samples]:
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=self.config.calibration_seq_length,
                return_tensors="pt",
            )
            dataset.append(tokens)
        
        return dataset
    
    def _get_default_calibration_data(self) -> List[str]:
        """Get default calibration prompts based on agent tasks"""
        # These should match the types of prompts the model will see in production
        return [
            # Triumvirate examples
            """You are a triumvirate coordinator in a neuro-inspired multi-agent swarm system.
## Current Swarm State
Timestamp: 1.234
Swarm Size: 25
Agent Positions (centroid): [2.5, -1.3]
Energy Range: [0.45, 0.92]
Mean Energy: 0.71

## Task
Coordinate agent movements to achieve optimal formation while maintaining swarm cohesion.

Respond with analysis, strategy, actions, and rationale.""",
            
            # Paladin examples
            """You are a Paladin defensive agent protecting critical infrastructure.
## Network State
Timestamp: 5.678
Swarm Size: 50
Active Traces: 15

## Incoming Signals
[{"type": "packet", "source": "node_42", "anomaly_score": 0.85}]

## Task
Identify any anomalous patterns in the current network/swarm state.

Respond with threat level, anomalies, classification, and recommended action.""",
            
            # Chaos examples
            """You are a Chaos agent conducting red team simulation.
## Target System State
Timestamp: 10.0
Swarm Size: 30
Dispersion: 3.45

## Known Defenses
Active Paladin agents: 5
Detection threshold: 0.7

## Task
Generate a realistic attack scenario to test defenses.

Respond with attack vector, expected detection, evasion tactics, and recommendations.""",
            
            # Swarm unit examples
            """You are a swarm unit agent making local decisions.
## Your State
Position: [1.2, -0.5]
Velocity: [0.1, 0.3]
Energy: 0.75

## Neighbor States
[{"relative_position": [0.5, 0.2], "velocity": [0.0, 0.1], "energy": 0.8}]

## Task
Determine local action based on observation and neighbors.

Respond with action, direction, and intensity.""",
        ] * 32  # Repeat to get enough samples
    
    def _save_quantization_info(self, output_path: Path, method: str):
        """Save quantization metadata"""
        info = {
            "method": method,
            "bits": self.config.bits,
            "group_size": self.config.group_size,
            "calibration_samples": self.config.calibration_samples,
        }
        
        with open(output_path / "quantization_info.json", "w") as f:
            json.dump(info, f, indent=2)


def quantize_agent_model(
    model_path: str,
    method: str = "gptq",
    bits: int = 4,
    output_dir: str = "./quantized_models",
    calibration_data: Optional[List[str]] = None,
) -> str:
    """
    Convenience function to quantize a trained agent model.
    
    Args:
        model_path: Path to the merged model
        method: Quantization method ('gptq', 'awq', 'bnb')
        bits: Bit width (4 or 8)
        output_dir: Where to save quantized model
        calibration_data: Optional calibration texts
        
    Returns:
        Path to quantized model
    """
    config = QuantizationConfig(
        method=method,
        bits=bits,
        output_dir=output_dir,
    )
    
    quantizer = ModelQuantizer(config)
    return quantizer.quantize(model_path, calibration_data)


def benchmark_quantized_model(
    model_path: str,
    test_prompts: Optional[List[str]] = None,
    num_runs: int = 10,
) -> Dict[str, float]:
    """
    Benchmark inference speed and memory usage of quantized model.
    
    Args:
        model_path: Path to quantized model
        test_prompts: List of test prompts
        num_runs: Number of inference runs
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if test_prompts is None:
        test_prompts = [
            "Analyze the swarm state and determine coordination.",
            "Detect anomalies in the network traffic.",
            "Plan defensive response to threat.",
        ]
    
    # Warmup
    for prompt in test_prompts[:2]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        _ = model.generate(**inputs, max_new_tokens=50)
    
    # Benchmark
    latencies = []
    for _ in range(num_runs):
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            
            _ = model.generate(**inputs, max_new_tokens=100)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latencies.append(time.perf_counter() - start)
    
    # Memory stats
    memory_stats = {}
    if torch.cuda.is_available():
        memory_stats = {
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        }
    
    return {
        "mean_latency_ms": sum(latencies) / len(latencies) * 1000,
        "min_latency_ms": min(latencies) * 1000,
        "max_latency_ms": max(latencies) * 1000,
        "throughput_samples_per_sec": len(latencies) / sum(latencies),
        **memory_stats,
    }
