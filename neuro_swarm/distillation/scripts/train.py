#!/usr/bin/env python3
"""
Train Agent Models CLI

Full pipeline for distilling specialized agent models:
1. Generate training data (optional - can use existing)
2. Train with LoRA fine-tuning
3. Merge LoRA weights
4. Quantize for deployment

Usage:
    # Train triumvirate agent
    python train.py triumvirate --base-model Qwen/Qwen2.5-7B-Instruct
    
    # Train all agents
    python train.py all
    
    # Generate training data first
    python train.py triumvirate --generate-data --teacher anthropic
    
    # Just quantize an existing model
    python train.py triumvirate --quantize-only --model-path ./models/triumvirate_merged
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from neuro_swarm.distillation.distiller import AgentDistiller, DistillationConfig, AgentTier
from neuro_swarm.distillation.data_generator import (
    generate_training_data,
    AnthropicTeacher,
    LocalTeacher,
    TaskType,
)
from neuro_swarm.distillation.quantizer import quantize_agent_model, benchmark_quantized_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(tier: str, config_path: str = None) -> dict:
    """Load configuration for agent tier"""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "agent_configs.yaml"
    
    with open(config_path) as f:
        all_configs = yaml.safe_load(f)
    
    return all_configs.get(tier, {})


def generate_data(
    tier: str,
    teacher_type: str = "anthropic",
    teacher_model: str = None,
    num_examples: int = 1000,
    output_dir: str = "./training_data",
) -> Path:
    """Generate training data for a tier"""
    logger.info(f"Generating {num_examples} training examples for {tier}")
    
    output_files = generate_training_data(
        teacher_type=teacher_type,
        teacher_model=teacher_model,
        num_synthetic_states=num_examples // 3,  # Each state generates ~3 examples
        output_dir=output_dir,
        agent_tiers=[tier],
    )
    
    return output_files.get(tier)


def train_model(
    tier: str,
    base_model: str = None,
    data_path: str = "./training_data",
    output_dir: str = "./distilled_models",
    config_overrides: dict = None,
) -> str:
    """Train a model for the specified tier"""
    tier_config = load_config(tier)
    tier_enum = AgentTier(tier)
    
    # Select base model
    if base_model is None:
        recommended = tier_config.get("base_models", {}).get("recommended", [])
        if recommended:
            base_model = recommended[0]
        else:
            raise ValueError(f"No base model specified for {tier}")
    
    logger.info(f"Training {tier} with base model: {base_model}")
    
    # Build config
    lora_config = tier_config.get("lora", {})
    training_config = tier_config.get("training", {})
    
    config = DistillationConfig(
        base_model=base_model,
        agent_tier=tier_enum,
        lora_r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("alpha", 128),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        num_epochs=training_config.get("epochs", 3),
        batch_size=training_config.get("batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation", 4),
        learning_rate=training_config.get("learning_rate", 2e-4),
        max_seq_length=training_config.get("max_seq_length", 2048),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        data_path=data_path,
        output_dir=output_dir,
    )
    
    # Apply any overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Train
    distiller = AgentDistiller(config)
    checkpoint_path = distiller.train()
    
    # Merge LoRA weights
    merged_path = Path(output_dir) / f"{tier}_merged"
    distiller.merge_and_save(str(merged_path))
    
    return str(merged_path)


def quantize_model(
    model_path: str,
    method: str = "gptq",
    bits: int = 4,
    output_dir: str = "./quantized_models",
) -> str:
    """Quantize a trained model"""
    logger.info(f"Quantizing {model_path} with {method} {bits}-bit")
    
    quantized_path = quantize_agent_model(
        model_path=model_path,
        method=method,
        bits=bits,
        output_dir=output_dir,
    )
    
    # Benchmark
    logger.info("Running benchmark...")
    stats = benchmark_quantized_model(quantized_path)
    
    logger.info(f"Benchmark results:")
    logger.info(f"  Mean latency: {stats['mean_latency_ms']:.2f}ms")
    logger.info(f"  Memory: {stats.get('gpu_memory_allocated_gb', 0):.2f}GB")
    
    return quantized_path


def main():
    parser = argparse.ArgumentParser(
        description="Train and deploy agent models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "tier",
        choices=["triumvirate", "paladin", "chaos", "swarm_unit", "all"],
        help="Agent tier to train",
    )
    
    # Data generation
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate training data before training",
    )
    parser.add_argument(
        "--teacher",
        choices=["anthropic", "local"],
        default="anthropic",
        help="Teacher model type for data generation",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Specific teacher model (e.g., claude-sonnet-4-20250514 or path to local model)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=1000,
        help="Number of training examples to generate",
    )
    
    # Training
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./training_data",
        help="Path to training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./distilled_models",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        help="Use QLoRA (4-bit quantized training)",
    )
    
    # Quantization
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize model after training",
    )
    parser.add_argument(
        "--quantize-only",
        action="store_true",
        help="Only quantize (skip training)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model for quantize-only mode",
    )
    parser.add_argument(
        "--quant-method",
        choices=["gptq", "awq", "bnb"],
        default="gptq",
        help="Quantization method",
    )
    parser.add_argument(
        "--quant-bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bit width",
    )
    
    args = parser.parse_args()
    
    # Determine tiers to process
    if args.tier == "all":
        tiers = ["triumvirate", "paladin", "chaos", "swarm_unit"]
    else:
        tiers = [args.tier]
    
    for tier in tiers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing tier: {tier}")
        logger.info(f"{'='*60}\n")
        
        # Generate data if requested
        if args.generate_data:
            generate_data(
                tier=tier,
                teacher_type=args.teacher,
                teacher_model=args.teacher_model,
                num_examples=args.num_examples,
                output_dir=args.data_path,
            )
        
        # Train or quantize
        if args.quantize_only:
            if args.model_path is None:
                args.model_path = f"./distilled_models/{tier}_merged"
            
            quantize_model(
                model_path=args.model_path,
                method=args.quant_method,
                bits=args.quant_bits,
                output_dir="./quantized_models",
            )
        else:
            # Build config overrides
            overrides = {}
            if args.epochs:
                overrides["num_epochs"] = args.epochs
            if args.batch_size:
                overrides["batch_size"] = args.batch_size
            if args.use_qlora:
                overrides["use_4bit_quantization"] = True
            
            # Train
            merged_path = train_model(
                tier=tier,
                base_model=args.base_model,
                data_path=args.data_path,
                output_dir=args.output_dir,
                config_overrides=overrides if overrides else None,
            )
            
            # Quantize if requested
            if args.quantize:
                quantize_model(
                    model_path=merged_path,
                    method=args.quant_method,
                    bits=args.quant_bits,
                    output_dir="./quantized_models",
                )
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
