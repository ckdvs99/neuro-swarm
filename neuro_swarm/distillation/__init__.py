"""
Neuro-Swarm Model Distillation Pipeline

This module provides infrastructure for training specialized, locally-deployable
models for each agent tier in the neuro-swarm hierarchy:

- Triumvirate: Consensus/coordination models (~3-7B params)
- Paladin: Defensive/anomaly detection models (~1-3B params)  
- Chaos: Adversarial/attack generation models (~1-3B params)
- Swarm Units: Minimal state-space models (~100M-500M params or sub-LLM)

The pipeline supports:
- Teacher model data generation (using API or large local models)
- LoRA fine-tuning for efficient training
- INT4/INT8 quantization for deployment
- Integration with the existing NeuroAgent architecture
"""

from .distiller import AgentDistiller, DistillationConfig
from .data_generator import TaskDataGenerator, SimulationCapture
from .quantizer import ModelQuantizer
from .inference import AgentInferenceEngine

__all__ = [
    "AgentDistiller",
    "DistillationConfig", 
    "TaskDataGenerator",
    "SimulationCapture",
    "ModelQuantizer",
    "AgentInferenceEngine",
]
