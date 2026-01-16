#!/usr/bin/env python3
"""
Train Mamba-based Swarm Unit Models

For swarm units, we don't need a full LLM - a small state-space model
provides faster inference with the local coordination capabilities needed.

This script trains a Mamba SSM (or MLP fallback) for swarm unit decisions.

Usage:
    python train_mamba.py --data-path ./training_data --epochs 100
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SwarmUnitDataset(Dataset):
    """Dataset for swarm unit training"""
    
    def __init__(
        self,
        data_path: Path,
        state_dim: int = 32,
    ):
        self.state_dim = state_dim
        self.examples = self._load_data(data_path)
        
    def _load_data(self, data_path: Path) -> List[Dict]:
        """Load training examples"""
        examples = []
        data_file = Path(data_path) / "swarm_unit_training.jsonl"
        
        if not data_file.exists():
            logger.warning(f"No data found at {data_file}, generating synthetic")
            return self._generate_synthetic_data(1000)
        
        with open(data_file) as f:
            for line in f:
                examples.append(json.loads(line))
        
        logger.info(f"Loaded {len(examples)} examples")
        return examples
    
    def _generate_synthetic_data(self, n: int) -> List[Dict]:
        """Generate synthetic training data for bootstrapping"""
        import random
        
        examples = []
        for _ in range(n):
            # Random agent state
            position = [random.uniform(-10, 10), random.uniform(-10, 10)]
            velocity = [random.uniform(-1, 1), random.uniform(-1, 1)]
            energy = random.uniform(0.1, 1.0)
            
            # Random neighbors
            num_neighbors = random.randint(0, 7)
            neighbors = []
            for _ in range(num_neighbors):
                neighbors.append({
                    "relative_position": [random.uniform(-5, 5), random.uniform(-5, 5)],
                    "velocity": [random.uniform(-1, 1), random.uniform(-1, 1)],
                    "energy": random.uniform(0.1, 1.0),
                })
            
            # Determine target action based on simple rules (teacher behavior)
            # This would normally come from the teacher model
            action, direction, intensity = self._compute_target_action(
                position, velocity, energy, neighbors
            )
            
            examples.append({
                "observation": {
                    "position": position,
                    "velocity": velocity,
                    "energy": energy,
                },
                "neighbors": neighbors,
                "target_action": action,
                "target_direction": direction,
                "target_intensity": intensity,
            })
        
        return examples
    
    def _compute_target_action(
        self,
        position: List[float],
        velocity: List[float],
        energy: float,
        neighbors: List[Dict],
    ) -> Tuple[int, List[float], float]:
        """
        Compute target action using biological swarm rules.
        
        This implements simplified Boids-like behavior:
        - Separation: avoid crowding neighbors
        - Alignment: steer towards average heading of neighbors
        - Cohesion: steer towards center of neighbors
        """
        action = 0  # Default: move
        direction = [0.0, 0.0]
        intensity = 0.5
        
        if energy < 0.2:
            # Low energy → rest
            action = 1
            return action, direction, intensity
        
        if not neighbors:
            # No neighbors → deposit trace
            action = 2
            intensity = energy * 0.5
            return action, direction, intensity
        
        # Compute swarm forces
        separation = np.array([0.0, 0.0])
        alignment = np.array([0.0, 0.0])
        cohesion = np.array([0.0, 0.0])
        
        for n in neighbors:
            rel_pos = np.array(n["relative_position"])
            dist = np.linalg.norm(rel_pos)
            
            # Separation (inverse distance)
            if dist > 0.1:
                separation -= rel_pos / (dist ** 2)
            
            # Alignment
            alignment += np.array(n.get("velocity", [0, 0]))
            
            # Cohesion
            cohesion += rel_pos
        
        # Normalize and combine
        num_n = len(neighbors)
        alignment /= num_n
        cohesion /= num_n
        
        # Weighted sum
        direction = 0.4 * separation + 0.3 * alignment + 0.3 * cohesion
        
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm > 0.1:
            direction = direction / norm
        
        return 0, direction.tolist(), 0.5
    
    def _example_to_tensors(
        self,
        example: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert example to input/target tensors"""
        features = []
        
        # Agent state
        obs = example["observation"]
        features.extend(obs.get("position", [0, 0]))
        features.extend(obs.get("velocity", [0, 0]))
        features.append(obs.get("energy", 0.5))
        features.append(obs.get("phase", 0))
        
        # Neighbors (pad/truncate to 7)
        neighbors = example.get("neighbors", [])[:7]
        for n in neighbors:
            features.extend(n.get("relative_position", [0, 0]))
            features.append(n.get("energy", 0.5))
        # Pad
        for _ in range(7 - len(neighbors)):
            features.extend([0, 0, 0])
        
        # Pad to state_dim
        while len(features) < self.state_dim:
            features.append(0)
        
        x = torch.tensor(features[:self.state_dim], dtype=torch.float32)
        
        # Targets
        action = torch.tensor(example.get("target_action", 0), dtype=torch.long)
        direction = torch.tensor(example.get("target_direction", [0, 0]), dtype=torch.float32)
        intensity = torch.tensor(example.get("target_intensity", 0.5), dtype=torch.float32)
        
        return x, action, direction, intensity
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self._example_to_tensors(self.examples[idx])


class MambaDecisionModel(nn.Module):
    """Mamba-based decision model for swarm units"""
    
    def __init__(
        self,
        state_dim: int = 32,
        hidden_dim: int = 64,
        action_dim: int = 4,
        use_mamba: bool = True,
    ):
        super().__init__()
        
        self.encoder = nn.Linear(state_dim, hidden_dim)
        
        if use_mamba:
            try:
                from mamba_ssm import Mamba
                self.ssm = Mamba(
                    d_model=hidden_dim,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                )
                self.has_mamba = True
            except ImportError:
                logger.warning("mamba_ssm not available, using MLP")
                self.ssm = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.has_mamba = False
        else:
            self.ssm = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.has_mamba = False
        
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.direction_head = nn.Linear(hidden_dim, 2)
        self.intensity_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: [batch, state_dim]
        h = self.encoder(x)
        
        if self.has_mamba:
            # Mamba expects [batch, seq, hidden]
            h = h.unsqueeze(1)
            h = self.ssm(h)
            h = h.squeeze(1)
        else:
            h = self.ssm(h)
        
        action_logits = self.action_head(h)
        direction = torch.tanh(self.direction_head(h))
        intensity = torch.sigmoid(self.intensity_head(h))
        
        return action_logits, direction, intensity


def train_mamba_model(
    data_path: str,
    output_dir: str,
    state_dim: int = 32,
    hidden_dim: int = 64,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "auto",
) -> str:
    """
    Train a Mamba swarm unit model.
    
    Returns:
        Path to saved model weights
    """
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Training on {device}")
    
    # Load data
    dataset = SwarmUnitDataset(Path(data_path), state_dim=state_dim)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Build model
    model = MambaDecisionModel(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=4,
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss functions
    action_criterion = nn.CrossEntropyLoss()
    direction_criterion = nn.MSELoss()
    intensity_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training loop
    best_loss = float('inf')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            x, action_target, direction_target, intensity_target = batch
            x = x.to(device)
            action_target = action_target.to(device)
            direction_target = direction_target.to(device)
            intensity_target = intensity_target.to(device)
            
            # Forward
            action_logits, direction_pred, intensity_pred = model(x)
            
            # Compute losses
            action_loss = action_criterion(action_logits, action_target)
            direction_loss = direction_criterion(direction_pred, direction_target)
            intensity_loss = intensity_criterion(intensity_pred.squeeze(), intensity_target)
            
            loss = action_loss + direction_loss + intensity_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: loss = {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_path / "mamba_swarm_unit.pt")
    
    # Save final model
    torch.save(model.state_dict(), output_path / "mamba_swarm_unit_final.pt")
    
    # Save config
    config = {
        "state_dim": state_dim,
        "hidden_dim": hidden_dim,
        "action_dim": 4,
        "has_mamba": model.has_mamba,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f)
    
    logger.info(f"Model saved to {output_path}")
    logger.info(f"Best loss: {best_loss:.4f}")
    
    return str(output_path / "mamba_swarm_unit.pt")


def benchmark_mamba_model(model_path: str, num_agents: int = 1000) -> Dict[str, float]:
    """Benchmark inference speed for many agents"""
    import time
    
    # Load model
    with open(Path(model_path).parent / "config.json") as f:
        config = json.load(f)
    
    model = MambaDecisionModel(
        state_dim=config["state_dim"],
        hidden_dim=config["hidden_dim"],
        action_dim=config["action_dim"],
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Random batch of agents
    x = torch.randn(num_agents, config["state_dim"]).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    return {
        "num_agents": num_agents,
        "total_time_ms": elapsed * 1000 / 100,
        "time_per_agent_us": elapsed * 1e6 / (100 * num_agents),
        "throughput_agents_per_sec": (100 * num_agents) / elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Mamba swarm unit model")
    
    parser.add_argument("--data-path", type=str, default="./training_data")
    parser.add_argument("--output-dir", type=str, default="./distilled_models/swarm_unit")
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark existing model")
    parser.add_argument("--model-path", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.benchmark:
        if args.model_path is None:
            args.model_path = f"{args.output_dir}/mamba_swarm_unit.pt"
        
        logger.info("Running benchmark...")
        for num_agents in [100, 1000, 10000]:
            stats = benchmark_mamba_model(args.model_path, num_agents)
            logger.info(f"  {num_agents} agents: {stats['time_per_agent_us']:.2f}µs/agent, "
                       f"{stats['throughput_agents_per_sec']:.0f} agents/sec")
    else:
        train_mamba_model(
            data_path=args.data_path,
            output_dir=args.output_dir,
            state_dim=args.state_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
        )


if __name__ == "__main__":
    main()
