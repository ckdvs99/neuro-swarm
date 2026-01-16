"""
Training Data Generation Pipeline

Generates training data for agent models by:
1. Running simulations to capture agent decision points
2. Querying teacher models (API or local) for optimal responses
3. Structuring data for distillation training

This is the critical bridge between your simulation work and model training.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Generator
from abc import ABC, abstractmethod
import json
import logging
import time
from datetime import datetime
import hashlib

import numpy as np

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks for different agent tiers"""
    # Triumvirate tasks
    CONSENSUS = "consensus"
    COORDINATION = "coordination"
    STRATEGIC_ALLOCATION = "strategic_allocation"
    CONFLICT_RESOLUTION = "conflict_resolution"
    
    # Paladin tasks
    ANOMALY_DETECTION = "anomaly_detection"
    THREAT_CLASSIFICATION = "threat_classification"
    DEFENSIVE_RESPONSE = "defensive_response"
    PATTERN_RECOGNITION = "pattern_recognition"
    
    # Chaos tasks
    ATTACK_GENERATION = "attack_generation"
    VULNERABILITY_PROBING = "vulnerability_probing"
    EVASION_PLANNING = "evasion_planning"
    
    # Swarm unit tasks
    LOCAL_COORDINATION = "local_coordination"
    NEIGHBOR_RESPONSE = "neighbor_response"
    STIGMERGIC_UPDATE = "stigmergic_update"


@dataclass
class SimulationState:
    """Captured state from a running simulation"""
    timestamp: float
    swarm_state: Dict[str, Any]
    agent_states: List[Dict[str, Any]]
    substrate_state: Optional[Dict[str, Any]] = None
    rhythm_phase: Optional[float] = None
    external_events: List[Dict[str, Any]] = field(default_factory=list)
    
    def serialize(self) -> str:
        """Convert to string representation for prompting"""
        lines = [
            f"Timestamp: {self.timestamp:.3f}",
            f"Swarm Size: {len(self.agent_states)}",
            f"Rhythm Phase: {self.rhythm_phase:.2f}" if self.rhythm_phase else "",
        ]
        
        # Summarize agent states
        if self.agent_states:
            positions = [a.get('position', [0, 0]) for a in self.agent_states]
            energies = [a.get('energy', 0) for a in self.agent_states]
            lines.append(f"Agent Positions (centroid): {np.mean(positions, axis=0).tolist()}")
            lines.append(f"Energy Range: [{min(energies):.2f}, {max(energies):.2f}]")
            lines.append(f"Mean Energy: {np.mean(energies):.2f}")
        
        # Include substrate info if present
        if self.substrate_state:
            trace_count = len(self.substrate_state.get('traces', []))
            lines.append(f"Active Traces: {trace_count}")
            
        return "\n".join(filter(None, lines))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp,
            "swarm_state": self.swarm_state,
            "agent_states": self.agent_states,
            "substrate_state": self.substrate_state,
            "rhythm_phase": self.rhythm_phase,
            "external_events": self.external_events,
        }


@dataclass
class TrainingExample:
    """A single training example for distillation"""
    example_id: str
    task_type: TaskType
    input_state: SimulationState
    task_description: str
    teacher_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_jsonl(self) -> str:
        """Convert to JSONL format for storage"""
        return json.dumps({
            "id": self.example_id,
            "task_type": self.task_type.value,
            "swarm_state": self.input_state.serialize(),
            "agent_states": json.dumps(self.input_state.agent_states),
            "task": self.task_description,
            "response": self.teacher_response,
            "metadata": self.metadata,
        })


class TeacherModel(ABC):
    """Abstract base class for teacher models"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the teacher model"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Return information about the teacher model"""
        pass


class AnthropicTeacher(TeacherModel):
    """Teacher model using Anthropic API (Claude)"""
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
            
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
    def generate(self, prompt: str, max_tokens: int = 2048, **kwargs) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return message.content[0].text
    
    def get_model_info(self) -> Dict[str, str]:
        return {"provider": "anthropic", "model": self.model}


class LocalTeacher(TeacherModel):
    """Teacher model using local model via transformers"""
    
    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model_path = model_path
        
    def generate(self, prompt: str, max_tokens: int = 2048, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            **kwargs,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        return response[len(prompt):].strip()
    
    def get_model_info(self) -> Dict[str, str]:
        return {"provider": "local", "model": self.model_path}


class SimulationCapture:
    """
    Captures decision points from running simulations for training data.
    
    Integrates with your NeuroAgent/Substrate/Rhythm architecture.
    """
    
    def __init__(
        self,
        capture_interval: float = 1.0,  # Capture every N seconds
        decision_threshold: float = 0.5,  # Capture when decision entropy > threshold
    ):
        self.capture_interval = capture_interval
        self.decision_threshold = decision_threshold
        self.captured_states: List[SimulationState] = []
        self.last_capture_time = 0.0
        
    def should_capture(
        self,
        current_time: float,
        decision_entropy: Optional[float] = None,
    ) -> bool:
        """Determine if we should capture the current state"""
        # Time-based capture
        if current_time - self.last_capture_time >= self.capture_interval:
            return True
            
        # Decision-point capture (high entropy = uncertain = good training example)
        if decision_entropy and decision_entropy > self.decision_threshold:
            return True
            
        return False
    
    def capture(
        self,
        agents: List[Any],  # List of NeuroAgent instances
        substrate: Optional[Any] = None,
        rhythm: Optional[Any] = None,
        external_events: Optional[List[Dict]] = None,
    ) -> SimulationState:
        """
        Capture current simulation state.
        
        This should be called from your simulation loop.
        """
        current_time = time.time()
        
        # Extract agent states
        agent_states = []
        for agent in agents:
            agent_states.append({
                "id": getattr(agent, 'agent_id', id(agent)),
                "position": getattr(agent, 'position', [0, 0]),
                "velocity": getattr(agent, 'velocity', [0, 0]),
                "energy": getattr(agent, 'energy', 1.0),
                "internal_state": getattr(agent, 'state', None),
                "phase": getattr(agent, 'phase', 0.0),
            })
        
        # Extract substrate state if available
        substrate_state = None
        if substrate:
            substrate_state = {
                "traces": getattr(substrate, 'traces', []),
                "grid_shape": getattr(substrate, 'shape', None),
            }
        
        # Extract rhythm phase
        rhythm_phase = None
        if rhythm:
            rhythm_phase = getattr(rhythm, 'phase', None)
        
        # Compute swarm-level metrics
        positions = np.array([a["position"] for a in agent_states])
        swarm_state = {
            "centroid": positions.mean(axis=0).tolist(),
            "dispersion": float(np.std(positions)),
            "agent_count": len(agents),
        }
        
        state = SimulationState(
            timestamp=current_time,
            swarm_state=swarm_state,
            agent_states=agent_states,
            substrate_state=substrate_state,
            rhythm_phase=rhythm_phase,
            external_events=external_events or [],
        )
        
        self.captured_states.append(state)
        self.last_capture_time = current_time
        
        return state
    
    def get_captured_states(self) -> List[SimulationState]:
        """Return all captured states"""
        return self.captured_states
    
    def clear(self):
        """Clear captured states"""
        self.captured_states = []


class TaskDataGenerator:
    """
    Generates training data by combining simulation captures with teacher model responses.
    
    This is the main orchestrator for your distillation data pipeline.
    """
    
    def __init__(
        self,
        teacher: TeacherModel,
        output_dir: str = "./training_data",
        examples_per_task: int = 1000,
    ):
        self.teacher = teacher
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.examples_per_task = examples_per_task
        
    def _generate_example_id(self, task_type: TaskType, state: SimulationState) -> str:
        """Generate unique ID for training example"""
        content = f"{task_type.value}_{state.timestamp}_{json.dumps(state.swarm_state)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _build_triumvirate_prompt(
        self,
        state: SimulationState,
        task_type: TaskType,
    ) -> tuple[str, str]:
        """Build prompt for triumvirate-level tasks"""
        task_descriptions = {
            TaskType.CONSENSUS: "Analyze the current swarm state and determine the consensus action all agents should take.",
            TaskType.COORDINATION: "Coordinate agent movements to achieve optimal formation while maintaining swarm cohesion.",
            TaskType.STRATEGIC_ALLOCATION: "Allocate agents to different tasks based on the current threat landscape.",
            TaskType.CONFLICT_RESOLUTION: "Resolve the conflicting agent goals while maintaining swarm integrity.",
        }
        
        task_desc = task_descriptions.get(task_type, "Analyze and respond to the swarm state.")
        
        prompt = f"""You are a triumvirate coordinator in a neuro-inspired multi-agent swarm system.
Your decisions guide the entire swarm through consensus-based coordination.

## Current Swarm State
{state.serialize()}

## Detailed Agent States
{json.dumps(state.agent_states, indent=2)}

## Task
{task_desc}

## Instructions
1. Analyze the swarm's current state and individual agent conditions
2. Consider the emergent behavior patterns
3. Determine the optimal coordination strategy
4. Provide specific, actionable guidance for the swarm

Respond with:
- ANALYSIS: Brief assessment of current state
- STRATEGY: Recommended coordination approach
- ACTIONS: Specific instructions for agents
- RATIONALE: Why this approach optimizes swarm performance
"""
        
        return prompt, task_desc
    
    def _build_paladin_prompt(
        self,
        state: SimulationState,
        task_type: TaskType,
    ) -> tuple[str, str]:
        """Build prompt for paladin defensive tasks"""
        task_descriptions = {
            TaskType.ANOMALY_DETECTION: "Identify any anomalous patterns in the current network/swarm state.",
            TaskType.THREAT_CLASSIFICATION: "Classify the detected threats by severity and type.",
            TaskType.DEFENSIVE_RESPONSE: "Recommend defensive actions to mitigate identified threats.",
            TaskType.PATTERN_RECOGNITION: "Identify patterns that may indicate emerging threats.",
        }
        
        task_desc = task_descriptions.get(task_type, "Analyze for threats.")
        
        # Simulate network signals for training
        signals = self._generate_synthetic_signals(state)
        
        prompt = f"""You are a Paladin defensive agent protecting critical infrastructure.
Your role is to detect threats and coordinate defensive responses.

## Network State
{state.serialize()}

## Incoming Signals
{json.dumps(signals, indent=2)}

## Historical Baseline
- Normal traffic range: 100-500 events/sec
- Typical agent count: {len(state.agent_states)}
- Expected latency: <50ms

## Task
{task_desc}

Respond with:
- THREAT_LEVEL: (NONE/LOW/MEDIUM/HIGH/CRITICAL)
- ANOMALIES: List of detected anomalies
- CLASSIFICATION: Type of threat if any
- RECOMMENDED_ACTION: Defensive response
- CONFIDENCE: Assessment confidence (0-1)
"""
        
        return prompt, task_desc
    
    def _build_chaos_prompt(
        self,
        state: SimulationState,
        task_type: TaskType,
    ) -> tuple[str, str]:
        """Build prompt for chaos adversarial tasks"""
        task_descriptions = {
            TaskType.ATTACK_GENERATION: "Generate a realistic attack scenario to test defenses.",
            TaskType.VULNERABILITY_PROBING: "Identify potential vulnerabilities in the current state.",
            TaskType.EVASION_PLANNING: "Plan evasion tactics to bypass current defenses.",
        }
        
        task_desc = task_descriptions.get(task_type, "Test system defenses.")
        
        prompt = f"""You are a Chaos agent conducting red team simulation.
Your role is to identify vulnerabilities and test defenses through simulated attacks.
This is ethical security testing within defined boundaries.

## Target System State
{state.serialize()}

## Known Defenses
- Active Paladin agents: {len([a for a in state.agent_states if a.get('energy', 0) > 0.5])}
- Detection threshold: 0.7 anomaly score
- Response latency: ~100ms

## Task
{task_desc}

## Ethical Boundaries
- Simulation only - no actual system impact
- Test defense capabilities, not cause harm
- Document all attack vectors for defensive improvement

Respond with:
- ATTACK_VECTOR: Description of attack approach
- EXPECTED_DETECTION: Will defenses catch this? Why/why not?
- EVASION_TACTICS: How to avoid detection
- DEFENSIVE_RECOMMENDATIONS: How to prevent this attack
"""
        
        return prompt, task_desc
    
    def _build_swarm_unit_prompt(
        self,
        state: SimulationState,
        agent_idx: int,
        task_type: TaskType,
    ) -> tuple[str, str]:
        """Build prompt for basic swarm unit tasks"""
        task_desc = "Determine local action based on observation and neighbors."
        
        agent = state.agent_states[agent_idx]
        
        # Get neighbors (within radius)
        neighbors = []
        for i, other in enumerate(state.agent_states):
            if i != agent_idx:
                dist = np.linalg.norm(
                    np.array(agent["position"]) - np.array(other["position"])
                )
                if dist < 5.0:  # Neighbor radius
                    neighbors.append({
                        "relative_position": (np.array(other["position"]) - np.array(agent["position"])).tolist(),
                        "velocity": other["velocity"],
                        "energy": other["energy"],
                    })
        
        prompt = f"""You are a swarm unit agent making local decisions.
Coordinate with neighbors using simple rules.

## Your State
Position: {agent['position']}
Velocity: {agent['velocity']}
Energy: {agent['energy']}

## Neighbor States (within sensing radius)
{json.dumps(neighbors[:7], indent=2)}  # Limited to 7 neighbors (starling rule)

## Substrate Traces Nearby
{state.substrate_state.get('traces', [])[:5] if state.substrate_state else 'None'}

## Task
{task_desc}

Respond with just:
- ACTION: (move/rest/deposit/follow)
- DIRECTION: [dx, dy] if moving
- INTENSITY: trace intensity if depositing (0-1)
"""
        
        return prompt, task_desc
    
    def _generate_synthetic_signals(self, state: SimulationState) -> List[Dict]:
        """Generate synthetic network signals for training"""
        import random
        
        signals = []
        for i in range(random.randint(5, 20)):
            signal_type = random.choice(["packet", "request", "heartbeat", "alert"])
            signals.append({
                "type": signal_type,
                "source": f"node_{random.randint(1, 100)}",
                "timestamp": state.timestamp + random.uniform(0, 1),
                "size": random.randint(64, 1500),
                "anomaly_score": random.uniform(0, 1),
            })
        return signals
    
    def generate_examples(
        self,
        states: List[SimulationState],
        task_types: List[TaskType],
        agent_tier: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Generator[TrainingExample, None, None]:
        """
        Generate training examples from captured states.
        
        Args:
            states: List of captured simulation states
            task_types: Types of tasks to generate
            agent_tier: 'triumvirate', 'paladin', 'chaos', or 'swarm_unit'
            progress_callback: Optional callback(current, total) for progress
            
        Yields:
            TrainingExample objects
        """
        total = len(states) * len(task_types)
        current = 0
        
        for state in states:
            for task_type in task_types:
                # Build appropriate prompt
                if agent_tier == "triumvirate":
                    prompt, task_desc = self._build_triumvirate_prompt(state, task_type)
                elif agent_tier == "paladin":
                    prompt, task_desc = self._build_paladin_prompt(state, task_type)
                elif agent_tier == "chaos":
                    prompt, task_desc = self._build_chaos_prompt(state, task_type)
                else:  # swarm_unit
                    # Generate for multiple agents
                    for agent_idx in range(min(3, len(state.agent_states))):
                        prompt, task_desc = self._build_swarm_unit_prompt(
                            state, agent_idx, task_type
                        )
                        
                        try:
                            teacher_response = self.teacher.generate(prompt)
                            
                            example = TrainingExample(
                                example_id=self._generate_example_id(task_type, state),
                                task_type=task_type,
                                input_state=state,
                                task_description=task_desc,
                                teacher_response=teacher_response,
                                metadata={
                                    "agent_idx": agent_idx,
                                    "teacher": self.teacher.get_model_info(),
                                    "generated_at": datetime.now().isoformat(),
                                },
                            )
                            yield example
                        except Exception as e:
                            logger.error(f"Failed to generate example: {e}")
                    continue
                
                # Query teacher model
                try:
                    teacher_response = self.teacher.generate(prompt)
                    
                    example = TrainingExample(
                        example_id=self._generate_example_id(task_type, state),
                        task_type=task_type,
                        input_state=state,
                        task_description=task_desc,
                        teacher_response=teacher_response,
                        metadata={
                            "teacher": self.teacher.get_model_info(),
                            "generated_at": datetime.now().isoformat(),
                        },
                    )
                    
                    yield example
                    
                except Exception as e:
                    logger.error(f"Failed to generate example: {e}")
                
                current += 1
                if progress_callback:
                    progress_callback(current, total)
    
    def save_examples(
        self,
        examples: List[TrainingExample],
        agent_tier: str,
    ) -> Path:
        """Save generated examples to JSONL file"""
        output_file = self.output_dir / f"{agent_tier}_training.jsonl"
        
        with open(output_file, "a") as f:
            for example in examples:
                f.write(example.to_jsonl() + "\n")
        
        logger.info(f"Saved {len(examples)} examples to {output_file}")
        return output_file


def generate_training_data(
    teacher_type: str = "anthropic",
    teacher_model: Optional[str] = None,
    simulation_states: Optional[List[SimulationState]] = None,
    num_synthetic_states: int = 100,
    output_dir: str = "./training_data",
    agent_tiers: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Convenience function to generate training data for all agent tiers.
    
    Args:
        teacher_type: 'anthropic' or 'local'
        teacher_model: Model name/path
        simulation_states: Pre-captured states (or generate synthetic)
        num_synthetic_states: Number of synthetic states if none provided
        output_dir: Where to save training data
        agent_tiers: Which tiers to generate for (default: all)
        
    Returns:
        Dictionary mapping agent_tier -> output file path
    """
    # Initialize teacher
    if teacher_type == "anthropic":
        teacher = AnthropicTeacher(model=teacher_model or "claude-sonnet-4-20250514")
    else:
        teacher = LocalTeacher(model_path=teacher_model)
    
    generator = TaskDataGenerator(teacher=teacher, output_dir=output_dir)
    
    # Generate synthetic states if none provided
    if simulation_states is None:
        logger.info(f"Generating {num_synthetic_states} synthetic simulation states")
        simulation_states = _generate_synthetic_states(num_synthetic_states)
    
    # Default to all tiers
    if agent_tiers is None:
        agent_tiers = ["triumvirate", "paladin", "chaos", "swarm_unit"]
    
    # Task types per tier
    tier_tasks = {
        "triumvirate": [TaskType.CONSENSUS, TaskType.COORDINATION, TaskType.STRATEGIC_ALLOCATION],
        "paladin": [TaskType.ANOMALY_DETECTION, TaskType.THREAT_CLASSIFICATION, TaskType.DEFENSIVE_RESPONSE],
        "chaos": [TaskType.ATTACK_GENERATION, TaskType.VULNERABILITY_PROBING],
        "swarm_unit": [TaskType.LOCAL_COORDINATION, TaskType.NEIGHBOR_RESPONSE],
    }
    
    output_files = {}
    
    for tier in agent_tiers:
        logger.info(f"Generating training data for {tier}")
        
        examples = list(generator.generate_examples(
            states=simulation_states,
            task_types=tier_tasks[tier],
            agent_tier=tier,
            progress_callback=lambda c, t: logger.info(f"{tier}: {c}/{t}") if c % 10 == 0 else None,
        ))
        
        output_files[tier] = generator.save_examples(examples, tier)
    
    return output_files


def _generate_synthetic_states(n: int) -> List[SimulationState]:
    """Generate synthetic simulation states for training data bootstrap"""
    import random
    
    states = []
    for i in range(n):
        # Random swarm configuration
        num_agents = random.randint(3, 50)
        
        agent_states = []
        for j in range(num_agents):
            agent_states.append({
                "id": f"agent_{j}",
                "position": [random.uniform(-10, 10), random.uniform(-10, 10)],
                "velocity": [random.uniform(-1, 1), random.uniform(-1, 1)],
                "energy": random.uniform(0.1, 1.0),
                "phase": random.uniform(0, 2 * np.pi),
            })
        
        positions = np.array([a["position"] for a in agent_states])
        
        state = SimulationState(
            timestamp=i * 0.1,
            swarm_state={
                "centroid": positions.mean(axis=0).tolist(),
                "dispersion": float(np.std(positions)),
                "agent_count": num_agents,
            },
            agent_states=agent_states,
            substrate_state={
                "traces": [
                    {"position": [random.uniform(-10, 10), random.uniform(-10, 10)], "intensity": random.uniform(0.1, 1.0)}
                    for _ in range(random.randint(0, 20))
                ]
            },
            rhythm_phase=random.uniform(0, 2 * np.pi),
        )
        states.append(state)
    
    return states
