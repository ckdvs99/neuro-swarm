# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**Neuro-Swarm: Hierarchical State Space Navigation in Neuro-Inspired Multi-Agent Swarm Systems**

Research project exploring how principles from neuroscience, swarm intelligence, and state space models (SSMs) can be unified into a coherent framework for multi-agent coordination.

### Research Context

This codebase supports dissertation research on:
- **State Space Models for Multi-Agent Coordination** — Mamba-style selective state dynamics for swarm coordination
- **Hierarchical Consensus** — Triumvirate architecture for robust decision-making
- **Cyber-Physical Defense** — Paladin agents (defenders) vs Chaos agents (adversarial testing)
- **Neuro-Inspired Computing** — Principles from neuroscience applied to artificial swarms

### Target Publications
- NeurIPS / ICML: Theoretical foundations of SSM-based swarm coordination
- ICRA / IROS: Cyber-physical applications and robotic swarm implementation
- Nature Machine Intelligence (stretch): Unified theory of hierarchical swarm intelligence

---

## Core Principles

**READ `neuro_swarm/philosophy/PRINCIPLES.md` BEFORE MAKING ANY CHANGES.**

The eight guiding principles:

1. **Simplicity Generates Complexity** — Simple local rules create emergent behavior
2. **Balance Over Power** — Inhibition matters as much as excitation
3. **The Medium is the Message** — Stigmergy over direct communication
4. **Locality is Honesty** — No agent has global information (7±2 neighbors max)
5. **Rhythm, Not Reaction** — Biological systems pulse, they don't run constantly
6. **Observation Before Intervention** — Watch before you change
7. **Hierarchical Sovereignty** — Each level has its own logic
8. **State is Memory; Gating is Attention** — Selective information integration

### Design Philosophy

- Prefer removal over addition
- If behavior can emerge from interaction, don't encode it explicitly
- The agent that does less, correctly, is superior to one that does more, approximately
- We are not building a tool. We are cultivating a system.

---

## Build & Development Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,viz]"

# Run studies
python -m neuro_swarm.studies.01_single_agent.observe
python -m neuro_swarm.studies.02_pair_dynamics.observe --scenario symmetric
python -m neuro_swarm.studies.03_triad_consensus.observe
python -m neuro_swarm.studies.04_small_swarm.observe --agents 7

# Run tests
pytest
pytest tests/ -v

# Type check
mypy neuro_swarm/

# Format
black neuro_swarm/
ruff check neuro_swarm/
```

---

## Project Structure

```
neuro-swarm/
├── CLAUDE.md                 # This file - Claude Code guidance
├── README.md                 # Project overview
├── pyproject.toml            # Package configuration
│
├── neuro_swarm/              # Main package
│   ├── __init__.py
│   │
│   ├── philosophy/           # Guiding principles
│   │   └── PRINCIPLES.md
│   │
│   ├── core/                 # Core components
│   │   ├── agent.py          # NeuroAgent - the fundamental unit
│   │   ├── substrate.py      # Stigmergic communication layer
│   │   └── rhythm.py         # Temporal dynamics
│   │
│   ├── protocols/            # Interaction protocols
│   │   ├── attention.py      # Local neighborhood attention
│   │   ├── consensus.py      # Emergent agreement
│   │   └── tension.py        # Opposing forces (Paladin/Chaos)
│   │
│   ├── environments/         # Simulation environments
│   │   └── simple_field.py   # 2D continuous space
│   │
│   ├── observations/         # Visualization and analysis
│   │   └── visualize.py      # Swarm visualization tools
│   │
│   └── studies/              # Structured experiments
│       ├── 01_single_agent/  # Understand one before many
│       ├── 02_pair_dynamics/ # Two agents finding balance
│       ├── 03_triad_consensus/ # The triumvirate emerges
│       └── 04_small_swarm/   # 7±2 agents
│
├── tests/                    # Test suite
└── docs/                     # Documentation and assets
```

---

## Key Components

### NeuroAgent (`core/agent.py`)

The fundamental unit. An agent should be like a haiku: constrained, complete, resonant.

```python
from neuro_swarm.core import NeuroAgent, AgentConfig

config = AgentConfig(
    state_dim=8,              # Internal state dimensionality
    neighbor_limit=7,         # Biological constant (Miller's law)
    energy_decay=0.05,        # Cost of action
    energy_recovery=0.15,     # Benefit of rest
    rest_threshold=0.1,       # When rest becomes mandatory
    memory_persistence=0.9,   # SSM A matrix analog
    observation_weight=0.1,   # SSM B matrix analog
)

agent = NeuroAgent("agent_0", position, config)
```

Core loop:
1. `perceive()` — Gather information from neighbors and substrate
2. `update()` — Update internal state (SSM dynamics with selective gating)
3. `act()` — Produce behavior (velocity change)
4. `deposit()` — Leave trace in substrate (stigmergy)

### Substrate (`core/substrate.py`)

The stigmergic communication layer. Agents communicate through environment, not direct messaging.

- Traces persist, decay, and diffuse
- Local reads/writes only
- Implements Principle 3: The medium is the message

### Rhythm (`core/rhythm.py`)

Temporal dynamics. Biological systems have cycles.

- Oscillatory activity levels
- Phase coupling between agents (Kuramoto model)
- Refractory periods
- Implements Principle 5: Rhythm, not reaction

### SimpleField (`environments/simple_field.py`)

2D continuous space for experiments.

- Bounded or toroidal topology
- Substrate integration
- Neighbor queries respecting locality
- Step-based simulation

---

## Development Roadmap

### Phase 1: Foundations ✓
- [x] Core agent implementation
- [x] Substrate (stigmergic layer)
- [x] Rhythm (temporal dynamics)
- [x] Single agent study
- [x] Pair dynamics study

### Phase 2: Emergence (Current)
- [ ] Triad consensus study (implement consensus protocol)
- [ ] Small swarm (7±2 agents)
- [ ] Attention protocols
- [ ] Basic consensus mechanisms

### Phase 3: Scale
- [ ] Large swarm experiments (50-100 agents)
- [ ] Hierarchical organization
- [ ] Learned SSM parameters
- [ ] Performance benchmarks

### Phase 4: Application
- [ ] Cyber-physical defense scenarios
- [ ] Paladin/Chaos agent framework
- [ ] Critical infrastructure testbed
- [ ] Real-world deployment considerations

---

## The Triumvirate

Three is the smallest number that can achieve consensus without deadlock.

| Role | Function | In Code |
|------|----------|---------|
| **Preserver** | Maintains coherence, resists disruption | High memory_persistence |
| **Challenger** | Tests assumptions, probes weakness | High observation_weight |
| **Integrator** | Synthesizes, finds the path between | Balanced parameters |

In cyber-physical defense:
- **Paladin agents** preserve system stability
- **Chaos agents** test resilience
- The swarm itself integrates

---

## Constants and Constraints

### Biological Constants
```python
NEIGHBOR_LIMIT = 7          # Miller's law (7±2)
MIN_ENERGY_FOR_ACTION = 0.1 # Rest threshold
TRACE_DIM = 4               # Substrate trace dimensionality
```

### Forbidden Patterns
- **No global state access** — Agents only know local neighborhood
- **No direct messaging** — Use substrate (stigmergy)
- **No constant vigilance** — Agents must rest
- **No imposed consensus** — Agreement emerges from interaction

---

## Adding New Components

### New Study

1. Create directory: `neuro_swarm/studies/NN_study_name/`
2. Add `__init__.py` with study description and questions
3. Add `observe.py` with runnable observation script
4. Follow the pattern: observe first, hypothesize later

### New Protocol

1. Add to `neuro_swarm/protocols/`
2. Respect locality (no global access)
3. Document which principles it embodies
4. Include TODO markers for unimplemented methods

### New Agent Type

Extend `NeuroAgent` or create new config profiles:
```python
PALADIN_CONFIG = AgentConfig(
    energy_decay=0.03,
    memory_persistence=0.95,  # Strong preservation
)

CHAOS_CONFIG = AgentConfig(
    energy_decay=0.07,
    observation_weight=0.2,  # High reactivity
)
```

---

## Testing Philosophy

- Every study is a test
- Observation is methodology
- Log everything; storage is cheap; insight is precious
- Write studies, not just unit tests

---

## References

- **Mamba (SSM)**: Gu & Dao, 2023 - Selective state space models
- **Boids**: Reynolds, 1987 - Flocking behavior simulation
- **Kuramoto Model**: Synchronization in coupled oscillators
- **Stigmergy**: Grassé, 1959 - Indirect coordination through environment
- **Miller's Law**: The magical number seven, plus or minus two

---

## Acknowledgments

- FAU GenAI Lab
- Dr. Koch (advisor feedback)
- The starlings, ants, and neurons who taught us balance

---

*"We are not building a tool. We are cultivating a system."*
