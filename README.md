# Neuro-Swarm

**Hierarchical State Space Navigation in Neuro-Inspired Multi-Agent Swarm Systems**

> "Simplicity generates complexity. Balance over power. The medium is the message."

---

## Vision

This project explores how principles from neuroscience, swarm intelligence, and state space models can be unified into a coherent framework for multi-agent coordination. We draw inspiration from:

- **Starling murmurations** — coherent motion from local rules
- **Cortical microcircuits** — balance of excitation and inhibition
- **Ant colonies** — stigmergic communication through environment
- **Mamba/SSM architectures** — selective state space models

The goal is not to build the most powerful system, but the most *balanced* one.

---

## Core Principles

Read [neuro_swarm/philosophy/PRINCIPLES.md](neuro_swarm/philosophy/PRINCIPLES.md) before diving into code. The principles are:

1. **Simplicity Generates Complexity** — simple local rules create emergent behavior
2. **Balance Over Power** — inhibition matters as much as excitation
3. **The Medium is the Message** — stigmergy over direct communication
4. **Locality is Honesty** — no agent has global information
5. **Rhythm, Not Reaction** — biological systems pulse, they don't run constantly
6. **Observation Before Intervention** — watch before you change
7. **Hierarchical Sovereignty** — each level has its own logic
8. **State is Memory; Gating is Attention** — selective information integration

---

## Project Structure

```
neuro-swarm/
├── CLAUDE.md                  # Claude Code guidance
├── README.md                  # This file
├── pyproject.toml             # Package configuration
├── Makefile                   # Development commands
│
├── neuro_swarm/               # Main package
│   ├── philosophy/
│   │   └── PRINCIPLES.md      # Guiding philosophy
│   │
│   ├── core/
│   │   ├── agent.py           # The NeuroAgent - minimal, complete
│   │   ├── substrate.py       # Stigmergic communication layer
│   │   └── rhythm.py          # Temporal dynamics
│   │
│   ├── evolution/             # Distributed evolutionary optimization
│   │   ├── algorithms.py      # ES, MAP-Elites, Novelty Search
│   │   ├── genome.py          # Genome representations
│   │   └── fitness.py         # Fitness functions
│   │
│   ├── protocols/
│   │   ├── attention.py       # Local neighborhood attention
│   │   ├── consensus.py       # Emergent agreement
│   │   └── tension.py         # Opposing forces (Paladin/Chaos)
│   │
│   ├── environments/
│   │   └── simple_field.py    # 2D continuous space
│   │
│   ├── observations/
│   │   └── visualize.py       # Watch. Learn. Adjust.
│   │
│   └── studies/
│       ├── 01_single_agent/   # Understand one before many
│       ├── 02_pair_dynamics/  # Two agents finding balance
│       ├── 03_triad_consensus/# The triumvirate emerges
│       └── 04_small_swarm/    # 7±2 agents
│
├── deploy/                    # Deployment configurations
│   └── k8s/                   # Kubernetes manifests
│
├── tests/                     # Test suite
└── docs/                      # Documentation
    └── DEPLOYMENT.md          # Kubernetes deployment guide
```

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd neuro-swarm

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Or with optional dependencies
pip install -e ".[dev,viz]"
```

### Running Studies

```bash
# Using Make (recommended)
make study-01      # Single agent
make study-02      # Pair dynamics
make study-03      # Triad consensus
make study-04      # Small swarm

# Or directly with options
python -m neuro_swarm.studies.01_single_agent.observe --steps 1000 --no-animate
python -m neuro_swarm.studies.02_pair_dynamics.observe --scenario pursuer_evader
```

### Development Commands

```bash
make test          # Run tests
make lint          # Linter + type check
make format        # Format code
```

### Kubernetes Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full deployment guide.

```bash
make deploy        # Deploy to k8s
make status        # Check status
make logs          # Tail logs
make scale N=10    # Scale workers
```

---

## Research Context

This codebase supports dissertation research on:

- **State Space Models for Multi-Agent Coordination** — How Mamba-style selective state dynamics can enable efficient swarm coordination
- **Hierarchical Consensus** — Triumvirate architecture for robust decision-making
- **Cyber-Physical Defense** — Paladin agents (defenders) vs Chaos agents (adversarial testing)
- **Neuro-Inspired Computing** — Principles from neuroscience applied to artificial swarms

### Key Publications (Target)

- NeurIPS / ICML: Theoretical foundations of SSM-based swarm coordination
- ICRA / IROS: Cyber-physical applications and robotic swarm implementation
- Nature Machine Intelligence (stretch): Unified theory of hierarchical swarm intelligence

---

## Development Roadmap

### Phase 1: Foundations ✓
- [x] Core agent implementation
- [x] Substrate (stigmergic layer)
- [x] Rhythm (temporal dynamics)
- [x] Single agent study
- [x] Pair dynamics study

### Phase 2: Emergence (Current)
- [ ] Triad consensus study
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

## Contributing

This is research code. Contributions are welcome, but please:

1. Read `PRINCIPLES.md` first
2. Prefer removal over addition
3. Write studies, not just code
4. Document your observations

---

## License

MIT License - See LICENSE file

---

## Acknowledgments

- FAU GenAI Lab
- Dr. Koch (advisor feedback)
- The starlings, ants, and neurons who taught us balance

---

*"We are not building a tool. We are cultivating a system."*
