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

# Using Make (preferred)
make install       # Install dependencies
make test          # Run tests
make lint          # Run linter + type check
make format        # Format code

# Run studies
make study-01      # Single agent
make study-02      # Pair dynamics
make study-03      # Triad consensus
make study-04      # Small swarm (7 agents)

# Or directly
python -m neuro_swarm.studies.01_single_agent.observe
python -m neuro_swarm.studies.02_pair_dynamics.observe --scenario symmetric
python -m neuro_swarm.studies.03_triad_consensus.observe
python -m neuro_swarm.studies.04_small_swarm.observe --agents 7

# Kubernetes deployment
make deploy        # Deploy to k8s
make status        # Check deployment status
make logs          # Tail controller logs
make scale N=10    # Scale workers

# Run tests
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
├── Makefile                  # Development and deployment commands
│
├── neuro_swarm/              # Main package
│   ├── __init__.py
│   │
│   ├── philosophy/           # Guiding principles
│   │   └── PRINCIPLES.md
│   │
│   ├── core/                 # Core components
│   │   ├── agent.py          # NeuroAgent - the fundamental unit
│   │   ├── agent_brain.py    # Brain-augmented agents with distilled models
│   │   ├── substrate.py      # Stigmergic communication layer
│   │   └── rhythm.py         # Temporal dynamics
│   │
│   ├── evolution/            # Evolutionary optimization
│   │   ├── algorithms.py     # ES, MAP-Elites, Novelty Search
│   │   ├── genome.py         # Genome representations
│   │   └── fitness.py        # Fitness functions & behavioral descriptors
│   │
│   ├── distillation/         # Model distillation pipeline
│   │   ├── distiller.py      # LoRA fine-tuning for agent models
│   │   ├── inference.py      # Fast inference for distilled brains
│   │   ├── quantizer.py      # INT4/INT8 quantization
│   │   ├── data_generator.py # Training data from teacher models
│   │   └── scripts/          # Training CLI tools
│   │       ├── train.py      # Full distillation pipeline CLI
│   │       └── train_mamba.py # Mamba SSM training for swarm units
│   │
│   ├── protocols/            # Interaction protocols
│   │   ├── attention.py      # Local neighborhood attention
│   │   ├── consensus.py      # Emergent agreement
│   │   └── tension.py        # Opposing forces (Paladin/Chaos)
│   │
│   ├── services/             # Distributed services
│   │   ├── queue.py          # Task queue (Redis/in-memory)
│   │   ├── controller.py     # Evolution controller
│   │   └── worker.py         # Evaluation worker
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
├── deploy/                   # Deployment configurations
│   └── k8s/                  # Kubernetes manifests
│       └── all-in-one.yaml   # Full deployment manifest
│
├── tests/                    # Test suite
│   ├── test_agent.py         # Core agent tests
│   ├── test_protocols.py     # Protocol tests
│   ├── test_evolution.py     # Evolution algorithm tests
│   └── test_services.py      # Distributed services tests
│
└── docs/                     # Documentation and assets
    ├── DEPLOYMENT.md         # Kubernetes deployment guide
    └── MODEL_DISTILLATION_ROADMAP.md  # Distillation pipeline roadmap
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

### Evolution Module (`evolution/`)

Distributed evolutionary optimization for swarm parameters.

**Algorithms** (`algorithms.py`):
- `EvolutionaryStrategy` — OpenAI ES, gradient-free optimization
- `MAPElites` — Quality-diversity, finds diverse high-quality solutions
- `NoveltySearch` — Rewards behavioral novelty, escapes local optima

**Genome** (`genome.py`):
- `AgentGenome` — Parameters for a single agent
- `SwarmGenome` — Parameters for entire swarm configuration

**Fitness** (`fitness.py`):
- `SwarmCoherenceFitness` — Rewards cohesion, alignment, efficiency
- `TaskCompletionFitness` — Rewards goal achievement

### Distillation Module (`distillation/`)

Model distillation pipeline for training specialized, locally-deployable agent brains.

**Tiers:**
- `Triumvirate` — Consensus/coordination models (~3-7B params)
- `Paladin` — Defensive/anomaly detection models (~1-3B params)
- `Chaos` — Adversarial/attack generation models (~1-3B params)
- `SwarmUnit` — Minimal Mamba SSM models (~100M-500M params)

**Components:**
- `AgentDistiller` — LoRA fine-tuning pipeline with QLoRA support
- `AgentInferenceEngine` — Fast inference for distilled brains
- `ModelQuantizer` — GPTQ/AWQ/BnB quantization for edge deployment
- `TaskDataGenerator` — Generate training data from teacher models (Claude API or local)

**Brain-Augmented Agents** (`core/agent_brain.py`):
- `BrainAugmentedAgent` — NeuroAgent with optional neural brain
- `TriumvirateAgent` — High-level coordinator with distilled reasoning
- `PaladinAgent` — Defensive agent with threat detection
- `ChaosAgent` — Adversarial agent for red team simulation
- `SwarmUnitAgent` — Fast Mamba-based swarm unit

```bash
# Train agent models
python -m neuro_swarm.distillation.scripts.train triumvirate --base-model Qwen/Qwen2.5-7B-Instruct
python -m neuro_swarm.distillation.scripts.train_mamba --epochs 100

# Generate training data
python -m neuro_swarm.distillation.scripts.train triumvirate --generate-data --teacher anthropic
```

### Protocols (`protocols/`)

Interaction protocols for swarm coordination.

**Attention** (`attention.py`):
- `LocalAttention` — Neighbor selection with attention weighting
  - `select_neighbors()` — Select most relevant neighbors (7±2)
  - `compute_attention_weights()` — Distance + alignment + energy weighted
  - `attend()` — Compute attended representation of neighbors

**Consensus** (`consensus.py`):
- `TriumvirateConsensus` — Three-agent consensus (2/3 agreement)
- `LocalConsensus` — Neighborhood-based consensus for larger swarms
- `RoleBasedConsensus` — Role-weighted consensus (Paladin/Explorer/Integrator)

**Tension** (`tension.py`):
- `LinearTension` — Weighted sum of forces
- `DynamicTension` — Context-dependent force resolution
  - Low energy → favor cohesion/preservation
  - High threat → favor separation/preservation
  - High energy + low threat → favor exploration
- `PaladinChaosBalance` — Preservation vs disruption dynamics

### Services (`services/`)

Distributed services for evolutionary optimization.

**Queue** (`queue.py`):
- `InMemoryTaskQueue` — Thread-safe queue for single-machine use
- `RedisTaskQueue` — Redis-backed queue for distributed execution
- `EvaluationTask` / `EvaluationResultMessage` — Task serialization

**Controller** (`controller.py`):
- `EvolutionController` — Manages evolution process
  - Generates candidate genomes
  - Distributes evaluation tasks
  - Collects results and updates algorithm
  - Supports ES, MAP-Elites, Novelty Search

**Worker** (`worker.py`):
- `EvaluationWorker` — Evaluates genomes from task queue
- `SwarmSimulator` — Runs swarm simulations for evaluation

```bash
# Run controller
python -m neuro_swarm.services.controller --algorithm es --redis-url redis://localhost:6379

# Run worker
python -m neuro_swarm.services.worker --redis-url redis://localhost:6379
```

---

## Development Roadmap

### Phase 1: Foundations ✓
- [x] Core agent implementation
- [x] Substrate (stigmergic layer)
- [x] Rhythm (temporal dynamics)
- [x] Single agent study
- [x] Pair dynamics study

### Phase 2: Emergence ✓
- [x] Attention protocols (LocalAttention)
- [x] Consensus mechanisms (Triumvirate, Local, Role-based)
- [x] Tension resolution (Dynamic, Paladin/Chaos)
- [x] Distributed services (Controller, Worker, Queue)
- [ ] Triad consensus study integration
- [ ] Small swarm (7±2 agents) study integration

### Phase 3: Scale & Distillation
- [ ] Large swarm experiments (50-100 agents)
- [ ] Hierarchical organization
- [ ] Learned SSM parameters
- [ ] Performance benchmarks
- [x] Model distillation pipeline (LoRA, quantization)
- [x] Brain-augmented agents
- [ ] Mamba SSM training for swarm units

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

## Kubernetes Environment

**Namespace: `neuro-swarm`**

All Kubernetes operations for this project MUST use the `neuro-swarm` namespace.

### Setup Commands

```bash
# Set namespace as default for current context
kubectl config set-context --current --namespace=neuro-swarm

# Verify namespace is active
kubectl config view --minify | grep namespace

# All kubectl commands should target this namespace
kubectl get pods -n neuro-swarm
kubectl apply -f <manifest> -n neuro-swarm
```

### Enforcement Rules

- **ALWAYS** specify `-n neuro-swarm` or ensure context is set before running kubectl commands
- **NEVER** deploy resources to `default` namespace
- **NEVER** deploy to production namespaces from this project context
- All manifests MUST include `namespace: neuro-swarm` in metadata

### Environment Variables

The VSCode workspace sets these automatically in integrated terminals:
```bash
KUBE_NAMESPACE=neuro-swarm
```

---

## Coding Standards

**Claude Code MUST operate as a senior developer and software architect would.**

### Code Quality

- **Consistency** — Follow existing patterns in the codebase; don't introduce new conventions without reason
- **Clarity** — Write self-documenting code; names should reveal intent
- **Simplicity** — Prefer the simplest solution that works; avoid premature abstraction
- **DRY** — Don't repeat yourself, but don't over-abstract either
- **SOLID** — Apply SOLID principles where appropriate for maintainability

### Security

- **Never hardcode secrets** — Use environment variables or secret management
- **Validate inputs** — Sanitize all external inputs (user, API, file)
- **Principle of least privilege** — Request only necessary permissions
- **Secure defaults** — Default to secure configurations
- **Avoid common vulnerabilities** — No SQL injection, XSS, command injection, path traversal

### Best Practices

- **Type hints** — Use Python type annotations for function signatures
- **Error handling** — Use specific exceptions; don't silently swallow errors
- **Logging** — Use structured logging at appropriate levels (debug, info, warning, error)
- **Testing** — Write tests for new functionality; maintain test coverage
- **Documentation** — Docstrings for public APIs; inline comments for complex logic only

### Code Hygiene

- **No dead code** — Remove unused imports, variables, and functions
- **No TODO accumulation** — Address TODOs or track them in issues
- **Consistent formatting** — Run `black` and `ruff` before committing
- **Small functions** — Functions should do one thing well
- **Meaningful commits** — Each commit should be a logical, atomic change

### Architecture Mindset

- **Think before coding** — Understand the problem and design before implementation
- **Consider scale** — Will this work with 100 agents? 1000? 10000?
- **Design for change** — Interfaces over implementations; dependency injection
- **Performance awareness** — Profile before optimizing; optimize hot paths
- **Backward compatibility** — Don't break existing APIs without migration path

---

## Commit and Documentation Requirements

### Mandatory Commit Practices

**EVERY code change MUST be accompanied by a commit.** Do not leave uncommitted work.

```bash
# After ANY code modification:
git add <changed-files>
git commit -m "type: concise description of change"
```

#### Commit Message Format

```
type: subject

[optional body]
[optional footer]
```

**Types:**
- `feat:` — New feature or capability
- `fix:` — Bug fix
- `refactor:` — Code restructuring without behavior change
- `docs:` — Documentation only
- `test:` — Adding or modifying tests
- `study:` — New study or observation script
- `chore:` — Maintenance tasks (deps, config, etc.)

#### Examples

```bash
git commit -m "feat: add phase coupling to rhythm module"
git commit -m "fix: correct energy decay calculation in agent update"
git commit -m "study: implement triad consensus observation"
git commit -m "docs: update CLAUDE.md with k8s namespace rules"
```

### Documentation Requirements

**EVERY significant change MUST update relevant documentation.**

#### What Requires Documentation Updates

| Change Type | Required Documentation |
|-------------|----------------------|
| New module/file | Update project structure in CLAUDE.md and README.md |
| New study | Add to studies list, update roadmap |
| New protocol | Document in protocols section, link to principles |
| API changes | Update Key Components section |
| New dependencies | Update pyproject.toml and setup instructions |
| Configuration changes | Update Build & Development Commands |
| New constants | Add to Constants and Constraints section |

#### Documentation Locations

- **CLAUDE.md** — Primary reference for Claude Code, detailed technical guidance
- **README.md** — User-facing overview, installation, quick start
- **PRINCIPLES.md** — Philosophy (rarely changes)
- **Study `__init__.py`** — Study-specific documentation and research questions

### Pre-Commit Checklist

Before completing any task, verify:

1. [ ] All code changes are committed
2. [ ] Commit message follows format
3. [ ] CLAUDE.md updated if structure/API changed
4. [ ] README.md updated if user-facing behavior changed
5. [ ] Tests pass: `pytest tests/ -v`
6. [ ] Type check passes: `mypy neuro_swarm/`
7. [ ] Code formatted: `black neuro_swarm/ && ruff check neuro_swarm/`

### Enforcement

**Claude Code MUST:**
- **Commit automatically without asking** — never prompt for permission to commit
- **Commit frequently and often** — small, atomic commits are preferred over large batches
- **Push and merge automatically** — after commits, push to remote and merge PRs when ready
- Commit changes immediately after implementing them
- Update documentation inline with code changes (not as separate follow-up)
- Never leave a session with uncommitted code
- Include documentation updates in the same commit as related code changes when feasible

---

*"We are not building a tool. We are cultivating a system."*
