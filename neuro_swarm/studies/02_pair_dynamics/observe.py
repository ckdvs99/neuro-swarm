"""
Study 02: Pair Dynamics Observation

Run: python -m neuro_swarm.studies.02_pair_dynamics.observe --scenario symmetric

Watch two agents interact.
Interaction emerges from individual behavior.
"""

import argparse
import numpy as np

from neuro_swarm.core.agent import AgentConfig
from neuro_swarm.environments.simple_field import SimpleField, FieldConfig
from neuro_swarm.observations.visualize import SwarmVisualizer


SCENARIOS = {
    "symmetric": {
        "description": "Two identical agents starting nearby",
        "agent_0": {"position": [-5.0, 0.0], "config": AgentConfig()},
        "agent_1": {"position": [5.0, 0.0], "config": AgentConfig()},
    },
    "pursuer_evader": {
        "description": "One high-energy pursuer, one conservative evader",
        "agent_0": {
            "position": [-10.0, 0.0],
            "config": AgentConfig(energy_decay=0.02, energy_recovery=0.2)
        },
        "agent_1": {
            "position": [10.0, 0.0],
            "config": AgentConfig(energy_decay=0.08, energy_recovery=0.1)
        },
    },
    "opposite_goals": {
        "description": "Agents with opposing internal biases",
        "agent_0": {"position": [0.0, -5.0], "config": AgentConfig()},
        "agent_1": {"position": [0.0, 5.0], "config": AgentConfig()},
    },
}


def run_study(
    scenario: str = "symmetric",
    steps: int = 500,
    animate: bool = True
):
    """
    Observe pair dynamics under different scenarios.
    """
    if scenario not in SCENARIOS:
        print(f"Unknown scenario: {scenario}")
        print(f"Available: {list(SCENARIOS.keys())}")
        return

    scenario_config = SCENARIOS[scenario]

    print("=" * 50)
    print(f"Study 02: Pair Dynamics - {scenario}")
    print("=" * 50)
    print(f"\n{scenario_config['description']}")
    print("-" * 50)

    # Create environment
    field = SimpleField(FieldConfig(bounds=(-30.0, 30.0)))

    # Create agents
    for agent_id in ["agent_0", "agent_1"]:
        cfg = scenario_config[agent_id]
        agent = field.add_agent(
            agent_id,
            position=np.array(cfg["position"]),
            agent_config=cfg["config"]
        )
        agent.record_history = True
        print(f"Created {agent}")

    print(f"\nRunning {steps} steps...")

    if animate:
        viz = SwarmVisualizer(field, trail_length=150)

        try:
            for step in range(steps):
                field.step()
                viz.record_frame()
                viz.render()

                if step % 100 == 0:
                    a0 = field.agents["agent_0"]
                    a1 = field.agents["agent_1"]
                    dist = a0.distance_to(a1)
                    print(f"  Step {step}: distance={dist:.2f}")

        finally:
            viz.close()
    else:
        for step in range(steps):
            field.step()

    # Analysis
    print("\n" + "=" * 50)
    print("Observations")
    print("=" * 50)

    a0 = field.agents["agent_0"]
    a1 = field.agents["agent_1"]

    final_distance = a0.distance_to(a1)
    print(f"\nFinal distance: {final_distance:.2f}")

    if a0.history and a1.history:
        # Track distance over time
        distances = []
        for s0, s1 in zip(a0.history, a1.history):
            d = np.linalg.norm(s0.position - s1.position)
            distances.append(d)

        print(f"Distance range: [{min(distances):.2f}, {max(distances):.2f}]")
        print(f"Mean distance: {np.mean(distances):.2f}")

        # Correlation of movements
        v0 = np.array([s.velocity for s in a0.history])
        v1 = np.array([s.velocity for s in a1.history])
        if len(v0) > 1 and len(v1) > 1:
            correlation = np.corrcoef(v0.flatten(), v1.flatten())[0, 1]
            print(f"Velocity correlation: {correlation:.2f}")

    print("\n" + "=" * 50)
    print("Study complete. What patterns emerged?")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Pair Dynamics Study")
    parser.add_argument("--scenario", type=str, default="symmetric",
                        choices=list(SCENARIOS.keys()))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--no-animate", action="store_true")
    args = parser.parse_args()

    run_study(
        scenario=args.scenario,
        steps=args.steps,
        animate=not args.no_animate
    )


if __name__ == "__main__":
    main()
