"""
Study 03: Triad Consensus Observation

Run: python -m neuro_swarm.studies.03_triad_consensus.observe

Watch three agents form a triumvirate.
Consensus emerges; it is never imposed from above.
"""

import argparse
import numpy as np

from neuro_swarm.core.agent import AgentConfig
from neuro_swarm.environments.simple_field import SimpleField, FieldConfig
from neuro_swarm.observations.visualize import SwarmVisualizer


def run_study(
    steps: int = 500,
    animate: bool = True,
    formation: str = "triangle"
):
    """
    Observe triad dynamics and emergent consensus.
    """
    print("=" * 50)
    print("Study 03: Triad Consensus")
    print("=" * 50)
    print("\nThree perspectives can triangulate truth.")
    print("-" * 50)

    # Create environment
    field = SimpleField(FieldConfig(bounds=(-30.0, 30.0)))

    # Initial positions based on formation
    if formation == "triangle":
        positions = [
            np.array([0.0, 10.0]),    # Top
            np.array([-8.66, -5.0]),  # Bottom left
            np.array([8.66, -5.0]),   # Bottom right
        ]
    elif formation == "line":
        positions = [
            np.array([-15.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([15.0, 0.0]),
        ]
    else:  # random
        positions = [np.random.randn(2) * 10 for _ in range(3)]

    # Create the triumvirate
    roles = ["preserver", "challenger", "integrator"]
    configs = [
        AgentConfig(energy_decay=0.03, memory_persistence=0.95),  # Preserver
        AgentConfig(energy_decay=0.05, observation_weight=0.15),  # Challenger
        AgentConfig(energy_decay=0.04, memory_persistence=0.85),  # Integrator
    ]

    for i, (role, pos, cfg) in enumerate(zip(roles, positions, configs)):
        agent = field.add_agent(f"agent_{role}", position=pos, agent_config=cfg)
        agent.record_history = True
        print(f"Created {role}: {agent}")

    print(f"\nRunning {steps} steps...")

    if animate:
        viz = SwarmVisualizer(field, trail_length=150)

        try:
            for step in range(steps):
                field.step()
                viz.record_frame()
                viz.render()

                if step % 100 == 0:
                    centroid = field.get_positions().mean(axis=0)
                    print(f"  Step {step}: centroid=[{centroid[0]:.1f}, {centroid[1]:.1f}]")

        finally:
            viz.close()
    else:
        for step in range(steps):
            field.step()

    # Analysis
    print("\n" + "=" * 50)
    print("Observations")
    print("=" * 50)

    positions = field.get_positions()
    centroid = positions.mean(axis=0)
    print(f"\nFinal centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}]")

    # Measure "consensus" as compactness
    distances_to_centroid = np.linalg.norm(positions - centroid, axis=1)
    print(f"Spread (std from centroid): {distances_to_centroid.std():.2f}")

    # Energy states
    energies = field.get_energies()
    print(f"Energy distribution: {energies}")

    print("\n" + "=" * 50)
    print("Study complete. Did consensus emerge?")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Triad Consensus Study")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--formation", type=str, default="triangle",
                        choices=["triangle", "line", "random"])
    parser.add_argument("--no-animate", action="store_true")
    args = parser.parse_args()

    run_study(
        steps=args.steps,
        animate=not args.no_animate,
        formation=args.formation
    )


if __name__ == "__main__":
    main()
