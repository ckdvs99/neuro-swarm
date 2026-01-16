"""
Study 01: Single Agent Observation

Run: python -m neuro_swarm.studies.01_single_agent.observe

Watch a single agent navigate the world.
No hypotheses yet—just observation.
"""

import argparse
import numpy as np

from neuro_swarm.core.agent import NeuroAgent, AgentConfig
from neuro_swarm.environments.simple_field import SimpleField, FieldConfig
from neuro_swarm.observations.visualize import SwarmVisualizer


def run_study(
    steps: int = 500,
    animate: bool = True,
    record_history: bool = True
):
    """
    Observe a single agent.

    Watch:
    - Energy cycles (action/rest rhythm)
    - State evolution
    - Substrate interaction
    - Trajectory patterns
    """
    print("=" * 50)
    print("Study 01: Single Agent Observation")
    print("=" * 50)
    print("\nPrinciple: Understand one before many")
    print("-" * 50)

    # Create environment
    field_config = FieldConfig(
        bounds=(-25.0, 25.0),
        wrap_edges=True,
        substrate_enabled=True
    )
    field = SimpleField(field_config)

    # Create agent with slightly randomized config
    agent_config = AgentConfig(
        state_dim=8,
        neighbor_limit=7,
        energy_decay=0.03,
        energy_recovery=0.12,
        rest_threshold=0.15
    )

    agent = field.add_agent(
        "agent_0",
        position=np.array([0.0, 0.0]),
        agent_config=agent_config
    )
    agent.record_history = record_history

    print(f"\nAgent created: {agent}")
    print(f"Config: state_dim={agent_config.state_dim}, "
          f"energy_decay={agent_config.energy_decay}")
    print(f"\nRunning {steps} steps...")

    if animate:
        viz = SwarmVisualizer(field, trail_length=100)

        try:
            for step in range(steps):
                field.step()
                viz.record_frame()
                viz.render()

                # Periodic status
                if step % 100 == 0:
                    print(f"  Step {step}: energy={agent.state.energy:.2f}, "
                          f"age={agent.state.age}")

        finally:
            viz.close()
    else:
        for step in range(steps):
            field.step()

            if step % 100 == 0:
                print(f"  Step {step}: energy={agent.state.energy:.2f}, "
                      f"age={agent.state.age}")

    # Analysis
    print("\n" + "=" * 50)
    print("Observations")
    print("=" * 50)

    if record_history and agent.history:
        energies = [s.energy for s in agent.history]
        positions = np.array([s.position for s in agent.history])

        print(f"\nEnergy range: [{min(energies):.2f}, {max(energies):.2f}]")
        print(f"Mean energy: {np.mean(energies):.2f}")

        # Count rest periods
        rest_periods = sum(1 for e in energies if e < agent_config.rest_threshold)
        print(f"Rest periods: {rest_periods} / {len(energies)} "
              f"({100*rest_periods/len(energies):.1f}%)")

        # Travel distance
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = distances.sum()
        print(f"Total distance traveled: {total_distance:.2f}")

        # Substrate activity
        if field.substrate:
            trace_mag = field.substrate.get_total_trace_magnitude()
            print(f"Substrate trace magnitude: {trace_mag:.2f}")

    print("\n" + "=" * 50)
    print("Study complete. What did you observe?")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Single Agent Observation Study")
    parser.add_argument("--steps", type=int, default=500, help="Simulation steps")
    parser.add_argument("--no-animate", action="store_true", help="Disable animation")
    args = parser.parse_args()

    run_study(
        steps=args.steps,
        animate=not args.no_animate
    )


if __name__ == "__main__":
    main()
