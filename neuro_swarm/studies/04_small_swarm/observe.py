"""
Study 04: Small Swarm Observation

Run: python -m neuro_swarm.studies.04_small_swarm.observe

Watch 7±2 agents self-organize.
Simplicity generates complexity.
"""

import argparse
import numpy as np

from neuro_swarm.core.agent import AgentConfig
from neuro_swarm.environments.simple_field import SimpleField, FieldConfig
from neuro_swarm.observations.visualize import SwarmVisualizer


def run_study(
    n_agents: int = 7,
    steps: int = 1000,
    animate: bool = True,
    substrate: bool = True
):
    """
    Observe small swarm dynamics.

    This is where emergence happens.
    Simple local rules create complex global patterns.
    """
    print("=" * 50)
    print(f"Study 04: Small Swarm (n={n_agents})")
    print("=" * 50)
    print("\nSimplicity generates complexity.")
    print("-" * 50)

    # Create environment
    field = SimpleField(FieldConfig(
        bounds=(-40.0, 40.0),
        substrate_enabled=substrate
    ))

    # Create agents in random positions
    for i in range(n_agents):
        position = np.random.randn(2) * 15
        config = AgentConfig(
            energy_decay=0.03 + np.random.rand() * 0.02,
            energy_recovery=0.10 + np.random.rand() * 0.05,
        )
        agent = field.add_agent(f"agent_{i}", position=position, agent_config=config)
        agent.record_history = True

    print(f"Created {n_agents} agents")
    print(f"Substrate: {'enabled' if substrate else 'disabled'}")
    print(f"\nRunning {steps} steps...")

    # Metrics over time
    cohesion_history = []
    energy_history = []

    if animate:
        viz = SwarmVisualizer(field, trail_length=100)

        try:
            for step in range(steps):
                field.step()
                viz.record_frame()
                viz.render()

                # Track metrics
                positions = field.get_positions()
                centroid = positions.mean(axis=0)
                cohesion = np.linalg.norm(positions - centroid, axis=1).mean()
                cohesion_history.append(cohesion)
                energy_history.append(field.get_energies().mean())

                if step % 200 == 0:
                    print(f"  Step {step}: cohesion={cohesion:.2f}, "
                          f"mean_energy={energy_history[-1]:.2f}")

        finally:
            viz.close()
    else:
        for step in range(steps):
            field.step()

            positions = field.get_positions()
            centroid = positions.mean(axis=0)
            cohesion = np.linalg.norm(positions - centroid, axis=1).mean()
            cohesion_history.append(cohesion)
            energy_history.append(field.get_energies().mean())

            if step % 200 == 0:
                print(f"  Step {step}: cohesion={cohesion:.2f}")

    # Analysis
    print("\n" + "=" * 50)
    print("Observations")
    print("=" * 50)

    print(f"\nCohesion (distance from centroid):")
    print(f"  Initial: {cohesion_history[0]:.2f}")
    print(f"  Final: {cohesion_history[-1]:.2f}")
    print(f"  Min: {min(cohesion_history):.2f}")
    print(f"  Max: {max(cohesion_history):.2f}")

    print(f"\nEnergy:")
    print(f"  Mean over time: {np.mean(energy_history):.2f}")
    print(f"  Std over time: {np.std(energy_history):.2f}")

    if substrate and field.substrate:
        trace_mag = field.substrate.get_total_trace_magnitude()
        print(f"\nSubstrate trace magnitude: {trace_mag:.2f}")

    # Velocity alignment (collective motion indicator)
    velocities = field.get_velocities()
    if len(velocities) > 1:
        mean_velocity = velocities.mean(axis=0)
        alignment = np.dot(velocities, mean_velocity).mean()
        print(f"\nVelocity alignment: {alignment:.2f}")

    print("\n" + "=" * 50)
    print("Study complete. What patterns emerged?")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Small Swarm Study")
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--no-animate", action="store_true")
    parser.add_argument("--no-substrate", action="store_true")
    args = parser.parse_args()

    run_study(
        n_agents=args.agents,
        steps=args.steps,
        animate=not args.no_animate,
        substrate=not args.no_substrate
    )


if __name__ == "__main__":
    main()
