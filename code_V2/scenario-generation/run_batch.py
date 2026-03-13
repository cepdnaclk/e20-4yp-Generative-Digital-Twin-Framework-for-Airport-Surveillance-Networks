"""
Generate randomized scenario JSON files for ns-3 simulation.
Produces scenarios that create congestion, packet loss, and varied QoS behavior.

Usage:
  python3 generate_scenarios.py --count 500
  python3 generate_scenarios.py --count 100 --output-dir ./scenarios
"""

import os
import json
import random
import argparse


def generate_one_scenario(scenario_id: str, seed: int) -> dict:
    """Generate a single randomized scenario."""
    rng = random.Random(seed)

    locations = ["edge1", "edge2", "edge3", "cloud"]
    model_types = ["lightweight", "medium", "heavy"]

    zones = []
    for z in range(6):
        zones.append({
            "zone_id": z,
            "bitrate_mbps": round(rng.uniform(8, 25), 1),
            "priority_class": rng.choice([1, 2, 3]),
            "model_type": rng.choice(model_types),
            "processing_location": rng.choice(locations)
        })

    # Background traffic: 0–200 Mbps
    # With 200 Mbps cloud link, even 100 Mbps bg traffic causes congestion
    bg_traffic = round(rng.uniform(0, 200), 1)

    sim_time = rng.choice([10, 10])

    return {
        "scenario_id": scenario_id,
        "simulation_time": sim_time,
        "background_traffic_mbps": bg_traffic,
        "zones": zones
    }


def generate_stress_scenario(scenario_id: str, seed: int) -> dict:
    """Generate a scenario designed to cause congestion."""
    rng = random.Random(seed)

    locations = ["edge1", "edge2", "edge3", "cloud"]
    model_types = ["lightweight", "medium", "heavy"]

    # Force many cameras to same destination → overload one link
    hot_location = rng.choice(locations)

    zones = []
    for z in range(6):
        # 60% chance to target the hot location
        if rng.random() < 0.6:
            loc = hot_location
        else:
            loc = rng.choice(locations)

        zones.append({
            "zone_id": z,
            "bitrate_mbps": round(rng.uniform(15, 25), 1),  # high bitrates
            "priority_class": rng.choice([1, 2, 3]),
            "model_type": rng.choice(model_types),
            "processing_location": loc
        })

    # Background traffic: 100–300 Mbps (will definitely congest cloud link)
    bg_traffic = round(rng.uniform(100, 300), 1)

    return {
        "scenario_id": scenario_id,
        "simulation_time": 10,
        "background_traffic_mbps": bg_traffic,
        "zones": zones
    }


def main():
    parser = argparse.ArgumentParser(description="Generate simulation scenarios")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of scenarios to generate")
    parser.add_argument("--output-dir", type=str, default="./scenarios",
                        help="Output directory")
    parser.add_argument("--stress-ratio", type=float, default=0.3,
                        help="Fraction of scenarios that are stress tests (0-1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_stress = int(args.count * args.stress_ratio)
    n_normal = args.count - n_stress

    print(f"[INFO] Generating {n_normal} normal + {n_stress} stress scenarios")

    for i in range(args.count):
        scenario_id = f"scenario_{i:04d}"
        seed = args.seed + i

        if i < n_normal:
            scenario = generate_one_scenario(scenario_id, seed)
        else:
            scenario = generate_stress_scenario(scenario_id, seed)

        filepath = os.path.join(args.output_dir, f"{scenario_id}.json")
        with open(filepath, "w") as f:
            json.dump(scenario, f, indent=2)

    print(f"[OK] {args.count} scenarios written to: {args.output_dir}")


if __name__ == "__main__":
    main()