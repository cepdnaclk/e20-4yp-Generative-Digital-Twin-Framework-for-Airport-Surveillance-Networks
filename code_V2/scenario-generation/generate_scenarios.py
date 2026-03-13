"""
Generate randomized scenario JSON files for ns-3 simulation.
Produces scenarios that create congestion, packet loss, and varied QoS behavior.

Topology link capacities (must match simulation.cc):
  Camera → Switch:   30 Mbps per camera
  Switch → Core:    100 Mbps per zone (10 cams share this)
  Core → Edge:      150 Mbps per edge server
  Core → Cloud:      80 Mbps (shared)

To get interesting data without 100% loss:
  - Per-zone bitrate should range 5-15 Mbps (50-150 Mbps aggregate)
  - Background traffic 0-100 Mbps (cloud link is 80 Mbps)
  - Stress scenarios push to 10-18 Mbps with 50-150 Mbps background

Usage:
  python3 generate_scenarios.py --count 500
  python3 generate_scenarios.py --count 100 --output-dir ./scenarios
"""

import os
import json
import random
import argparse


def generate_one_scenario(scenario_id: str, seed: int) -> dict:
    """Generate a single randomized scenario with moderate load."""
    rng = random.Random(seed)

    locations = ["edge1", "edge2", "edge3", "cloud"]
    model_types = ["lightweight", "medium", "heavy"]

    zones = []
    for z in range(6):
        zones.append({
            "zone_id": z,
            "bitrate_mbps": round(rng.uniform(5, 15), 1),
            "priority_class": rng.choice([1, 2, 3]),
            "model_type": rng.choice(model_types),
            "processing_location": rng.choice(locations)
        })

    # Background traffic: 0–80 Mbps
    # Cloud link = 80 Mbps, so even 40 Mbps bg + 1 zone = congestion
    bg_traffic = round(rng.uniform(0, 80), 1)

    sim_time = 10

    return {
        "scenario_id": scenario_id,
        "simulation_time": sim_time,
        "background_traffic_mbps": bg_traffic,
        "zones": zones
    }


def generate_stress_scenario(scenario_id: str, seed: int) -> dict:
    """Generate a scenario designed to cause congestion without total blackout."""
    rng = random.Random(seed)

    locations = ["edge1", "edge2", "edge3", "cloud"]
    model_types = ["lightweight", "medium", "heavy"]

    # Force many cameras to same destination → overload one link
    hot_location = rng.choice(locations)

    zones = []
    for z in range(6):
        # 50% chance to target the hot location
        if rng.random() < 0.5:
            loc = hot_location
        else:
            loc = rng.choice(locations)

        zones.append({
            "zone_id": z,
            "bitrate_mbps": round(rng.uniform(8, 18), 1),
            "priority_class": rng.choice([1, 2, 3]),
            "model_type": rng.choice(model_types),
            "processing_location": loc
        })

    # Background traffic: 30–120 Mbps
    bg_traffic = round(rng.uniform(30, 120), 1)

    return {
        "scenario_id": scenario_id,
        "simulation_time": 10,
        "background_traffic_mbps": bg_traffic,
        "zones": zones
    }


def generate_light_scenario(scenario_id: str, seed: int) -> dict:
    """Generate a low-load scenario — should have minimal loss."""
    rng = random.Random(seed)

    locations = ["edge1", "edge2", "edge3", "cloud"]
    model_types = ["lightweight", "medium", "heavy"]

    zones = []
    for z in range(6):
        zones.append({
            "zone_id": z,
            "bitrate_mbps": round(rng.uniform(5, 9), 1),
            "priority_class": rng.choice([1, 2, 3]),
            "model_type": rng.choice(model_types),
            "processing_location": rng.choice(locations)
        })

    bg_traffic = round(rng.uniform(0, 20), 1)

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
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Distribution: 20% light, 50% normal, 30% stress
    n_light  = int(args.count * 0.20)
    n_normal = int(args.count * 0.50)
    n_stress = args.count - n_light - n_normal

    print(f"[INFO] Generating {n_light} light + {n_normal} normal + {n_stress} stress scenarios")

    idx = 0
    for i in range(n_light):
        scenario_id = f"scenario_{idx:04d}"
        scenario = generate_light_scenario(scenario_id, args.seed + idx)
        filepath = os.path.join(args.output_dir, f"{scenario_id}.json")
        with open(filepath, "w") as f:
            json.dump(scenario, f, indent=2)
        idx += 1

    for i in range(n_normal):
        scenario_id = f"scenario_{idx:04d}"
        scenario = generate_one_scenario(scenario_id, args.seed + idx)
        filepath = os.path.join(args.output_dir, f"{scenario_id}.json")
        with open(filepath, "w") as f:
            json.dump(scenario, f, indent=2)
        idx += 1

    for i in range(n_stress):
        scenario_id = f"scenario_{idx:04d}"
        scenario = generate_stress_scenario(scenario_id, args.seed + idx)
        filepath = os.path.join(args.output_dir, f"{scenario_id}.json")
        with open(filepath, "w") as f:
            json.dump(scenario, f, indent=2)
        idx += 1

    print(f"[OK] {args.count} scenarios written to: {args.output_dir}")


if __name__ == "__main__":
    main()