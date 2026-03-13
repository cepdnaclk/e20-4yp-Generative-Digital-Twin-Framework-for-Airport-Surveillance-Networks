# ============================================================
# Step 1: Aggregate per-camera flows → zone-level QoS
# With outlier rejection (IQR method within each zone)
# ============================================================

import json
import numpy as np
import glob
import os


def aggregate_flows_to_zones(scenario_data):
    """
    Aggregate 60 per-camera flows into 6 zone-level summaries.
    Uses IQR-based outlier rejection within each zone.
    """
    # Handle both formats
    if 'input' in scenario_data:
        inp = scenario_data['input']
        out = scenario_data['output']
    else:
        inp = scenario_data
        out = scenario_data

    flows = out['flows']
    zones_config = {z['zone_id']: z for z in inp['zones']}

    metrics = ['throughput_mbps', 'packet_loss_rate', 'avg_delay_ms', 'jitter_ms']

    zone_results = []

    for z in range(6):
        zone_flows = [f for f in flows if f.get('zone_id') == z and 'throughput_mbps' in f]

        if len(zone_flows) == 0:
            print(f"  WARNING: zone {z} has no valid flows!")
            continue

        zc = zones_config[z]
        zone_summary = {
            'zone_id': z,
            'processing_location': zc['processing_location'],
            'model_type': zc['model_type'],
            'bitrate_mbps': zc['bitrate_mbps'],
            'priority_class': zc['priority_class'],
            'num_cameras': len(zone_flows),
        }

        for metric in metrics:
            values = np.array([f[metric] for f in zone_flows])

            # IQR outlier rejection
            if len(values) > 4:
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (values >= lower) & (values <= upper)
                clean = values[mask]
                n_removed = len(values) - len(clean)
            else:
                clean = values
                n_removed = 0

            if len(clean) == 0:
                clean = values  # fallback to all if everything was "outlier"

            zone_summary[metric] = round(float(np.mean(clean)), 4)
            zone_summary[f'{metric}_std'] = round(float(np.std(clean)), 4)
            zone_summary[f'{metric}_min'] = round(float(np.min(clean)), 4)
            zone_summary[f'{metric}_max'] = round(float(np.max(clean)), 4)
            if n_removed > 0:
                zone_summary[f'{metric}_outliers_removed'] = n_removed

        # Extra derived metrics
        zone_summary['total_demand_mbps'] = round(zc['bitrate_mbps'] * len(zone_flows), 2)
        zone_summary['effective_throughput_ratio'] = round(
            zone_summary['throughput_mbps'] / zc['bitrate_mbps'], 4
        ) if zc['bitrate_mbps'] > 0 else 0.0

        # Packet stats
        zone_summary['total_tx_packets'] = sum(f.get('tx_packets', 0) for f in zone_flows)
        zone_summary['total_rx_packets'] = sum(f.get('rx_packets', 0) for f in zone_flows)
        zone_summary['total_lost_packets'] = sum(f.get('lost_packets', 0) for f in zone_flows)

        zone_results.append(zone_summary)

    return zone_results


def process_scenario(scenario_data):
    """Process a full scenario into a clean input + zone-level output."""
    if 'input' in scenario_data:
        inp = scenario_data['input']
    else:
        inp = scenario_data

    zone_summaries = aggregate_flows_to_zones(scenario_data)

    return {
        'scenario_id': inp.get('scenario_id', scenario_data.get('scenario_id', 'unknown')),
        'input': {
            'background_traffic_mbps': inp['background_traffic_mbps'],
            'simulation_time': inp.get('simulation_time', 10),
            'zones': sorted(inp['zones'], key=lambda z: z['zone_id']),
        },
        'output': {
            'zone_summaries': zone_summaries,
        }
    }


# ============================================================
# Process all scenarios
# ============================================================
if __name__ == '__main__':
    import sys

    input_dir = sys.argv[1] if len(sys.argv) > 1 else './dataset'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'zone_aggregated.json'

    json_files = sorted(glob.glob(os.path.join(input_dir, 'scenario_*.json')))
    print(f"Found {len(json_files)} scenario files in {input_dir}")

    all_scenarios = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        processed = process_scenario(data)
        all_scenarios.append(processed)

    with open(output_file, 'w') as f:
        json.dump(all_scenarios, f, indent=2)

    print(f"✅ Saved {len(all_scenarios)} scenarios to {output_file}")
    print(f"\nSample zone summary:")
    print(json.dumps(all_scenarios[0]['output']['zone_summaries'][0], indent=2))