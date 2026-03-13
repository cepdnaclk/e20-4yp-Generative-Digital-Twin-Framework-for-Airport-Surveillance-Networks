# ============================================================
# Step 2: Convert aggregated scenarios → Semantic Knowledge Base
# Creates natural language documents for RAG retrieval
# ============================================================

import json
import numpy as np
import glob
import os


# ============================================================
# Zone Aggregation (was in aggregate_zones.py)
# ============================================================

def aggregate_flows_to_zones(scenario_data):
    """Aggregate 60 per-camera flows into 6 zone-level summaries with IQR outlier rejection."""
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
            if len(values) > 4:
                q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
                iqr = q3 - q1
                mask = (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)
                clean = values[mask]
                n_removed = len(values) - len(clean)
            else:
                clean = values
                n_removed = 0

            if len(clean) == 0:
                clean = values

            zone_summary[metric] = round(float(np.mean(clean)), 4)
            zone_summary[f'{metric}_std'] = round(float(np.std(clean)), 4)
            zone_summary[f'{metric}_min'] = round(float(np.min(clean)), 4)
            zone_summary[f'{metric}_max'] = round(float(np.max(clean)), 4)
            if n_removed > 0:
                zone_summary[f'{metric}_outliers_removed'] = n_removed

        zone_summary['total_demand_mbps'] = round(zc['bitrate_mbps'] * len(zone_flows), 2)
        zone_summary['effective_throughput_ratio'] = round(
            zone_summary['throughput_mbps'] / zc['bitrate_mbps'], 4
        ) if zc['bitrate_mbps'] > 0 else 0.0
        zone_summary['total_tx_packets'] = sum(f.get('tx_packets', 0) for f in zone_flows)
        zone_summary['total_rx_packets'] = sum(f.get('rx_packets', 0) for f in zone_flows)
        zone_summary['total_lost_packets'] = sum(f.get('lost_packets', 0) for f in zone_flows)

        zone_results.append(zone_summary)
    return zone_results


def process_scenario(scenario_data):
    """Process a full scenario into clean input + zone-level output."""
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
# Semantic Descriptions
# ============================================================

def describe_congestion_level(loss_rate):
    if loss_rate == 0:
        return "no congestion"
    elif loss_rate < 0.05:
        return "minimal congestion"
    elif loss_rate < 0.15:
        return "moderate congestion"
    elif loss_rate < 0.35:
        return "significant congestion"
    elif loss_rate < 0.5:
        return "heavy congestion"
    else:
        return "severe congestion"


def describe_delay(delay_ms, proc_location):
    baseline = 31 if proc_location == 'cloud' else 8  # rough baseline
    if delay_ms < baseline * 1.1:
        return "optimal latency"
    elif delay_ms < baseline * 1.5:
        return "acceptable latency"
    elif delay_ms < baseline * 2:
        return "elevated latency"
    else:
        return "high latency"


def describe_throughput_efficiency(actual, configured):
    ratio = actual / configured if configured > 0 else 0
    if ratio > 0.95:
        return "full throughput achieved"
    elif ratio > 0.8:
        return "minor throughput degradation"
    elif ratio > 0.5:
        return "significant throughput loss"
    else:
        return "severe throughput loss"


def describe_jitter(jitter_ms):
    if jitter_ms < 0.01:
        return "negligible jitter"
    elif jitter_ms < 0.05:
        return "low jitter"
    elif jitter_ms < 0.2:
        return "moderate jitter"
    else:
        return "high jitter"


def describe_priority(priority_class):
    return {1: "high priority", 2: "medium priority", 3: "low priority"}.get(
        priority_class, f"priority {priority_class}")


def describe_model(model_type):
    return {
        'lightweight': "lightweight model (5ms processing, low compute)",
        'medium': "medium model (10ms processing, moderate compute)",
        'heavy': "heavy model (20ms processing, high compute)",
    }.get(model_type, model_type)


# ============================================================
# Document Generators
# ============================================================

def generate_scenario_document(scenario):
    """Generate a natural language document describing the full scenario."""
    inp = scenario['input']
    zones = scenario['output']['zone_summaries']
    bg = inp['background_traffic_mbps']

    # Compute overall stats
    total_demand = sum(z['total_demand_mbps'] for z in zones)
    avg_loss = np.mean([z['packet_loss_rate'] for z in zones])
    avg_throughput = np.mean([z['throughput_mbps'] for z in zones])
    cloud_zones = [z for z in zones if z['processing_location'] == 'cloud']
    edge_zones = [z for z in zones if z['processing_location'] != 'cloud']

    # Identify bottlenecks
    congested_zones = [z for z in zones if z['packet_loss_rate'] > 0.05]
    high_delay_zones = [z for z in zones if z['avg_delay_ms'] > 40]

    lines = []
    lines.append(f"# Scenario: {scenario['scenario_id']}")
    lines.append(f"")
    lines.append(f"## Network Configuration")
    lines.append(f"- Background traffic: {bg} Mbps")
    lines.append(f"- Total camera demand: {total_demand:.1f} Mbps across 6 zones (60 cameras)")
    lines.append(f"- Cloud-processed zones: {len(cloud_zones)}, Edge-processed zones: {len(edge_zones)}")
    lines.append(f"")

    # Zone details
    lines.append(f"## Zone Configurations and Results")
    for z in zones:
        zc = inp['zones'][z['zone_id']]
        congestion = describe_congestion_level(z['packet_loss_rate'])
        delay_desc = describe_delay(z['avg_delay_ms'], z['processing_location'])
        tp_desc = describe_throughput_efficiency(z['throughput_mbps'], z['bitrate_mbps'])
        jitter_desc = describe_jitter(z['jitter_ms'])
        priority_desc = describe_priority(z['priority_class'])
        model_desc = describe_model(z['model_type'])

        lines.append(f"### Zone {z['zone_id']} → {z['processing_location']}")
        lines.append(f"- Configuration: {z['bitrate_mbps']} Mbps per camera, {priority_desc}, {model_desc}")
        lines.append(f"- Total zone demand: {z['total_demand_mbps']} Mbps")
        lines.append(f"- Throughput: {z['throughput_mbps']:.3f} Mbps ({tp_desc})")
        lines.append(f"- Packet loss: {z['packet_loss_rate']*100:.2f}% ({congestion})")
        lines.append(f"- Delay: {z['avg_delay_ms']:.2f} ms ({delay_desc})")
        lines.append(f"- Jitter: {z['jitter_ms']:.4f} ms ({jitter_desc})")
        lines.append(f"- Packets: {z['total_tx_packets']} sent, {z['total_rx_packets']} received, {z['total_lost_packets']} lost")
        lines.append(f"")

    # Key observations
    lines.append(f"## Key Observations")

    if len(congested_zones) == 0:
        lines.append(f"- No zones experienced significant packet loss. Network capacity is sufficient.")
    else:
        for z in congested_zones:
            lines.append(f"- Zone {z['zone_id']} ({z['processing_location']}) experienced {describe_congestion_level(z['packet_loss_rate'])} "
                        f"with {z['packet_loss_rate']*100:.1f}% packet loss.")

    if len(cloud_zones) > 0:
        cloud_loss = np.mean([z['packet_loss_rate'] for z in cloud_zones])
        cloud_delay = np.mean([z['avg_delay_ms'] for z in cloud_zones])
        lines.append(f"- Cloud-processed zones: avg loss={cloud_loss*100:.1f}%, avg delay={cloud_delay:.1f}ms")

    if len(edge_zones) > 0:
        edge_loss = np.mean([z['packet_loss_rate'] for z in edge_zones])
        edge_delay = np.mean([z['avg_delay_ms'] for z in edge_zones])
        lines.append(f"- Edge-processed zones: avg loss={edge_loss*100:.1f}%, avg delay={edge_delay:.1f}ms")

    if bg > 10:
        lines.append(f"- High background traffic ({bg} Mbps) may impact cloud link performance.")
    elif bg < 3:
        lines.append(f"- Low background traffic ({bg} Mbps), minimal impact on network.")

    # Capacity analysis
    cloud_demand = sum(z['total_demand_mbps'] for z in cloud_zones)
    if cloud_demand > 0:
        cloud_link = 80.0  # Mbps
        cloud_util = (cloud_demand + bg) / cloud_link
        lines.append(f"- Cloud link utilization: {cloud_util*100:.1f}% "
                     f"({cloud_demand:.0f} Mbps camera + {bg} Mbps background on 80 Mbps link)")
        if cloud_util > 1.0:
            lines.append(f"  ⚠️ Cloud link is OVERLOADED. Packet loss is expected.")

    # Edge server load distribution
    edge_loads = {}
    for z in zones:
        loc = z['processing_location']
        if loc != 'cloud':
            edge_loads.setdefault(loc, 0)
            edge_loads[loc] += z['total_demand_mbps']

    for server, load in edge_loads.items():
        util = (load + bg) / 150.0  # edge link = 150 Mbps
        lines.append(f"- {server} utilization: {util*100:.1f}% ({load:.0f} Mbps camera + {bg} Mbps bg on 150 Mbps)")

    return '\n'.join(lines)


def generate_zone_document(scenario, zone_summary):
    """Generate a focused document about a single zone's performance."""
    inp = scenario['input']
    bg = inp['background_traffic_mbps']
    z = zone_summary

    congestion = describe_congestion_level(z['packet_loss_rate'])
    tp_desc = describe_throughput_efficiency(z['throughput_mbps'], z['bitrate_mbps'])

    doc = (
        f"Zone {z['zone_id']} in {scenario['scenario_id']} is processed at {z['processing_location']} "
        f"using {z['model_type']} model with {z['bitrate_mbps']} Mbps per camera "
        f"({describe_priority(z['priority_class'])}). "
        f"Background traffic is {bg} Mbps. "
        f"Result: throughput={z['throughput_mbps']:.3f} Mbps ({tp_desc}), "
        f"packet loss={z['packet_loss_rate']*100:.2f}% ({congestion}), "
        f"delay={z['avg_delay_ms']:.2f}ms, jitter={z['jitter_ms']:.4f}ms. "
        f"Total zone demand is {z['total_demand_mbps']} Mbps. "
        f"{z['total_lost_packets']} packets were lost out of {z['total_tx_packets']} sent."
    )
    return doc


def generate_comparison_insights(all_scenarios):
    """Generate cross-scenario pattern documents."""
    insights = []

    # Group by processing location patterns
    cloud_scenarios = []
    edge_scenarios = []
    mixed_scenarios = []

    for s in all_scenarios:
        zones = s['output']['zone_summaries']
        cloud_count = sum(1 for z in zones if z['processing_location'] == 'cloud')
        if cloud_count == 0:
            edge_scenarios.append(s)
        elif cloud_count == 6:
            cloud_scenarios.append(s)
        else:
            mixed_scenarios.append(s)

    # Pattern: Cloud congestion
    cloud_zone_stats = []
    for s in all_scenarios:
        for z in s['output']['zone_summaries']:
            if z['processing_location'] == 'cloud':
                cloud_zone_stats.append({
                    'loss': z['packet_loss_rate'],
                    'delay': z['avg_delay_ms'],
                    'throughput': z['throughput_mbps'],
                    'bitrate': z['bitrate_mbps'],
                    'bg': s['input']['background_traffic_mbps'],
                    'scenario': s['scenario_id'],
                })

    if cloud_zone_stats:
        avg_loss = np.mean([c['loss'] for c in cloud_zone_stats])
        avg_delay = np.mean([c['delay'] for c in cloud_zone_stats])

        high_loss = [c for c in cloud_zone_stats if c['loss'] > 0.3]
        low_loss = [c for c in cloud_zone_stats if c['loss'] < 0.05]

        doc = (
            f"# Cloud Processing Pattern Analysis\n\n"
            f"Across {len(cloud_zone_stats)} cloud-processed zones from {len(all_scenarios)} scenarios:\n"
            f"- Average packet loss: {avg_loss*100:.1f}%\n"
            f"- Average delay: {avg_delay:.1f}ms\n"
            f"- {len(high_loss)} zones had >30% loss, {len(low_loss)} zones had <5% loss\n\n"
        )

        if high_loss:
            avg_bg_high = np.mean([c['bg'] for c in high_loss])
            avg_br_high = np.mean([c['bitrate'] for c in high_loss])
            doc += (
                f"Zones with high loss (>30%) typically had:\n"
                f"- Average background traffic: {avg_bg_high:.1f} Mbps\n"
                f"- Average bitrate: {avg_br_high:.1f} Mbps per camera\n"
            )

        if low_loss:
            avg_bg_low = np.mean([c['bg'] for c in low_loss])
            avg_br_low = np.mean([c['bitrate'] for c in low_loss])
            doc += (
                f"\nZones with low loss (<5%) typically had:\n"
                f"- Average background traffic: {avg_bg_low:.1f} Mbps\n"
                f"- Average bitrate: {avg_br_low:.1f} Mbps per camera\n"
            )

        insights.append({
            'type': 'pattern_analysis',
            'topic': 'cloud_processing',
            'content': doc,
        })

    # Pattern: Edge processing
    edge_zone_stats = []
    for s in all_scenarios:
        for z in s['output']['zone_summaries']:
            if z['processing_location'] != 'cloud':
                edge_zone_stats.append({
                    'loss': z['packet_loss_rate'],
                    'delay': z['avg_delay_ms'],
                    'throughput': z['throughput_mbps'],
                    'location': z['processing_location'],
                    'bg': s['input']['background_traffic_mbps'],
                })

    if edge_zone_stats:
        avg_loss = np.mean([e['loss'] for e in edge_zone_stats])
        avg_delay = np.mean([e['delay'] for e in edge_zone_stats])

        doc = (
            f"# Edge Processing Pattern Analysis\n\n"
            f"Across {len(edge_zone_stats)} edge-processed zones:\n"
            f"- Average packet loss: {avg_loss*100:.1f}%\n"
            f"- Average delay: {avg_delay:.1f}ms\n"
            f"- Edge processing generally provides lower delay and loss than cloud\n"
            f"- Edge link capacity (150 Mbps) is nearly 2x cloud link (80 Mbps)\n"
        )

        for loc in ['edge1', 'edge2', 'edge3']:
            loc_stats = [e for e in edge_zone_stats if e['location'] == loc]
            if loc_stats:
                doc += (
                    f"\n{loc}: {len(loc_stats)} zones, "
                    f"avg loss={np.mean([e['loss'] for e in loc_stats])*100:.1f}%, "
                    f"avg delay={np.mean([e['delay'] for e in loc_stats]):.1f}ms\n"
                )

        insights.append({
            'type': 'pattern_analysis',
            'topic': 'edge_processing',
            'content': doc,
        })

    # Pattern: Background traffic impact
    bg_low = [s for s in all_scenarios if s['input']['background_traffic_mbps'] < 5]
    bg_high = [s for s in all_scenarios if s['input']['background_traffic_mbps'] > 12]

    if bg_low and bg_high:
        loss_low = np.mean([z['packet_loss_rate']
                           for s in bg_low for z in s['output']['zone_summaries']])
        loss_high = np.mean([z['packet_loss_rate']
                            for s in bg_high for z in s['output']['zone_summaries']])
        delay_low = np.mean([z['avg_delay_ms']
                            for s in bg_low for z in s['output']['zone_summaries']])
        delay_high = np.mean([z['avg_delay_ms']
                             for s in bg_high for z in s['output']['zone_summaries']])

        doc = (
            f"# Background Traffic Impact Analysis\n\n"
            f"Low background (<5 Mbps): {len(bg_low)} scenarios\n"
            f"- Average loss: {loss_low*100:.1f}%, Average delay: {delay_low:.1f}ms\n\n"
            f"High background (>12 Mbps): {len(bg_high)} scenarios\n"
            f"- Average loss: {loss_high*100:.1f}%, Average delay: {delay_high:.1f}ms\n\n"
            f"Higher background traffic increases contention on shared links, "
            f"especially the cloud link (80 Mbps) which is the primary bottleneck.\n"
        )

        insights.append({
            'type': 'pattern_analysis',
            'topic': 'background_traffic',
            'content': doc,
        })

    # Network topology knowledge
    topology_doc = (
        "# Airport Network Topology\n\n"
        "The airport surveillance network consists of:\n"
        "- 60 cameras across 6 zones (10 cameras per zone)\n"
        "- 6 zone switches (one per zone)\n"
        "- 1 core router\n"
        "- 3 edge servers (edge1, edge2, edge3)\n"
        "- 1 cloud server\n\n"
        "## Link Capacities\n"
        "- Camera → Zone Switch: 30 Mbps, 1ms delay\n"
        "- Zone Switch → Core: 100 Mbps, 2-5ms delay (varies by zone)\n"
        "- Core → Edge Server: 150 Mbps, 5ms delay\n"
        "- Core → Cloud: 80 Mbps, 25ms delay\n\n"
        "## Key Bottleneck\n"
        "The cloud link (80 Mbps) is the primary bottleneck. When multiple zones "
        "send traffic to cloud AND background traffic is present, the combined demand "
        "often exceeds 80 Mbps, causing packet drops via DropTail queuing.\n\n"
        "## Processing Models\n"
        "- Lightweight: 5ms processing delay, low compute\n"
        "- Medium: 10ms processing delay, moderate compute\n"
        "- Heavy: 20ms processing delay, high compute (best accuracy)\n\n"
        "## Priority Classes\n"
        "- Priority 1 (high): Critical surveillance feeds\n"
        "- Priority 2 (medium): Standard surveillance\n"
        "- Priority 3 (low): Background monitoring\n"
    )

    insights.append({
        'type': 'domain_knowledge',
        'topic': 'network_topology',
        'content': topology_doc,
    })

    return insights


# ============================================================
# Build Knowledge Base
# ============================================================

def build_knowledge_base(input_dir, output_path='knowledge_base.json'):
    """
    Build complete semantic knowledge base from scenario JSONs.

    Output structure:
    {
        "documents": [
            {"id": "...", "type": "...", "topic": "...", "content": "...", "metadata": {...}},
            ...
        ],
        "stats": {...}
    }
    """
    json_files = sorted(glob.glob(os.path.join(input_dir, 'scenario_*.json')))
    print(f"Found {len(json_files)} scenario files")

    all_scenarios = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        all_scenarios.append(process_scenario(data))

    documents = []
    doc_id = 0

    # 1. Full scenario documents
    print("Generating scenario documents...")
    for s in all_scenarios:
        doc = generate_scenario_document(s)
        documents.append({
            'id': f"scenario_{doc_id:04d}",
            'type': 'scenario_summary',
            'topic': s['scenario_id'],
            'content': doc,
            'metadata': {
                'scenario_id': s['scenario_id'],
                'background_traffic': s['input']['background_traffic_mbps'],
                'num_cloud_zones': sum(1 for z in s['output']['zone_summaries']
                                       if z['processing_location'] == 'cloud'),
                'avg_loss': round(np.mean([z['packet_loss_rate']
                                          for z in s['output']['zone_summaries']]), 4),
                'avg_delay': round(np.mean([z['avg_delay_ms']
                                           for z in s['output']['zone_summaries']]), 2),
            }
        })
        doc_id += 1

    # 2. Per-zone documents
    print("Generating zone documents...")
    for s in all_scenarios:
        for z in s['output']['zone_summaries']:
            doc = generate_zone_document(s, z)
            documents.append({
                'id': f"zone_{doc_id:04d}",
                'type': 'zone_detail',
                'topic': f"{s['scenario_id']}_zone_{z['zone_id']}",
                'content': doc,
                'metadata': {
                    'scenario_id': s['scenario_id'],
                    'zone_id': z['zone_id'],
                    'processing_location': z['processing_location'],
                    'model_type': z['model_type'],
                    'bitrate': z['bitrate_mbps'],
                    'priority': z['priority_class'],
                    'loss_rate': z['packet_loss_rate'],
                    'delay_ms': z['avg_delay_ms'],
                    'background_traffic': s['input']['background_traffic_mbps'],
                }
            })
            doc_id += 1

    # 3. Cross-scenario insights
    print("Generating pattern insights...")
    insights = generate_comparison_insights(all_scenarios)
    for insight in insights:
        documents.append({
            'id': f"insight_{doc_id:04d}",
            'type': insight['type'],
            'topic': insight['topic'],
            'content': insight['content'],
            'metadata': {}
        })
        doc_id += 1

    # Build output
    kb = {
        'documents': documents,
        'stats': {
            'total_documents': len(documents),
            'scenario_summaries': sum(1 for d in documents if d['type'] == 'scenario_summary'),
            'zone_details': sum(1 for d in documents if d['type'] == 'zone_detail'),
            'pattern_analyses': sum(1 for d in documents if d['type'] == 'pattern_analysis'),
            'domain_knowledge': sum(1 for d in documents if d['type'] == 'domain_knowledge'),
            'total_scenarios': len(all_scenarios),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(kb, f, indent=2)

    print(f"\n✅ Knowledge base saved to {output_path}")
    print(f"   📄 {kb['stats']['scenario_summaries']} scenario summaries")
    print(f"   🔍 {kb['stats']['zone_details']} zone details")
    print(f"   📊 {kb['stats']['pattern_analyses']} pattern analyses")
    print(f"   🏗️  {kb['stats']['domain_knowledge']} domain knowledge docs")
    print(f"   📦 Total: {kb['stats']['total_documents']} documents")

    return kb


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':
    import sys
    input_dir = sys.argv[1] if len(sys.argv) > 1 else './dataset'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'knowledge_base.json'

    kb = build_knowledge_base(input_dir, output_path)

    # Preview
    print("\n" + "=" * 60)
    print("SAMPLE DOCUMENTS")
    print("=" * 60)

    # Show one scenario summary
    scenario_docs = [d for d in kb['documents'] if d['type'] == 'scenario_summary']
    if scenario_docs:
        print(f"\n--- Scenario Summary (first 500 chars) ---")
        print(scenario_docs[0]['content'][:500])

    # Show one zone detail
    zone_docs = [d for d in kb['documents'] if d['type'] == 'zone_detail']
    if zone_docs:
        print(f"\n--- Zone Detail ---")
        print(zone_docs[0]['content'])

    # Show one insight
    insight_docs = [d for d in kb['documents'] if d['type'] == 'pattern_analysis']
    if insight_docs:
        print(f"\n--- Pattern Insight (first 500 chars) ---")
        print(insight_docs[0]['content'][:500])