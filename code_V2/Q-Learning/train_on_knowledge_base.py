"""
train_on_knowledge_base.py
Train Q-Learning agent directly using knowledge_base.json
No external GNN calls needed!
"""

import json
import os
import pickle
import re
import numpy as np
from pathlib import Path
from qlearning_agent import QNetworkConfigAgent
from typing import Dict, Any, List

# Configuration
KNOWLEDGE_BASE_PATH = "RAG files/knowledge_base.json"
AGENT_CHECKPOINT_DIR = "./agent_checkpoints"
os.makedirs(AGENT_CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# KNOWLEDGE BASE PARSER
# ============================================================

def parse_zone_from_content(content: str, zone_id: int) -> Dict[str, Any]:
    """Extract zone configuration from content string."""
    
    # Find zone section
    zone_pattern = rf"### Zone {zone_id}\s*→\s*(\w+(?:\d)?)"
    match = re.search(zone_pattern, content)
    if not match:
        return None
    
    location = match.group(1)
    
    # Extract bitrate
    bitrate_pattern = rf"Zone {zone_id}.*?Configuration:\s*([\d.]+)\s*Mbps"
    bitrate_match = re.search(bitrate_pattern, content, re.DOTALL)
    bitrate = float(bitrate_match.group(1)) if bitrate_match else 5.0
    
    # Extract model type
    model_pattern = rf"Zone {zone_id}.*?(?:lightweight|medium|heavy)"
    model_match = re.search(r"\b(lightweight|medium|heavy)\b", content[match.start():match.start() + 500])
    model_type = model_match.group(1) if model_match else "medium"
    
    # Extract priority (looking at description)
    priority_map = {'low': 3, 'medium': 2, 'high': 1}
    priority_pattern = rf"Zone {zone_id}.*?(low|medium|high)\s*priority"
    priority_match = re.search(priority_pattern, content, re.IGNORECASE)
    priority = priority_map.get(priority_match.group(1).lower(), 2) if priority_match else 2
    
    return {
        'zone_id': zone_id,
        'bitrate_mbps': bitrate,
        'model_type': model_type,
        'processing_location': location,
        'priority_class': priority
    }

def extract_qos_metrics_from_content(content: str) -> Dict[str, float]:
    """Extract QoS metrics from scenario content."""
    metrics = {
        'avg_loss': 0.0,
        'avg_delay': 0.0,
        'total_packet_loss': 0.0
    }
    
    # Extract average loss and delay from Key Observations
    loss_pattern = r"avg\s+loss\s*=\s*([\d.]+)%"
    delay_pattern = r"avg\s+delay\s*=\s*([\d.]+)\s*ms"
    
    loss_match = re.search(loss_pattern, content)
    if loss_match:
        metrics['avg_loss'] = float(loss_match.group(1)) / 100.0
    
    delay_match = re.search(delay_pattern, content)
    if delay_match:
        metrics['avg_delay'] = float(delay_match.group(1))
    
    return metrics

def load_knowledge_base(filepath: str) -> List[Dict[str, Any]]:
    """Load and parse knowledge base into config + QoS pairs."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    scenarios = []
    
    for doc in data.get('documents', []):
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        
        # Extract zones
        zones = []
        for zone_id in range(6):
            zone_data = parse_zone_from_content(content, zone_id)
            if zone_data:
                zones.append(zone_data)
        
        if len(zones) != 6:
            continue  # Skip incomplete scenarios
        
        # Extract background traffic
        bg_traffic_pattern = r"Background traffic:\s*([\d.]+)\s*Mbps"
        bg_match = re.search(bg_traffic_pattern, content)
        bg_traffic = float(bg_match.group(1)) if bg_match else 10.0
        
        # Extract QoS metrics
        qos_metrics = extract_qos_metrics_from_content(content)
        
        # Build config
        config = {
            'background_traffic_mbps': bg_traffic,
            'zones': zones
        }
        
        # Compute composite QoS score
        # Simple formula: packet_loss (40%) + delay_normalized (30%) + other (30%)
        qos_score = (qos_metrics['avg_loss'] * 40 + 
                     (qos_metrics['avg_delay'] / 100.0) * 30 +
                     (qos_metrics.get('total_packet_loss', 0) * 30))
        
        scenarios.append({
            'scenario_id': metadata.get('scenario_id', doc.get('id')),
            'config': config,
            'qos_score': qos_score,
            'metadata': metadata
        })
    
    return scenarios

def qos_evaluator_from_knowledge(scenario_map: Dict[str, float]):
    """
    Create a QoS evaluator that looks up values from knowledge base.
    This is a heuristic: finds the closest config in our knowledge base.
    """
    def evaluate(config):
        state_tuple = QNetworkConfigAgent().encode_state(config)
        
        # Find closest known state (simple Euclidean distance on state vector)
        best_score = 100.0  # Default high score
        
        for state_key, qos in scenario_map.items():
            # Could add proper distance calculation here
            # For now, use a default heuristic based on config properties
            pass
        
        # Heuristic: penalize cloud overload
        cloud_load = sum(z['bitrate_mbps'] * 10 for z in config['zones'] 
                        if z['processing_location'] == 'cloud')
        bg = config['background_traffic_mbps']
        
        if cloud_load + bg > 80:
            overload_penalty = ((cloud_load + bg - 80) / 80) * 50
        else:
            overload_penalty = 0
        
        return max(1.0, overload_penalty + 5.0)
    
    return evaluate

# ============================================================
# TRAINING
# ============================================================

def train_agent_on_knowledge_base(scenarios: List[Dict], episodes: int = 200):
    """Train agent on knowledge base scenarios."""
    
    print(f"\n{'='*70}")
    print(f"Training Q-Learning Agent on Knowledge Base")
    print(f"{'='*70}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Training episodes: {episodes}\n")
    
    agent = QNetworkConfigAgent(
        learning_rate=0.15,
        discount_factor=0.95,
        exploration_rate=0.4,
        exploration_decay=0.997
    )
    
    best_qos = 999
    best_config = None
    training_log = []
    
    for episode in range(episodes):
        # Pick random scenario
        scenario = np.random.choice(scenarios)
        config = scenario['config']
        qos_score = scenario['qos_score']
        
        # Train on this scenario
        result = agent.train_on_config_pair(config, qos_score)
        
        # Track improvement
        if qos_score < best_qos:
            best_qos = qos_score
            best_config = config
            status = "✓ BEST"
        else:
            status = ""
        
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{episodes} | "
                  f"QoS={qos_score:.4f} | "
                  f"ε={agent.exploration_rate:.4f} | "
                  f"Q-States={len(agent.q_table)} | {status}")
        
        training_log.append({
            'episode': episode + 1,
            'qos_score': qos_score,
            'zone_id': result['zone_id'],
            'action': result['action'],
            'epsilon': agent.exploration_rate
        })
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best QoS Score: {best_qos:.4f}")
    print(f"States Learned: {len(agent.q_table)}")
    print(f"Final Epsilon: {agent.exploration_rate:.4f}")
    print(f"{'='*70}\n")
    
    return agent, training_log, best_config, best_qos

def save_agent(agent: QNetworkConfigAgent, version: str = "v1.0"):
    """Save trained agent."""
    checkpoint = {
        'q_table': dict(agent.q_table),
        'learning_rate': agent.learning_rate,
        'discount_factor': agent.discount_factor,
        'exploration_rate': agent.exploration_rate,
        'exploration_decay': agent.exploration_decay,
        'version': version,
        'timestamp': str(np.datetime64('now'))
    }
    
    filepath = f"{AGENT_CHECKPOINT_DIR}/qlearning_agent_{version}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"✓ Agent saved: {filepath}")
    return filepath

def save_metadata(training_log: List, best_config: Dict, best_qos: float, version: str = "v1.0"):
    """Save training metadata."""
    metadata = {
        'version': version,
        'timestamp': str(np.datetime64('now')),
        'training_episodes': len(training_log),
        'best_qos_score': best_qos,
        'best_config': best_config,
        'latest_training_log': training_log[-50:]
    }
    
    filepath = f"{AGENT_CHECKPOINT_DIR}/config_{version}.json"
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved: {filepath}")
    return filepath

if __name__ == "__main__":
    # Load knowledge base
    print("Loading knowledge base...")
    scenarios = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    print(f"✓ Loaded {len(scenarios)} scenarios\n")
    
    # Train agent
    agent, training_log, best_config, best_qos = train_agent_on_knowledge_base(
        scenarios,
        episodes=250
    )
    
    # Save agent
    save_agent(agent, version="v1.0")
    
    # Save metadata  
    save_metadata(training_log, best_config, best_qos, version="v1.0")
    
    print("\n✅ Agent trained and ready for deployment!")