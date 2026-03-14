"""
qlearning_agent.py
Q-Learning agent for network configuration optimization
"""

import numpy as np
from collections import defaultdict
import json

class QNetworkConfigAgent:
    """Q-Learning agent for optimizing network configurations."""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=0.2, exploration_decay=0.995):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        # Q-table: {state_tuple: {action: q_value}}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Action space for modifying configurations
        self.actions = {
            'bitrate_-2': lambda v: max(1.0, v - 2.0),
            'bitrate_+2': lambda v: min(25.0, v + 2.0),
            'model_lightweight': 'lightweight',
            'model_medium': 'medium',
            'model_heavy': 'heavy',
            'location_edge1': 'edge1',
            'location_edge2': 'edge2',
            'location_edge3': 'edge3',
            'location_cloud': 'cloud',
            'priority_1': 1,
            'priority_2': 2,
            'priority_3': 3,
        }
    
    def encode_state(self, config):
        """Convert config dict to a hashable state tuple."""
        state_parts = [round(config['background_traffic_mbps'], 1)]
        
        for zone in sorted(config['zones'], key=lambda z: z['zone_id']):
            state_parts.extend([
                round(zone['bitrate_mbps'], 1),
                zone['model_type'],
                zone['processing_location'],
                zone['priority_class']
            ])
        
        return tuple(state_parts)
    
    def decode_state(self, state_tuple):
        """Reconstruct config from state tuple."""
        config = {'background_traffic_mbps': state_tuple[0], 'zones': []}
        
        idx = 1
        for zone_id in range(6):
            config['zones'].append({
                'zone_id': zone_id,
                'bitrate_mbps': state_tuple[idx],
                'model_type': state_tuple[idx + 1],
                'processing_location': state_tuple[idx + 2],
                'priority_class': state_tuple[idx + 3]
            })
            idx += 4
        
        return config
    
    def select_action(self, state, exploit_only=False):
        """Select zone and action using epsilon-greedy."""
        if exploit_only or np.random.random() > self.exploration_rate:
            # Exploit
            state_actions = self.q_table[state]
            if not state_actions:
                zone_id = np.random.randint(0, 6)
                action_name = np.random.choice(list(self.actions.keys()))
            else:
                best = max(state_actions.items(), key=lambda x: x[1])
                zone_id, action_name = best[0]
        else:
            # Explore
            zone_id = np.random.randint(0, 6)
            action_name = np.random.choice(list(self.actions.keys()))
        
        return zone_id, action_name
    
    def apply_action(self, config, zone_id, action_name):
        """Apply action to zone and return new config."""
        new_config = json.loads(json.dumps(config))
        zone = new_config['zones'][zone_id]
        action_fn = self.actions[action_name]
        
        if 'bitrate' in action_name:
            zone['bitrate_mbps'] = action_fn(zone['bitrate_mbps'])
        elif 'model_' in action_name:
            zone['model_type'] = action_fn
        elif 'location_' in action_name:
            zone['processing_location'] = action_fn
        elif 'priority_' in action_name:
            zone['priority_class'] = action_fn
        
        return new_config
    
    def compute_reward(self, qos_score):
        """Convert QoS score to reward (negative, since lower QoS is better)."""
        return -qos_score
    
    def update_q_value(self, state, action_tuple, reward, next_state):
        """Update Q-table using Q-learning formula."""
        old_q = self.q_table[state][action_tuple]
        next_q_values = self.q_table[next_state]
        max_next_q = max(next_q_values.values()) if next_q_values else 0
        
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_next_q - old_q)
        self.q_table[state][action_tuple] = new_q
    
    def train_on_config_pair(self, config, qos_score):
        """Single training step: take random action and update Q-value."""
        state = self.encode_state(config)
        zone_id, action_name = self.select_action(state)
        action_tuple = (zone_id, action_name)
        
        # Apply action
        new_config = self.apply_action(config, zone_id, action_name)
        new_state = self.encode_state(new_config)
        
        # Reward is based on current QoS (this is a heuristic during training)
        reward = self.compute_reward(qos_score)
        
        # Update Q-value
        self.update_q_value(state, action_tuple, reward, new_state)
        
        # Decay exploration
        self.exploration_rate *= self.exploration_decay
        
        return {
            'config': new_config,
            'zone_id': zone_id,
            'action': action_name,
            'reward': reward,
            'old_qos': qos_score
        }
    
    def optimize_config(self, initial_config, qos_evaluator, max_steps=5):
        """Use trained agent to optimize a config."""
        current_config = initial_config
        state = self.encode_state(current_config)
        
        best_qos = qos_evaluator(current_config)
        best_config = json.loads(json.dumps(current_config))
        history = [{'config': best_config, 'qos_score': best_qos}]
        
        self.exploration_rate = 0.0  # Pure exploitation
        
        for step in range(max_steps):
            zone_id, action_name = self.select_action(state, exploit_only=True)
            new_config = self.apply_action(current_config, zone_id, action_name)
            qos_score = qos_evaluator(new_config)
            
            history.append({'config': new_config, 'qos_score': qos_score})
            
            if qos_score < best_qos:
                best_qos = qos_score
                best_config = new_config
            else:
                break
            
            current_config = new_config
            state = self.encode_state(current_config)
        
        return {
            'best_config': best_config,
            'best_qos_score': best_qos,
            'history': history
        }