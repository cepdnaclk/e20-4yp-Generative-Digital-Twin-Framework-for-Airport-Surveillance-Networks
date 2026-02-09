import json
import torch
import numpy as np
import os
import networkx as nx
import sys
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import LabelEncoder

class AirportGraphDataset(Dataset):
    def __init__(self, root_dir, scenario_dir, result_dir):
        self.scenario_dir = scenario_dir
        self.result_dir = result_dir
        self.scenario_files = sorted([f for f in os.listdir(scenario_dir) if f.endswith('.json')])
        
        # --- ENCODERS ---
        self.node_type_encoder = LabelEncoder()
        self.zone_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder() # Encodes "TerminalA/CheckIn" -> Digit
        
        # 1. Fit Fixed Categories
        self.node_type_encoder.fit(['camera', 'edge', 'cloud', 'router'])
        self.zone_encoder.fit(['PZ1_CRITICAL_SECURITY', 'PZ2_BOARDING_GATES', 
                               'PZ3_PUBLIC_AREA', 'PZ4_VIP_RESTRICTED', 
                               'PZ5_ARRIVAL_BAGGAGE'])
        
        # 2. Dynamic Location Scanning
        all_locations = set()
        print("Scanning scenarios for locations...")
        for f_name in self.scenario_files:
            try:
                with open(os.path.join(self.scenario_dir, f_name)) as f:
                    scen = json.load(f)
                    for node in scen['cameras'] + scen['edge_servers'] + scen['cloud_endpoints']:
                        if 'location' in node:
                            all_locations.add(node['location'])
            except Exception as e:
                print(f"Warning: Could not read {f_name} for location scanning.")
        
        # Add default for unknown nodes
        all_locations.add("Network_Core")
        self.location_encoder.fit(list(all_locations))
        print(f"Found {len(self.location_encoder.classes_)} unique locations.")
        
        super().__init__(root_dir)

    def len(self):
        return len(self.scenario_files)

    def get(self, idx):
        # Load Files
        scen_path = os.path.join(self.scenario_dir, self.scenario_files[idx])
        res_path = os.path.join(self.result_dir, f"results_{idx:04d}.json")
        
        if not os.path.exists(res_path):
            return None 

        with open(scen_path) as f: scen = json.load(f)
        with open(res_path) as f: res = json.load(f)

        # --- TOPOLOGY RECONSTRUCTION ---
        G = nx.Graph()
        node_caps = {}
        node_locs = {} 
        all_nodes = set()
        
        # 1. Parse Nodes
        for node in scen['edge_servers'] + scen['cloud_endpoints']:
            nid = node['id']
            all_nodes.add(nid)
            cap = node.get('network_bandwidth_gbps', 10.0) * 1000.0
            node_caps[nid] = cap
            node_locs[nid] = node.get('location', "Network_Core")
            G.add_node(nid)

        for cam in scen['cameras']:
            nid = cam['id']
            all_nodes.add(nid)
            node_caps[nid] = 1000.0
            node_locs[nid] = cam.get('location', "Network_Core")
            G.add_node(nid)

        # 2. Parse Links
        for link in scen['network_links']:
            src, dst = link['src'], link['dst']
            all_nodes.add(src)
            all_nodes.add(dst)
            
            if src not in node_caps: 
                node_caps[src] = link['capacity_mbps']
                node_locs[src] = "Network_Core"
            if dst not in node_caps: 
                node_caps[dst] = link['capacity_mbps']
                node_locs[dst] = "Network_Core"
                
            G.add_edge(src, dst, bandwidth=link['capacity_mbps'], weight=1.0)

        # --- CALCULATE LOAD (Workaround) ---
        node_load = {n: 0.0 for n in all_nodes}
        
        for bg in scen.get('background_traffic', []):
            try:
                path = nx.shortest_path(G, bg['src'], bg['dst'])
                for n in path: node_load[n] += bg['bitrate_mbps']
            except: pass

        for flow in scen['flows']:
            try:
                path = nx.shortest_path(G, flow['source'], flow['destination'])
                for n in path: node_load[n] += flow['bitrate_mbps']
            except: pass

        # --- BUILD NODE TENSORS ---
        sorted_nodes = sorted(list(all_nodes))
        node_id_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
        
        x_features = []
        node_location_digits = []
        
        for n_id in sorted_nodes:
            feat = [0.0] * 32
            if "cam" in n_id: feat[0] = 1.0
            elif "edge" in n_id: feat[1] = 1.0
            elif "cloud" in n_id: feat[2] = 1.0
            else: feat[3] = 1.0
            
            cap = node_caps.get(n_id, 1000.0)
            feat[4] = node_load[n_id] / (cap + 1e-5)
            feat[5] = np.log1p(node_load[n_id])
            
            loc_str = node_locs.get(n_id, "Network_Core")
            loc_digit = self.location_encoder.transform([loc_str])[0]
            feat[6] = loc_digit / (len(self.location_encoder.classes_) + 1e-5)
            
            x_features.append(feat)
            node_location_digits.append(loc_digit)

        x = torch.tensor(x_features, dtype=torch.float)
        
        # --- BUILD EDGE TENSORS ---
        src_list, dst_list = [], []
        edge_attrs = []
        for u, v, d in G.edges(data=True):
            u_idx, v_idx = node_id_to_idx[u], node_id_to_idx[v]
            src_list.extend([u_idx, v_idx])
            dst_list.extend([v_idx, u_idx])
            bw = d.get('bandwidth', 1000.0) / 10000.0
            attr = [bw, 0.0] + [0]*14
            edge_attrs.extend([attr, attr])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # --- BUILD FLOW TENSORS & TARGETS ---
        flow_feats = []
        flow_mappings = []
        flow_zones = []
        flow_locs = []
        
        # Lists for ALL 4 Targets
        target_lat, target_loss = [], []
        target_jit, target_tput = [], []  # <--- [FIX] New Lists
        
        res_lookup = {r['flow_id']: r for r in res['flows']}
        
        for flow in scen['flows']:
            # Features
            f_vec = [
                flow['bitrate_mbps'] / 50.0,
                float(flow['priority']) / 3.0,
                flow.get('compute_intensity', 0.5),
                flow.get('processing_delay_ms', 0.0) / 200.0
            ] + [0]*20
            flow_feats.append(f_vec)
            
            # Map
            s_idx = node_id_to_idx[flow['source']]
            d_idx = node_id_to_idx[flow['destination']]
            flow_mappings.append([s_idx, d_idx])
            
            # Zone
            z_idx = self.zone_encoder.transform([flow['zone']])[0]
            flow_zones.append(z_idx)
            
            # Location Digit
            src_loc_str = node_locs.get(flow['source'], "Network_Core")
            l_idx = self.location_encoder.transform([src_loc_str])[0]
            flow_locs.append(l_idx)
            
            # Targets Extraction [FIX IS HERE]
            r = res_lookup.get(flow['id'], {})
            
            # Latency (Default 1000ms if missing)
            target_lat.append(r.get('avg_delay_ms', 1000.0))
            # Packet Loss (Default 1.0 if missing)
            target_loss.append(r.get('packet_loss_rate', 1.0))
            # Jitter (Default 100ms if missing)
            target_jit.append(r.get('jitter_ms', 100.0))
            # Throughput (Default 0.0 if missing)
            target_tput.append(r.get('throughput_mbps', 0.0))

        # --- AGGREGATION ---
        zone_targets = []
        for z_id in range(5):
            indices = [i for i, z in enumerate(flow_zones) if z == z_id]
            if not indices: zone_targets.append([0,0,0,0])
            else:
                lats = [target_lat[i] for i in indices]
                losses = [target_loss[i] for i in indices]
                zone_targets.append([np.mean(lats), np.max(lats), np.mean(losses), 1.0-np.max(losses)])

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            flow_features=torch.tensor(flow_feats, dtype=torch.float),
            flow_to_node=torch.tensor(flow_mappings, dtype=torch.long),
            flow_zone=torch.tensor(flow_zones, dtype=torch.long),
            flow_location=torch.tensor(flow_locs, dtype=torch.long),
            
            # ALL 4 Targets Packed Here
            flow_latency=torch.tensor(target_lat, dtype=torch.float),
            flow_packet_loss=torch.tensor(target_loss, dtype=torch.float),
            flow_jitter=torch.tensor(target_jit, dtype=torch.float),       # <--- [FIX]
            flow_throughput=torch.tensor(target_tput, dtype=torch.float),  # <--- [FIX]
            
            zone_kpis=torch.tensor(zone_targets, dtype=torch.float)
        )
        
        return data

# --- MAIN EXPORT BLOCK ---
if __name__ == "__main__":
    import torch
    import os
    import sys
    
    # 1. Setup Paths
    scenario_dir = "final_scenarios_core" 
    result_dir = "output"
    
    if not os.path.exists(scenario_dir):
        print(f"Error: Directory '{scenario_dir}' not found. Did you mean 'scenarios'?")
        sys.exit(1)

    print(f"--- Processing data from {scenario_dir} ---")
    
    # 2. Initialize
    ds = AirportGraphDataset(".", scenario_dir, result_dir)
    
    processed_data = []
    print(f"Exporting {len(ds)} scenarios to RAM...")
    
    # Iterate
    valid_count = 0
    for i in range(len(ds)):
        try:
            sample = ds[i]
            if sample is not None:
                processed_data.append(sample)
                valid_count += 1
            if i % 100 == 0:
                print(f"Processed {i}/{len(ds)}")
        except Exception as e:
            print(f"Skipping index {i} due to error: {e}")

    # 3. SAVE EVERYTHING (Data + Encoders)
    export_package = {
        'dataset': processed_data,
        'encoders': {
            'location': ds.location_encoder,
            'zone': ds.zone_encoder,
            'node_type': ds.node_type_encoder
        }
    }

    output_file = "airport_gnn_dataset.pt"
    print(f"Saving {valid_count} samples to {output_file}...")
    torch.save(export_package, output_file)
    print("Done! You can now download this file to Colab.")