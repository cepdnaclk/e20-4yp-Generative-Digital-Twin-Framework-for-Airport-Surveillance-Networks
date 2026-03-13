# ============================================================
# Airport GNN API Server — Single file, minimal setup
# Run: pip install fastapi uvicorn torch torch-geometric numpy
# Then: python gnn_server.py
# ============================================================

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# ============================================================
# CONSTANTS
# ============================================================
CAMS_PER_ZONE = 10
NUM_ZONES = 6
TOTAL_CAMERAS = 60
NUM_EDGE_SERVERS = 3

SWITCH_START = 60
NODE_CORE = 66
EDGE_START = 67
NODE_CLOUD = 70
NUM_NODES = 71

LINK_CAM_TO_SWITCH = 30.0
LINK_SWITCH_TO_CORE = 100.0
LINK_CORE_TO_EDGE = 150.0
LINK_CORE_TO_CLOUD = 80.0

DELAY_CAM_TO_SWITCH = 1.0
DELAY_CORE_TO_EDGE = 5.0
DELAY_CORE_TO_CLOUD = 25.0

SWITCH_TO_CORE_DELAY = {0: 2.0, 1: 3.0, 2: 4.0, 3: 5.0, 4: 2.0, 5: 3.0}

MODEL_TYPE_MAP = {'lightweight': 0, 'medium': 1, 'heavy': 2}
PROC_DELAY_MAP = {'lightweight': 5.0, 'medium': 10.0, 'heavy': 20.0}
PROC_LOC_MAP = {
    'edge1': EDGE_START,
    'edge2': EDGE_START + 1,
    'edge3': EDGE_START + 2,
    'cloud': NODE_CLOUD,
}
TARGET_NAMES = ['throughput_mbps', 'packet_loss_rate', 'avg_delay_ms', 'jitter_ms']


# ============================================================
# GNN MODEL
# ============================================================
class AirportGNN(nn.Module):
    def __init__(self, in_channels=12, edge_dim=4, hidden=96, heads=4,
                 out_channels=4, dropout=0.15, num_layers=4):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(hidden, hidden // heads, heads=heads,
                        edge_dim=edge_dim, dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden))

        self.congestion_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 2),
        )
        self.delay_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1),
        )
        self.jitter_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.input_proj(x)
        for i in range(self.num_layers):
            residual = x
            x = self.convs[i](x, edge_index, edge_attr)
            x = F.elu(x)
            x = self.norms[i](x + residual)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        zone_features = x[data.zone_mask]
        batch_size = batch.max().item() + 1

        predictions = torch.cat([
            self.congestion_head(zone_features),
            self.delay_head(zone_features),
            self.jitter_head(zone_features),
        ], dim=-1)
        return predictions.view(batch_size, NUM_ZONES, 4)


# ============================================================
# GRAPH BUILDER
# ============================================================
def compute_analytical_delay(zone_config):
    z = zone_config['zone_id']
    core_to_dest = DELAY_CORE_TO_CLOUD if zone_config['processing_location'] == 'cloud' else DELAY_CORE_TO_EDGE
    return DELAY_CAM_TO_SWITCH + SWITCH_TO_CORE_DELAY[z] + core_to_dest + PROC_DELAY_MAP[zone_config['model_type']]


def build_graph(scenario_input):
    zones = sorted(scenario_input['zones'], key=lambda z: z['zone_id'])
    bg_traffic = scenario_input['background_traffic_mbps']

    load_on_dest = defaultdict(float)
    zones_on_dest = defaultdict(int)
    total_core_load = 0.0

    for zc in zones:
        dest_node = PROC_LOC_MAP[zc['processing_location']]
        zone_demand = zc['bitrate_mbps'] * CAMS_PER_ZONE
        effective = min(zone_demand, LINK_SWITCH_TO_CORE)
        load_on_dest[dest_node] += effective
        zones_on_dest[dest_node] += 1
        total_core_load += effective

    baselines = [compute_analytical_delay(zc) for zc in zones]

    FEAT_DIM = 12
    node_features = torch.zeros((NUM_NODES, FEAT_DIM), dtype=torch.float)

    for zc in zones:
        z = zc['zone_id']
        bitrate = zc['bitrate_mbps']
        priority = zc['priority_class']
        model_type = MODEL_TYPE_MAP[zc['model_type']]
        dest_node = PROC_LOC_MAP[zc['processing_location']]
        dest_load = load_on_dest.get(dest_node, 0.0) + bg_traffic

        proc_delay = PROC_DELAY_MAP[zc['model_type']]
        switch_to_core_delay = SWITCH_TO_CORE_DELAY[z]
        core_to_dest_delay = DELAY_CORE_TO_CLOUD if zc['processing_location'] == 'cloud' else DELAY_CORE_TO_EDGE
        dest_capacity = LINK_CORE_TO_CLOUD if zc['processing_location'] == 'cloud' else LINK_CORE_TO_EDGE

        total_path_delay = DELAY_CAM_TO_SWITCH + switch_to_core_delay + core_to_dest_delay
        dest_congestion = dest_load / dest_capacity
        zone_demand = bitrate * CAMS_PER_ZONE
        switch_congestion = zone_demand / LINK_SWITCH_TO_CORE

        for c in range(CAMS_PER_ZONE):
            cam_idx = z * CAMS_PER_ZONE + c
            node_features[cam_idx] = torch.tensor([
                bitrate / 25.0,
                float(priority == 1), float(priority == 2), float(priority == 3),
                float(model_type == 0), float(model_type == 1), float(model_type == 2),
                bg_traffic / 20.0, total_path_delay / 60.0, proc_delay / 25.0,
                switch_congestion, dest_congestion,
            ])

    for zc in zones:
        z = zc['zone_id']
        sw_node = SWITCH_START + z
        zone_demand = zc['bitrate_mbps'] * CAMS_PER_ZONE
        dest_node = PROC_LOC_MAP[zc['processing_location']]
        proc_delay = PROC_DELAY_MAP[zc['model_type']]
        switch_to_core_delay = SWITCH_TO_CORE_DELAY[z]
        core_to_dest_delay = DELAY_CORE_TO_CLOUD if zc['processing_location'] == 'cloud' else DELAY_CORE_TO_EDGE
        dest_capacity = LINK_CORE_TO_CLOUD if zc['processing_location'] == 'cloud' else LINK_CORE_TO_EDGE
        total_path_delay = DELAY_CAM_TO_SWITCH + switch_to_core_delay + core_to_dest_delay
        dest_load = load_on_dest.get(dest_node, 0.0) + bg_traffic
        dest_congestion = dest_load / dest_capacity
        switch_congestion = zone_demand / LINK_SWITCH_TO_CORE
        switch_qf = 1.0 / max(1.0 - min(switch_congestion, 0.95), 0.05)
        dest_qf = 1.0 / max(1.0 - min(dest_congestion, 0.95), 0.05)

        node_features[sw_node] = torch.tensor([
            zone_demand / LINK_SWITCH_TO_CORE, zc['bitrate_mbps'] / 25.0,
            float(zc['priority_class']) / 3.0, proc_delay / 25.0,
            total_path_delay / 60.0, dest_congestion,
            switch_qf / 20.0, dest_qf / 20.0,
            float(zc['processing_location'] == 'cloud'),
            float(zc['processing_location'] == 'edge1'),
            float(zc['processing_location'] == 'edge2'),
            float(zc['processing_location'] == 'edge3'),
        ])

    total_downstream_cap = LINK_CORE_TO_EDGE * NUM_EDGE_SERVERS + LINK_CORE_TO_CLOUD
    core_util = total_core_load / total_downstream_cap
    core_qf = 1.0 / max(1.0 - min(core_util, 0.95), 0.05)
    node_features[NODE_CORE] = torch.tensor([
        core_util, total_core_load / 1000.0,
        float(len(zones_on_dest)) / 4.0, bg_traffic / 20.0, core_qf / 20.0,
        load_on_dest.get(NODE_CLOUD, 0) / LINK_CORE_TO_CLOUD,
        load_on_dest.get(EDGE_START, 0) / LINK_CORE_TO_EDGE,
        load_on_dest.get(EDGE_START + 1, 0) / LINK_CORE_TO_EDGE,
        load_on_dest.get(EDGE_START + 2, 0) / LINK_CORE_TO_EDGE,
        0.0, 0.0, 0.0,
    ])

    for e in range(NUM_EDGE_SERVERS):
        edge_node = EDGE_START + e
        load = load_on_dest.get(edge_node, 0.0)
        total_load = load + bg_traffic
        n_zones = zones_on_dest.get(edge_node, 0)
        util = total_load / LINK_CORE_TO_EDGE
        qf = 1.0 / max(1.0 - min(util, 0.95), 0.05)
        node_features[edge_node] = torch.tensor([
            util, load / LINK_CORE_TO_EDGE, float(n_zones) / NUM_ZONES,
            bg_traffic / 20.0, qf / 20.0, DELAY_CORE_TO_EDGE / 60.0,
            float(e == 0), float(e == 1), float(e == 2), 0.0, 0.0, 0.0,
        ])

    cloud_load = load_on_dest.get(NODE_CLOUD, 0.0)
    cloud_total = cloud_load + bg_traffic
    cloud_zones = zones_on_dest.get(NODE_CLOUD, 0)
    cloud_util = cloud_total / LINK_CORE_TO_CLOUD
    cloud_qf = 1.0 / max(1.0 - min(cloud_util, 0.95), 0.05)
    node_features[NODE_CLOUD] = torch.tensor([
        cloud_util, cloud_load / LINK_CORE_TO_CLOUD,
        float(cloud_zones) / NUM_ZONES, bg_traffic / 20.0,
        cloud_qf / 20.0, DELAY_CORE_TO_CLOUD / 60.0,
        cloud_load / 1000.0, float(cloud_zones), 0.0, 0.0, 0.0, 0.0,
    ])

    # Edges
    src, dst, attrs = [], [], []

    for zc in zones:
        z = zc['zone_id']
        sw = SWITCH_START + z
        cam_load = zc['bitrate_mbps'] / LINK_CAM_TO_SWITCH
        bn = float(zc['bitrate_mbps'] > LINK_CAM_TO_SWITCH * 0.8)
        a = [cam_load, LINK_CAM_TO_SWITCH / 1000.0, DELAY_CAM_TO_SWITCH / 100.0, bn]
        for c in range(CAMS_PER_ZONE):
            ci = z * CAMS_PER_ZONE + c
            src.extend([ci, sw]); dst.extend([sw, ci]); attrs.extend([a, a])

    for zc in zones:
        z = zc['zone_id']
        sw = SWITCH_START + z
        zd = zc['bitrate_mbps'] * CAMS_PER_ZONE
        lr = zd / LINK_SWITCH_TO_CORE
        d = SWITCH_TO_CORE_DELAY[z]
        bn = float(zd > LINK_SWITCH_TO_CORE * 0.8)
        a = [lr, LINK_SWITCH_TO_CORE / 1000.0, d / 100.0, bn]
        src.extend([sw, NODE_CORE]); dst.extend([NODE_CORE, sw]); attrs.extend([a, a])

    for e in range(NUM_EDGE_SERVERS):
        en = EDGE_START + e
        ld = load_on_dest.get(en, 0.0) + bg_traffic
        lr = ld / LINK_CORE_TO_EDGE
        bn = float(ld > LINK_CORE_TO_EDGE * 0.8)
        a = [lr, LINK_CORE_TO_EDGE / 1000.0, DELAY_CORE_TO_EDGE / 100.0, bn]
        src.extend([NODE_CORE, en]); dst.extend([en, NODE_CORE]); attrs.extend([a, a])

    cl = load_on_dest.get(NODE_CLOUD, 0.0) + bg_traffic
    cr = cl / LINK_CORE_TO_CLOUD
    bn = float(cl > LINK_CORE_TO_CLOUD * 0.8)
    a = [cr, LINK_CORE_TO_CLOUD / 1000.0, DELAY_CORE_TO_CLOUD / 100.0, bn]
    src.extend([NODE_CORE, NODE_CLOUD]); dst.extend([NODE_CLOUD, NODE_CORE]); attrs.extend([a, a])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float)

    zone_mask = torch.zeros(NUM_NODES, dtype=torch.bool)
    zone_mask[SWITCH_START:SWITCH_START + NUM_ZONES] = True

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, zone_mask=zone_mask)
    return data, baselines


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================
class ZoneInput(BaseModel):
    zone_id: int
    bitrate_mbps: float
    priority_class: int
    model_type: str
    processing_location: str

class ScenarioInput(BaseModel):
    background_traffic_mbps: float
    zones: List[ZoneInput]

class ZoneResult(BaseModel):
    zone_id: int
    processing_location: str
    model_type: str
    bitrate_mbps: float
    throughput_mbps: float
    packet_loss_rate: float
    avg_delay_ms: float
    jitter_ms: float

class PredictionResponse(BaseModel):
    status: str
    predictions: List[ZoneResult]


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Airport GNN QoS Predictor", version="1.0")

# Global predictor — loaded once at startup
predictor = None


@app.on_event("startup")
def load_model():
    global predictor
    # Look for checkpoint in same folder
    checkpoint_path = os.environ.get(
        "CHECKPOINT_PATH",
        os.path.join(os.path.dirname(__file__), "airport_gnn_checkpoint.pt")
    )
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("   Set CHECKPOINT_PATH env var or place file in same folder")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = AirportGNN(**ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    predictor = {
        'model': model,
        'device': device,
        'target_mean': ckpt['target_mean'].to(device),
        'target_std': ckpt['target_std'].to(device),
    }
    print(f"✅ Model loaded on {device}")


@app.get("/health")
def health():
    return {
        "status": "ok" if predictor else "model_not_loaded",
        "device": str(predictor['device']) if predictor else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(scenario: ScenarioInput):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate
    if len(scenario.zones) != 6:
        raise HTTPException(status_code=400, detail="Must have exactly 6 zones")

    scenario_dict = scenario.model_dump()

    # Build graph
    graph, baselines = build_graph(scenario_dict)
    graph = graph.to(predictor['device'])
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=predictor['device'])

    # Predict
    with torch.no_grad():
        pred_norm = predictor['model'](graph)

    # Denormalize
    pred = pred_norm * predictor['target_std'] + predictor['target_mean']

    for z in range(NUM_ZONES):
        pred[0, z, 2] = pred[0, z, 2] + baselines[z]

    pred[:, :, 3] = torch.expm1(pred[:, :, 3])
    pred = pred.squeeze(0).cpu().numpy()

    zones = sorted(scenario_dict['zones'], key=lambda z: z['zone_id'])

    results = []
    for z in range(NUM_ZONES):
        results.append(ZoneResult(
            zone_id=z,
            processing_location=zones[z]['processing_location'],
            model_type=zones[z]['model_type'],
            bitrate_mbps=zones[z]['bitrate_mbps'],
            throughput_mbps=round(float(max(pred[z, 0], 0.0)), 4),
            packet_loss_rate=round(float(np.clip(pred[z, 1], 0.0, 1.0)), 4),
            avg_delay_ms=round(float(max(pred[z, 2], 0.0)), 4),
            jitter_ms=round(float(max(pred[z, 3], 0.0)), 4),
        ))

    return PredictionResponse(status="ok", predictions=results)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)