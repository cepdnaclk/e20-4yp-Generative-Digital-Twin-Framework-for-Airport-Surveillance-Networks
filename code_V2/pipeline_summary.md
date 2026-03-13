# Airport Surveillance Network Optimization Pipeline — Technical Summary

**Project:** AI-Driven Network Configuration Optimization for Airport Surveillance  
**Stack:** GNN (Graph Attention Network) + RAG (Retrieval-Augmented Generation) + Gemini LLM  
**Date:** March 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Network Topology](#2-network-topology)
3. [Input Parameters](#3-input-parameters)
4. [GNN Architecture & Predictions](#4-gnn-architecture--predictions)
5. [QoS Metrics Explained](#5-qos-metrics-explained)
6. [Composite QoS Score](#6-composite-qos-score)
7. [RAG + LLM System](#7-rag--llm-system)
8. [Feedback Loop — Iterative Optimization](#8-feedback-loop--iterative-optimization)
9. [API Endpoints](#9-api-endpoints)
10. [Swagger UI](#10-swagger-ui)
11. [Deployment Architecture](#11-deployment-architecture)

---

## 1. System Overview

This pipeline provides **automated, AI-driven network configuration optimization** for an airport surveillance camera network. It combines three components:

| Component | Technology | Role |
|---|---|---|
| **GNN Server** | PyTorch Geometric (GAT) | Predict QoS metrics for any given config |
| **RAG Server** | ChromaDB + Gemini LLM | Translate natural language → config, iterate |
| **Knowledge Base** | 100 simulation scenarios | Ground truth for RAG retrieval |

The end-to-end flow:

```
User natural language request
        ↓
  RAG retrieves similar scenarios from knowledge base
        ↓
  Gemini LLM generates a network config
        ↓
  GNN predicts QoS for that config
        ↓
  Gemini evaluates results, refines config
        ↓
  Repeat (feedback loop) until converged or max iterations
        ↓
  Return best config + predictions
```

---

## 2. Network Topology

The physical network consists of **60 surveillance cameras** distributed across **6 zones**, with 10 cameras per zone.

```
[Zone 0–5 cameras]
       ↓
  Zone Switch (x6)
       ↓
  Core Router
    /    \
Edge Servers   Cloud Server
(x3, 150 Mbps  (80 Mbps link)
 each)
```

### Infrastructure capacity

| Resource | Bandwidth |
|---|---|
| Edge Server 1 (`edge1`) | 150 Mbps |
| Edge Server 2 (`edge2`) | 150 Mbps |
| Edge Server 3 (`edge3`) | 150 Mbps |
| Cloud (`cloud`) | 80 Mbps (bottleneck) |

> **Key constraint:** The cloud uplink is only 80 Mbps. If too many high-bitrate zones are assigned to cloud processing, it becomes a congestion bottleneck. The GNN learns this relationship from training data.

---

## 3. Input Parameters

### Per-zone parameters (6 zones × 4 parameters)

| Parameter | Type | Range / Values | Why We Chose This |
|---|---|---|---|
| `bitrate_mbps` | float | 1.0 – 25.0 Mbps | Controls how much bandwidth each camera stream consumes; directly drives congestion |
| `model_type` | categorical | `lightweight`, `medium`, `heavy` | Heavier models consume more compute on servers and can increase delay |
| `processing_location` | categorical | `edge1`, `edge2`, `edge3`, `cloud` | Determines which server handles the stream; drives load balancing |
| `priority_class` | integer | 1, 2, 3 | Higher priority (1) gets preferential queueing; affects delay/loss under congestion |

### Global parameter

| Parameter | Type | Range | Why We Chose This |
|---|---|---|---|
| `background_traffic_mbps` | float | 0.0 – 100.0 | Simulates non-camera network load (e.g., admin systems, VoIP); affects overall congestion |

### Why these parameters?

These parameters were chosen because they are the **actionable knobs** available to a network administrator — the things that can actually be changed in a real deployment. They collectively determine:
- How much traffic enters the network (`bitrate_mbps`, `background_traffic_mbps`)
- Where it gets processed (`processing_location`)
- How the network prioritizes it under load (`priority_class`)
- How computationally intensive the processing is (`model_type`)

All other network characteristics (link speeds, topology, number of cameras) are fixed infrastructure constraints captured in the GNN graph structure.

---

## 4. GNN Architecture & Predictions

### Model: `AirportGNN`

The GNN uses **Graph Attention Networks (GAT)** to model the entire airport network as a graph and predict QoS metrics.

### Graph structure

| Element | Count | Description |
|---|---|---|
| Nodes | 71 | 60 cameras + 6 zone switches + 3 edge servers + 1 cloud server + 1 core router |
| Edges | ~200+ | Physical links between cameras→switches→router→servers |
| Node features | 12 | Per-node input features (see below) |
| Edge features | 4 | Per-link properties (bandwidth, congestion, etc.) |

### Node feature engineering (12 features per node)

Each node in the graph carries 12 numerical features derived from the scenario config:

1. `bitrate_mbps` — stream bitrate (cameras); 0 for infrastructure nodes
2. `is_camera` — binary flag
3. `is_switch` — binary flag
4. `is_server` — binary flag
5. `model_type_encoded` — lightweight=0, medium=1, heavy=2
6. `priority_class` — 1/2/3
7. `processing_location_encoded` — edge1=0, edge2=1, edge3=2, cloud=3
8. `zone_total_bitrate` — sum of bitrate in this zone (propagated to switch)
9. `server_load` — total incoming bitrate at each server
10. `background_traffic_mbps` — global background load (replicated to all nodes)
11. `link_utilization` — estimated utilization of the relevant link
12. `is_cloud` — binary flag (highlights the bottleneck server)

### GNN layers

```
Input (12 node features, 4 edge features)
         ↓
  GATConv Layer 1  (96 hidden, 4 attention heads, ELU activation)
         ↓
  GATConv Layer 2  (96 hidden, 4 heads)
         ↓
  GATConv Layer 3  (96 hidden, 4 heads)
         ↓
  GATConv Layer 4  (96 hidden, 4 heads)
         ↓
  Per-zone aggregation (mean pooling of camera nodes in each zone)
         ↓
  Linear head (96 → 4 outputs per zone)
         ↓
Output: [throughput, packet_loss_rate, avg_delay_ms, jitter_ms] per zone
```

### Why GAT (Graph Attention Networks)?

Standard GNNs treat all neighbors equally. GAT learns **attention weights** — it can learn that, for example, the core router node should pay more attention to highly loaded switches than lightly loaded ones. This is crucial for modeling **congestion propagation** accurately across the network.

### Training

- **Dataset:** 100 simulation scenarios (`scenario_0000.json` – `scenario_0099.json`), each with a different config and measured QoS
- **Checkpoint:** `airport_gnn_checkpoint.pt`
- **Output:** 4 QoS values per zone × 6 zones = 24 predictions per inference

---

## 5. QoS Metrics Explained

The GNN predicts 4 Quality of Service metrics for each zone:

### Throughput (Mbps)
**Definition:** The actual data rate successfully delivered from the cameras in this zone to the processing server.  
**Why it matters:** If throughput is much lower than the configured bitrate, frames are being dropped and the surveillance feed is degraded.  
**Ideal:** As close to `bitrate_mbps` as possible (ideally = bitrate, meaning zero frame loss).

### Packet Loss Rate
**Definition:** The fraction of network packets that are dropped in transit (0.0 = no loss, 1.0 = all packets dropped).  
**Why it matters:** High packet loss means surveillance video frames are lost, creating gaps in the recorded footage. Even 5% packet loss visibly degrades video quality.  
**Typical threshold in literature:** < 1% for surveillance systems.  
**Causes:** Network congestion (especially cloud bottleneck), insufficient priority class.

### Average Delay (ms)
**Definition:** Mean end-to-end latency for packets from camera to server (in milliseconds).  
**Why it matters:** High delay affects real-time monitoring — if an incident occurs, operators see a delayed feed.  
**Typical threshold:** < 50ms for real-time monitoring, < 150ms acceptable for recorded footage.  
**Causes:** Heavy model processing time, congested links, routing through overloaded servers.

### Jitter (ms)
**Definition:** The variation in packet arrival time (standard deviation of delay). Also in milliseconds.  
**Why it matters:** High jitter causes video stuttering even if average delay is acceptable. It disrupts smooth playback because packets arrive at uneven intervals.  
**Typical threshold:** < 5ms for smooth video streams.  
**Causes:** Same as delay, but particularly affected by bursty traffic patterns.

---

## 6. Composite QoS Score

To compare configurations objectively, we compute a single **composite QoS score** that aggregates all four metrics across all six zones.

### Formula

$$\text{score}_{\text{zone}} = (L \times 40) + \left(\frac{D}{100} \times 30\right) + \left(\frac{\max(0, B - T)}{B} \times 20\right) + \left(\min\left(\frac{J}{1.0}, 1\right) \times 10\right)$$

$$\text{QoS Score} = \frac{1}{6} \sum_{\text{zones}} \text{score}_{\text{zone}}$$

Where:
- $L$ = `packet_loss_rate`
- $D$ = `avg_delay_ms`
- $B$ = `bitrate_mbps` (configured)
- $T$ = `throughput_mbps` (actual)
- $J$ = `jitter_ms`

### Weight rationale

| Metric | Weight | Rationale |
|---|---|---|
| Packet loss | **40%** | Highest impact — lost packets = lost video frames, unrecoverable |
| Delay | **30%** | Second highest — real-time monitoring requirement |
| Throughput deficit | **20%** | Bandwidth not delivered = degraded resolution/framerate |
| Jitter | **10%** | Lowest — annoying but less critical than outright loss |

### Score interpretation

| Score | Quality |
|---|---|
| 0 – 5 | Excellent — all metrics near-ideal |
| 5 – 15 | Good — minor degradation |
| 15 – 30 | Moderate — noticeable quality loss |
| 30+ | Poor — significant congestion or loss |

**Lower is always better.** No thresholds are needed — Gemini uses these scores comparatively to decide if further optimization is worthwhile.

---

## 7. RAG + LLM System

### Knowledge Base Construction (`rag.py`)

The RAG system is built offline by processing the 100 simulation scenario files:

```
dataset/scenario_0000.json → scenario_0099.json
               ↓
   Parse each scenario (config + measured QoS)
               ↓
   Generate natural language documents:
   - Per-zone performance documents
   - Global scenario summaries
   - Pattern documents ("when cloud is used with high bitrate...")
               ↓
   Embed with all-MiniLM-L6-v2 (sentence-transformers)
               ↓
   Store in ChromaDB (7,004 documents total)
```

### Retrieval at inference time

When a user sends a message like *"minimize packet loss in all zones"*:

1. The message is embedded using the same `all-MiniLM-L6-v2` model
2. ChromaDB performs a **cosine similarity search** against all 7,004 documents
3. The **top 10 most similar** simulation results are retrieved
4. These are injected into the Gemini prompt as context

This means Gemini has access to real simulation data showing *"when zone 3 was moved from cloud to edge2 with bitrate 6.0, packet loss dropped from 0.08 to 0.002"* — enabling evidence-based configuration decisions.

### Gemini LLM (`gemini-2.5-flash-lite`)

Gemini plays two roles:

| Role | When | What it does |
|---|---|---|
| **Config generator** | Iteration 1 | Translates user intent + RAG context → valid JSON config |
| **Evaluator/Refiner** | Iterations 2–N | Reviews full history of configs + GNN results → decides whether to continue and generates next config |

---

## 8. Feedback Loop — Iterative Optimization

The `/optimize` endpoint implements a closed-loop optimization system:

```
┌─────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION LOOP                        │
│                                                             │
│  User message + current config                              │
│        ↓                                                    │
│  [ChromaDB] Retrieve 10 similar scenarios                   │
│        ↓                                                    │
│  Iteration 1:                                               │
│    Gemini generates config from message + RAG context       │
│        ↓                                                    │
│  GNN predicts QoS → compute composite score                 │
│        ↓                                                    │
│  Gemini reviews: "Can we do better?"                        │
│    YES → generates next config to try                       │
│    NO  → stop early, return best config                     │
│        ↓                                                    │
│  Iteration 2, 3, ... (up to max_iterations)                 │
│        ↓                                                    │
│  Return config with lowest QoS score + full history         │
└─────────────────────────────────────────────────────────────┘
```

### What Gemini sees at each evaluation step

At iteration $i$, Gemini receives:
- The user's original optimization goal
- **Complete history** of every config tried so far + its GNN predictions
- The composite QoS score for each iteration
- Relevant simulation data from the knowledge base
- Network capacity constraints (link speeds, valid parameter ranges)

Gemini then outputs:
```json
{
  "should_continue": true,
  "reasoning": "Zones 2 and 4 are routed to cloud which is at 78 Mbps, near the 80 Mbps limit. Moving zone 2 to edge1 should reduce cloud congestion and lower packet loss.",
  "next_config": { ... full 6-zone config ... }
}
```

### Why no hardcoded thresholds?

Rather than defining *"packet loss must be < 1%"*, Gemini autonomously evaluates whether the current results are satisfactory given:
- The user's goal (e.g., "minimize packet loss" vs "balance all metrics")
- The trajectory of improvement across iterations
- Physical network constraints from the simulation data

This makes the system **goal-adaptive** — it doesn't treat a 0.5% packet loss as failure when the goal was just to reduce background traffic.

---

## 9. API Endpoints

Both servers expose REST APIs documented automatically via Swagger.

### GNN Server — `http://<host>:8002`

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | GET | Check if model is loaded |
| `POST /predict` | POST | Run GNN inference on a config |
| `GET /docs` | GET | Swagger UI |

**`POST /predict` input:**
```json
{
  "background_traffic_mbps": 15.9,
  "zones": [
    {"zone_id": 0, "bitrate_mbps": 7.0, "priority_class": 2,
     "model_type": "medium", "processing_location": "edge2"},
    ...
  ]
}
```

**`POST /predict` output:**
```json
{
  "predictions": [
    {
      "zone_id": 0,
      "processing_location": "edge2",
      "model_type": "medium",
      "bitrate_mbps": 7.0,
      "throughput_mbps": 6.87,
      "packet_loss_rate": 0.0032,
      "avg_delay_ms": 18.4,
      "jitter_ms": 0.21
    },
    ...
  ]
}
```

### RAG Server — `http://<host>:8001`

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | GET | Check ChromaDB + Gemini + GNN status |
| `POST /chat` | POST | Natural language → config + GNN predictions |
| `POST /ask` | POST | Q&A against knowledge base |
| `POST /optimize` | POST | Full iterative feedback loop |
| `GET /docs` | GET | Swagger UI |

**`POST /chat` — single-shot config generation:**
```json
// Request
{"message": "move zone 3 to edge1 and increase its bitrate"}

// Response
{
  "status": "ok",
  "generated_config": { ... },
  "gnn_predictions": [ ... ],
  "changes": [
    {"zone_id": 3, "field": "processing_location", "old_value": "edge3", "new_value": "edge1"},
    {"zone_id": 3, "field": "bitrate_mbps", "old_value": 6.4, "new_value": 9.0}
  ]
}
```

**`POST /optimize` — iterative optimization:**
```json
// Request
{
  "message": "minimize packet loss for all zones",
  "max_iterations": 4
}

// Response
{
  "status": "ok",
  "best_config": { ... },
  "best_qos_score": 3.21,
  "total_iterations": 3,
  "stopped_early": true,
  "iterations": [
    {
      "iteration": 1,
      "qos_score": 8.45,
      "gemini_assessment": "Initial config routes zones 1, 3, 5 to cloud. Cloud link at 72 Mbps (~90% utilization). Moving high-bitrate zones to edge should reduce congestion.",
      "should_continue": true
    },
    {
      "iteration": 2,
      "qos_score": 4.12,
      "gemini_assessment": "Significant improvement. Zone 4 still has 0.8% loss. Reducing priority_class from 3 to 1 should help.",
      "should_continue": true
    },
    {
      "iteration": 3,
      "qos_score": 3.21,
      "gemini_assessment": "Marginal improvement possible. All zones below 0.3% loss. Further changes unlikely to yield meaningful gains.",
      "should_continue": false
    }
  ]
}
```

---

## 10. Swagger UI

Both servers automatically generate **interactive API documentation** via FastAPI's built-in Swagger integration.

### Accessing Swagger

| Server | URL |
|---|---|
| GNN Server | `http://<host>:8002/docs` |
| RAG Server | `http://<host>:8001/docs` |

### What Swagger provides

- **Interactive forms** — fill in request fields directly in the browser, no Postman needed
- **Live API testing** — click "Try it out", enter parameters, hit "Execute", see the real JSON response
- **Schema documentation** — every request and response field is described with its type and constraints
- **Request examples** — pre-filled example values for each endpoint

### How it's generated

FastAPI automatically creates Swagger docs from the **Pydantic models** defined in the code. For example:

```python
class OptimizeRequest(BaseModel):
    message: str                              # → required string field in Swagger
    current_config: Optional[Dict] = None    # → optional field, shown as nullable
    max_iterations: int = 4                  # → integer field with default value 4
```

This means the documentation is always **in sync with the code** — no separate documentation files to maintain.

---

## 11. Deployment Architecture

### Current deployment (AWS EC2)

```
                    Internet
                        │
              ┌─────────┴─────────┐
              │    EC2 Instance   │
              │   Ubuntu, t2.x    │
              │                   │
              │  Port 8001 (RAG)  │
              │  Port 8002 (GNN)  │
              │                   │
              │  ~/myenv/         │  ← Python virtualenv
              │  ~/rag_server.py  │
              │  ~/gnn_server.py  │
              │  ~/airport_gnn_checkpoint.pt │
              │  ~/airport_chromadb/         │
              └───────────────────┘
```

### Service startup

```bash
source ~/myenv/bin/activate
export GEMINI_API_KEY="your_key_here"

# Start GNN server
nohup python3 gnn_server.py > gnn.log 2>&1 &

# Start RAG server
nohup python3 rag_server.py > rag.log 2>&1 &
```

### Environment requirements

| Dependency | Purpose |
|---|---|
| `torch` + `torch-geometric` | GNN inference |
| `fastapi` + `uvicorn` | REST API servers |
| `chromadb` | Vector database |
| `sentence-transformers` | Document embeddings |
| `google-generativeai` | Gemini API client |
| `requests` | RAG→GNN HTTP calls |
| `GEMINI_API_KEY` (env var) | Gemini authentication |

---

## Summary

This pipeline demonstrates a novel **closed-loop AI optimization system** for network management:

1. **GNN** provides fast, accurate QoS predictions without requiring live network simulation — inference takes milliseconds
2. **RAG** grounds the LLM in real simulation data, preventing hallucinated configurations
3. **Gemini** provides intelligent goal interpretation and adaptive refinement that no rule-based system could match
4. **The feedback loop** enables the system to iteratively improve configurations without human intervention, stopping automatically when convergence is detected

The result is a system where a network administrator with no deep networking expertise can type *"my zone 3 cameras have terrible latency, fix it"* and receive a fully optimized configuration backed by both machine learning predictions and historical simulation evidence.
