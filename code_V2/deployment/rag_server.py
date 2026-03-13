# ============================================================
# Airport RAG + GNN API Server — Single file
# Run: pip install fastapi uvicorn chromadb sentence-transformers google-generativeai requests
# Then: python rag_server.py
# ============================================================

import json
import os
import re
import requests
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from copy import deepcopy

# ============================================================
# CONFIG
# ============================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")  # set via env or edit here
GNN_URL = os.environ.get("GNN_URL", "http://localhost:8002/predict")
CHROMADB_PATH = os.environ.get("CHROMADB_PATH", os.path.join(os.path.dirname(__file__), "airport_chromadb"))
# config
NS3_SIM_URL = os.environ.get("NS3_SIM_URL", "http://16.171.194.207:8000/simulate")

# ============================================================
# SYSTEM PROMPT FOR CONFIG GENERATION
# ============================================================
SYSTEM_PROMPT = """You are a network configuration generator for an airport surveillance system.

NETWORK RULES:
- 60 cameras, 6 zones, 10 cameras per zone
- processing_location: ONLY "edge1", "edge2", "edge3", or "cloud"
- model_type: ONLY "lightweight", "medium", or "heavy"
- priority_class: ONLY 1, 2, or 3
- bitrate_mbps: float between 1.0 and 25.0
- background_traffic_mbps: float between 0.0 and 100.0

IMPORTANT:
- Output ONLY valid JSON. No explanation, no markdown, no text before or after.
- Always output ALL 6 zones.
- Only change what the user asks for. Keep everything else EXACTLY the same.
- Use the retrieved simulation data to make smart choices about what values to use.

Output format (ONLY this, nothing else):
{
  "background_traffic_mbps": <float>,
  "zones": [
    {"zone_id": 0, "bitrate_mbps": <float>, "priority_class": <int>, "model_type": "<str>", "processing_location": "<str>"},
    {"zone_id": 1, "bitrate_mbps": <float>, "priority_class": <int>, "model_type": "<str>", "processing_location": "<str>"},
    {"zone_id": 2, "bitrate_mbps": <float>, "priority_class": <int>, "model_type": "<str>", "processing_location": "<str>"},
    {"zone_id": 3, "bitrate_mbps": <float>, "priority_class": <int>, "model_type": "<str>", "processing_location": "<str>"},
    {"zone_id": 4, "bitrate_mbps": <float>, "priority_class": <int>, "model_type": "<str>", "processing_location": "<str>"},
    {"zone_id": 5, "bitrate_mbps": <float>, "priority_class": <int>, "model_type": "<str>", "processing_location": "<str>"}
  ]
}"""

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_CONFIG = {
    "background_traffic_mbps": 15.9,
    "zones": [
        {"zone_id": 0, "bitrate_mbps": 7.0, "priority_class": 2, "model_type": "medium", "processing_location": "edge2"},
        {"zone_id": 1, "bitrate_mbps": 7.8, "priority_class": 2, "model_type": "lightweight", "processing_location": "edge3"},
        {"zone_id": 2, "bitrate_mbps": 5.9, "priority_class": 3, "model_type": "lightweight", "processing_location": "edge2"},
        {"zone_id": 3, "bitrate_mbps": 6.4, "priority_class": 1, "model_type": "medium", "processing_location": "edge3"},
        {"zone_id": 4, "bitrate_mbps": 5.9, "priority_class": 1, "model_type": "medium", "processing_location": "cloud"},
        {"zone_id": 5, "bitrate_mbps": 7.5, "priority_class": 1, "model_type": "heavy", "processing_location": "edge3"},
    ]
}

# ============================================================
# PYDANTIC SCHEMAS
# ============================================================
class ZoneConfig(BaseModel):
    zone_id: int
    bitrate_mbps: float
    priority_class: int
    model_type: str
    processing_location: str

class ScenarioConfig(BaseModel):
    background_traffic_mbps: float
    zones: List[ZoneConfig]

class ChatRequest(BaseModel):
    message: str
    current_config: Optional[Dict[str, Any]] = None

class ZonePrediction(BaseModel):
    zone_id: int
    processing_location: str
    model_type: str
    bitrate_mbps: float
    throughput_mbps: float
    packet_loss_rate: float
    avg_delay_ms: float
    jitter_ms: float

class ConfigChange(BaseModel):
    zone_id: Optional[int] = None
    field: str
    old_value: Any
    new_value: Any

class ChatResponse(BaseModel):
    status: str
    generated_config: Dict[str, Any]
    gnn_predictions: Optional[List[ZonePrediction]] = None
    changes: List[ConfigChange]
    gnn_error: Optional[str] = None

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    status: str
    answer: str
    sources: List[Dict[str, Any]]

class OptimizeRequest(BaseModel):
    message: str
    current_config: Optional[Dict[str, Any]] = None
    max_iterations: int = 4

class IterationResult(BaseModel):
    iteration: int
    config: Dict[str, Any]
    predictions: Optional[List[ZonePrediction]]
    qos_score: float           # lower is better (composite score, no thresholds)
    gemini_assessment: str     # Gemini's reasoning about this iteration
    should_continue: bool      # did Gemini think further improvement was possible

class OptimizeResponse(BaseModel):
    status: str
    best_config: Dict[str, Any]
    best_predictions: Optional[List[ZonePrediction]]
    best_qos_score: float
    iterations: List[IterationResult]
    total_iterations: int
    stopped_early: bool        # True if Gemini decided no further improvement possible
    gnn_error: Optional[str] = None

class DeployRequest(BaseModel):
    simulation_time: float
    scenario_id: Optional[str] = None

class DeployResponse(BaseModel):
    status: str
    deployed_payload: Dict[str, Any]
    ns3_result: Dict[str, Any]

# ============================================================
# GLOBALS
# ============================================================
collection = None
gemini_model = None
last_best_config = None

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def extract_json(text):
    """Extract and validate JSON config from LLM response."""
    raw = text.strip()

    if '```json' in raw:
        raw = raw.split('```json')[1]
    if '```' in raw:
        raw = raw.split('```')[0]
    raw = raw.strip()

    if not raw.startswith('{'):
        start = raw.find('{')
        if start != -1:
            raw = raw[start:]
    if not raw.endswith('}'):
        end = raw.rfind('}')
        if end != -1:
            raw = raw[:end + 1]

    config = json.loads(raw)

    assert 'background_traffic_mbps' in config, "Missing background_traffic_mbps"
    assert 'zones' in config and len(config['zones']) == 6, "Must have exactly 6 zones"
    for z in config['zones']:
        assert z['processing_location'] in ['edge1', 'edge2', 'edge3', 'cloud'], \
            f"Invalid location: {z['processing_location']}"
        assert z['model_type'] in ['lightweight', 'medium', 'heavy'], \
            f"Invalid model: {z['model_type']}"
        assert z['priority_class'] in [1, 2, 3], \
            f"Invalid priority: {z['priority_class']}"
        assert 0.5 <= z['bitrate_mbps'] <= 25.0, \
            f"Invalid bitrate: {z['bitrate_mbps']}"

    config['zones'] = sorted(config['zones'], key=lambda x: x['zone_id'])
    return config


def find_changes(old_config, new_config):
    """Compare two configs and return list of changes."""
    changes = []

    if new_config['background_traffic_mbps'] != old_config['background_traffic_mbps']:
        changes.append(ConfigChange(
            field="background_traffic_mbps",
            old_value=old_config['background_traffic_mbps'],
            new_value=new_config['background_traffic_mbps'],
        ))

    old_zones = sorted(old_config['zones'], key=lambda z: z['zone_id'])
    new_zones = sorted(new_config['zones'], key=lambda z: z['zone_id'])

    for old_z, new_z in zip(old_zones, new_zones):
        zid = old_z['zone_id']
        for field in ['processing_location', 'model_type', 'bitrate_mbps', 'priority_class']:
            old_val = old_z[field]
            new_val = new_z[field]
            if field == 'bitrate_mbps':
                if abs(old_val - new_val) > 0.01:
                    changes.append(ConfigChange(zone_id=zid, field=field, old_value=old_val, new_value=new_val))
            elif old_val != new_val:
                changes.append(ConfigChange(zone_id=zid, field=field, old_value=old_val, new_value=new_val))

    return changes


def call_gnn(config):
    """Call GNN server and return predictions."""
    try:
        resp = requests.post(GNN_URL, json=config, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get('predictions', []), None
    except requests.exceptions.ConnectionError:
        return None, "GNN server not reachable at " + GNN_URL
    except Exception as e:
        return None, str(e)


def compute_qos_score(predictions):
    """
    Compute a threshold-free composite QoS score across all zones.
    Lower is better. No targets needed — just raw quality.
    Weights: packet_loss (40%), delay (30%), throughput_deficit (20%), jitter (10%)
    """
    score = 0.0
    for p in predictions:
        loss = p['packet_loss_rate'] if isinstance(p, dict) else p.packet_loss_rate
        delay = p['avg_delay_ms'] if isinstance(p, dict) else p.avg_delay_ms
        throughput = p['throughput_mbps'] if isinstance(p, dict) else p.throughput_mbps
        jitter = p['jitter_ms'] if isinstance(p, dict) else p.jitter_ms
        bitrate = p['bitrate_mbps'] if isinstance(p, dict) else p.bitrate_mbps

        # Normalize each metric to a 0-1+ penalty
        loss_penalty     = loss * 40                          # 0-40
        delay_penalty    = (delay / 100.0) * 30              # 30 at 100ms
        tp_deficit       = max(0, bitrate - throughput) / max(bitrate, 1) * 20  # 0-20
        jitter_penalty   = min(jitter / 1.0, 1.0) * 10      # 0-10 capped at 1ms

        score += loss_penalty + delay_penalty + tp_deficit + jitter_penalty

    return round(score / len(predictions), 4)  # average across zones


def gemini_evaluate_and_refine(user_message, retrieved_docs, iteration_history):
    """
    Ask Gemini to evaluate all iteration results so far and decide:
    - Is the current best config good enough, or can it be improved?
    - If improvable, generate the next config to try.
    Returns (should_continue, reasoning, next_config_or_None)
    """
    context = "\n\n".join([f"--- {d['id']} ---\n{d['content']}" for d in retrieved_docs[:4]])

    # Format full iteration history
    history_lines = []
    for it in iteration_history:
        history_lines.append(f"\n--- Iteration {it['iteration']} (QoS score: {it['qos_score']}, lower=better) ---")
        history_lines.append(f"Config: {json.dumps(it['config'])}")
        history_lines.append("GNN Predictions:")
        for p in it['predictions']:
            zid = p['zone_id'] if isinstance(p, dict) else p.zone_id
            loss = p['packet_loss_rate'] if isinstance(p, dict) else p.packet_loss_rate
            delay = p['avg_delay_ms'] if isinstance(p, dict) else p.avg_delay_ms
            tp = p['throughput_mbps'] if isinstance(p, dict) else p.throughput_mbps
            jitter = p['jitter_ms'] if isinstance(p, dict) else p.jitter_ms
            loc = p['processing_location'] if isinstance(p, dict) else p.processing_location
            history_lines.append(
                f"  Zone {zid} ({loc}): throughput={tp:.3f} Mbps, loss={loss:.4f}, delay={delay:.2f}ms, jitter={jitter:.4f}ms"
            )

    history_text = "\n".join(history_lines)
    best = min(iteration_history, key=lambda x: x['qos_score'])

    prompt = f"""You are a network optimization expert for an airport surveillance system.

User's optimization goal: "{user_message}"

COMPLETE ITERATION HISTORY (all configs tried and their GNN-predicted QoS):
{history_text}

Best config so far: iteration {best['iteration']} with QoS score {best['qos_score']}

RELEVANT SIMULATION DATA:
{context}

NETWORK RULES:
- processing_location: ONLY "edge1", "edge2", "edge3", or "cloud"
- model_type: ONLY "lightweight", "medium", or "heavy"
- priority_class: ONLY 1, 2, or 3
- bitrate_mbps: float between 1.0 and 25.0
- Cloud link: 80 Mbps (bottleneck). Edge links: 150 Mbps each.

Analyze the iteration history and decide:
1. Is there a meaningful way to improve further, or has the config converged?
2. What specific changes would help (e.g. move congested cloud zones to edge, reduce bitrate, use lighter models)?

Output ONLY this JSON:
{{
  "should_continue": <true or false>,
  "reasoning": "<your analysis of the results and what you'd change>",
  "next_config": <full 6-zone config JSON to try next, or null if should_continue is false>
}}"""

    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()

    if '```json' in raw:
        raw = raw.split('```json')[1].split('```')[0].strip()
    elif '```' in raw:
        raw = raw.split('```')[1].split('```')[0].strip()

    result = json.loads(raw)
    should_continue = result.get('should_continue', False)
    reasoning = result.get('reasoning', '')
    next_config_raw = result.get('next_config')

    next_config = None
    if should_continue and next_config_raw:
        try:
            next_config = extract_json(json.dumps(next_config_raw))
        except Exception:
            should_continue = False

    return should_continue, reasoning, next_config


def retrieve_docs(query, n_results=10):
    """Retrieve relevant documents from ChromaDB."""
    results = collection.query(query_texts=[query], n_results=n_results)
    retrieved = []
    for i in range(len(results['ids'][0])):
        retrieved.append({
            'id': results['ids'][0][i],
            'content': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i] if results['distances'] else None,
        })
    return retrieved



def generate_config(user_message, current_config, retrieved_docs):
    """Use Gemini to generate a config from user intent + RAG context."""
    context = "\n\n".join([
        f"--- {d['id']} ---\n{d['content']}" for d in retrieved_docs
    ])

    prompt = f"""{SYSTEM_PROMPT}

CURRENT CONFIG:
{json.dumps(current_config, indent=2)}

RETRIEVED SIMULATION DATA (use this to make informed decisions):
{context}

User request: {user_message}

JSON output:"""

    response = gemini_model.generate_content(prompt)
    raw = response.text

    try:
        return extract_json(raw)
    except Exception:
        # Retry with simpler prompt
        retry_prompt = f"""Return ONLY valid JSON, no other text.
Current config: {json.dumps(current_config)}
User wants: {user_message}
Output the full modified JSON config with all 6 zones:"""
        response = gemini_model.generate_content(retry_prompt)
        return extract_json(response.text)


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Airport RAG + GNN Config Generator", version="1.0")


@app.on_event("startup")
def startup():
    global collection, gemini_model

    # --- Load ChromaDB ---
    if not os.path.exists(CHROMADB_PATH):
        print(f"❌ ChromaDB not found at {CHROMADB_PATH}")
        print("   Download it first (see README)")
        return

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_collection(
        name="airport_qos",
        embedding_function=embedding_fn,
    )
    print(f"✅ ChromaDB loaded: {collection.count()} documents from {CHROMADB_PATH}")

    # --- Setup Gemini ---
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️  Set GEMINI_API_KEY env var or edit rag_server.py")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        print("✅ Gemini model ready")


@app.get("/health")
def health():
    return {
        "status": "ok" if (collection and gemini_model) else "not_ready",
        "chromadb_docs": collection.count() if collection else 0,
        "gemini": "ready" if gemini_model else "not_configured",
        "gnn_url": GNN_URL,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main endpoint: natural language → config + GNN predictions.

    Send a message like:
      "Use heavy model for zone 3 and move it to cloud"
      "Optimize all zones for minimum delay"
      "Reduce bitrate on zone 0 to 3.0"

    Optionally pass current_config. If not provided, uses default.
    """
    if not collection or not gemini_model:
        raise HTTPException(status_code=503, detail="RAG system not ready")

    current = req.current_config or DEFAULT_CONFIG

    # Step 1: Retrieve relevant docs
    docs = retrieve_docs(req.message, n_results=10)

    # Step 2: Generate config via Gemini
    try:
        config = generate_config(req.message, current, docs)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to generate valid config: {str(e)}")

    # Step 3: Find changes
    changes = find_changes(current, config)

    # Step 4: Call GNN for predictions
    predictions, gnn_error = call_gnn(config)

    return ChatResponse(
        status="ok",
        generated_config=config,
        gnn_predictions=[ZonePrediction(**p) for p in predictions] if predictions else None,
        changes=changes,
        gnn_error=gnn_error,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Ask a question about the knowledge base (no config generation).

    Example: "What happens when too many zones use cloud processing?"
    """
    if not collection or not gemini_model:
        raise HTTPException(status_code=503, detail="RAG system not ready")

    docs = retrieve_docs(req.question, n_results=8)
    context = "\n\n".join([
        f"--- {d['id']} ---\n{d['content']}" for d in docs
    ])

    prompt = f"""You are an expert network engineer analyzing an airport surveillance camera network.
The network has 60 cameras across 6 zones, connected through zone switches to a core router,
which connects to 3 edge servers (150 Mbps each) and 1 cloud server (80 Mbps link).

Based ONLY on the following retrieved simulation data, answer the user's question.
Be specific with numbers. If the data doesn't contain enough information, say so.

=== RETRIEVED DATA ===
{context}
=== END DATA ===

User Question: {req.question}

Answer:"""

    response = gemini_model.generate_content(prompt)

    sources = []
    for d in docs[:5]:
        sources.append({
            "id": d['id'],
            "similarity": round(1 - d['distance'], 3) if d['distance'] else None,
            "scenario_id": d['metadata'].get('scenario_id', ''),
            "zone_id": d['metadata'].get('zone_id', ''),
            "processing_location": d['metadata'].get('processing_location', ''),
        })

    return AskResponse(status="ok", answer=response.text, sources=sources)


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    """
    Gemini-driven iterative config optimization — no thresholds needed.

    The loop:
      1. Gemini generates an initial config from your message + RAG context
      2. GNN predicts QoS metrics for that config
      3. Gemini reviews ALL iteration history and decides:
         - Can the config be meaningfully improved?
         - If yes, it generates the next config to try
      4. Repeat up to max_iterations times
      5. Return the config with the best composite QoS score

    Example:
      {"message": "minimize packet loss for all zones", "max_iterations": 4}
    """
    global last_best_config

    if not collection or not gemini_model:
        raise HTTPException(status_code=503, detail="RAG system not ready")

    current = req.current_config or DEFAULT_CONFIG
    docs = retrieve_docs(req.message, n_results=10)

    iterations = []
    best_config = None
    best_predictions = None
    best_score = float("inf")
    gnn_error = None
    stopped_early = False
    config = current

    for i in range(1, req.max_iterations + 1):
        # Iteration 1: generate from user message; subsequent: use config Gemini chose
        if i == 1:
            try:
                config = generate_config(req.message, config, docs)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Initial config generation failed: {str(e)}")

        # Get GNN predictions for this config
        predictions, gnn_error = call_gnn(config)
        if predictions is None:
            break  # GNN unreachable

        # Compute threshold-free composite QoS score
        qos_score = compute_qos_score(predictions)

        # Build iteration record (without assessment yet)
        iter_record = {
            "iteration": i,
            "config": config,
            "predictions": predictions,
            "qos_score": qos_score,
        }

        # Track best
        if qos_score < best_score:
            best_score = qos_score
            best_config = config
            best_predictions = predictions

        # Ask Gemini to evaluate all results so far and decide next step
        should_continue = False
        reasoning = "Max iterations reached."
        next_config = None

        if i < req.max_iterations:
            try:
                # Convert IterationResult Pydantic objects to dicts before passing
                history_as_dicts = [
                    {
                        "iteration": it.iteration,
                        "config": it.config,
                        "predictions": [p.model_dump() for p in it.predictions],
                        "qos_score": it.qos_score,
                    }
                    for it in iterations
                ]
                history_as_dicts.append(iter_record)  # current iteration already a dict

                should_continue, reasoning, next_config = gemini_evaluate_and_refine(
                    req.message, docs, history_as_dicts
                )
            except Exception as e:
                reasoning = f"Gemini evaluation failed: {e}"
                should_continue = False

        iterations.append(
            IterationResult(
                iteration=i,
                config=config,
                predictions=[ZonePrediction(**p) for p in predictions],
                qos_score=qos_score,
                gemini_assessment=reasoning,
                should_continue=should_continue,
            )
        )

        if not should_continue:
            stopped_early = (i < req.max_iterations)
            break

        # Use Gemini's suggested next config for next loop
        config = next_config

    # Cache the best config for /deploy
    if best_config is not None:
        last_best_config = deepcopy(best_config)

    return OptimizeResponse(
        status="ok",
        best_config=best_config or config,
        best_predictions=[ZonePrediction(**p) for p in best_predictions] if best_predictions else None,
        best_qos_score=best_score,
        iterations=iterations,
        total_iterations=len(iterations),
        stopped_early=stopped_early,
        gnn_error=gnn_error,
    )

@app.post("/deploy", response_model=DeployResponse)
def deploy(req: DeployRequest):
    global last_best_config

    if last_best_config is None:
        raise HTTPException(
            status_code=409,
            detail="No best config available. Run /optimize first."
        )

    # reuse your validator; guarantees 6 zones + valid fields
    try:
        best = extract_json(json.dumps(last_best_config))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cached best_config invalid: {e}")

    payload = {
        "scenario_id": req.scenario_id or f"scenario_{uuid.uuid4().hex[:8]}",
        "simulation_time": req.simulation_time,
        "background_traffic_mbps": best["background_traffic_mbps"],
        "zones": sorted(best["zones"], key=lambda z: z["zone_id"]),
    }

    try:
        resp = requests.post(NS3_SIM_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return DeployResponse(
            status="ok",
            deployed_payload=payload,
            ns3_result=resp.json()
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail=f"ns-3 endpoint unreachable: {NS3_SIM_URL}")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="ns-3 request timed out")
    except requests.exceptions.HTTPError:
        raise HTTPException(status_code=502, detail=f"ns-3 error: {resp.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)