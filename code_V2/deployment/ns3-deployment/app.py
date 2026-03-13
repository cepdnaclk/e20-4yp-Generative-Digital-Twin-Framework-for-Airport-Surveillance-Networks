import os
import json
import uuid
import tempfile
import subprocess
import threading
import statistics
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional

# Path to ns-3 installation
NS3_DIR = os.getenv("NS3_DIR", "/home/ubuntu/ns-allinone-3.41/ns-3.41")

# Prevent multiple simulations running at the same time
simulation_lock = threading.Lock()

# -------- Request Models -------- #

class Zone(BaseModel):
    zone_id: int
    bitrate_mbps: float
    priority_class: int
    model_type: Literal["lightweight", "medium", "heavy"]
    processing_location: Literal["edge1", "edge2", "edge3", "cloud"]

class Scenario(BaseModel):
    scenario_id: Optional[str] = None
    simulation_time: float
    background_traffic_mbps: float
    zones: List[Zone]

# -------- FastAPI App -------- #

app = FastAPI(title="NS3 Simulation API")

# -------- Aggregation Function -------- #

def aggregate_zone_metrics(sim_output):

    flows = sim_output.get("flows", [])

    if not flows:
        return {
            "status": "error",
            "message": "No flows found in simulation output"
        }

    zone_groups = defaultdict(list)

    for f in flows:
        zone_groups[f["zone_id"]].append(f)

    predictions = []

    for zone_id in sorted(zone_groups.keys()):

        items = zone_groups[zone_id]

        predictions.append({
            "zone_id": zone_id,
            "processing_location": items[0]["processing_location"],
            "model_type": items[0]["model_type"],
            "bitrate_mbps": items[0]["bitrate_configured_mbps"],

            "throughput_mbps": round(statistics.mean(i["throughput_mbps"] for i in items), 4),
            "packet_loss_rate": round(statistics.mean(i["packet_loss_rate"] for i in items), 4),
            "avg_delay_ms": round(statistics.mean(i["avg_delay_ms"] for i in items), 4),
            "jitter_ms": round(statistics.mean(i["jitter_ms"] for i in items), 4)
        })

    return {
        "status": "ok",
        "predictions": predictions
    }

# -------- Simulation Endpoint -------- #

@app.post("/simulate")
def simulate(payload: Scenario):

    data = payload.model_dump()

    # Generate scenario ID if not provided
    data["scenario_id"] = data.get("scenario_id") or f"api_{uuid.uuid4().hex[:8]}"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            input_path = os.path.join(temp_dir, "input.json")
            output_path = os.path.join(temp_dir, "output.json")

            # Save simulation input
            with open(input_path, "w") as f:
                json.dump(data, f)

            cmd = [
                "./ns3",
                "run",
                f"scratch/simulation --input={input_path} --output={output_path}"
            ]

            print("Running command:", cmd)

            # Ensure only one simulation runs at a time
            with simulation_lock:

                proc = subprocess.run(
                    cmd,
                    cwd=NS3_DIR,
                    capture_output=True,
                    text=True,
                    timeout=600
                )

            print("stdout:", proc.stdout)
            print("stderr:", proc.stderr)

            if proc.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Simulation failed",
                        "stdout": proc.stdout,
                        "stderr": proc.stderr
                    }
                )

            # Read simulation output
            if not os.path.exists(output_path):
                raise HTTPException(
                    status_code=500,
                    detail="Simulation finished but output file not found"
                )

            with open(output_path) as f:
                sim_output = json.load(f)

            # Convert flow output ? zone predictions
            result = aggregate_zone_metrics(sim_output)

            return result

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Simulation timed out"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )