import os
import sys
import traci
import tensorflow as tf
import numpy as np
import random
from collections import defaultdict

# -------------------------------
# SUMO Setup - adjust these paths to your environment
# -------------------------------
if "SUMO_HOME" not in os.environ:
    # try a reasonable default (Windows example)
    os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

sumo_cfg = r"C:\Users\Sanjay Ramalingam\Desktop\CNN_Based_Traffic_Control_System\simple_intersection.sumocfg"
model_path = r"C:\Users\Sanjay Ramalingam\Desktop\CNN_Based_Traffic_Control_System\traffic_model_4phases.h5"
sumoBinary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui.exe")

# -------------------------------
# Load model
# -------------------------------
model = tf.keras.models.load_model(model_path)
print("Loaded model; input shape:", model.input_shape)
model.summary()

# -------------------------------
# Parameters
# -------------------------------
MIN_PHASE_DURATION = 2  # keep phase active at least this many simulation steps
MAX_STEPS = 3600
EPSILON = 0.05         # smaller exploration by default; you can decay this
DEBUG_EVERY = 100      # print debug info every N steps
LANES = ["north_in_0", "south_in_0", "east_in_0", "west_in_0"]  # must match training
PHASE_VECTOR_SIZE = 4  # training used 4-phase one-hot
# Normalization constants used in training
CNT_NORM = 20.0
QUEUE_NORM = 20.0
WAIT_NORM = 100.0

# -------------------------------
# Utilities: build state exactly like training code
# -------------------------------
def get_state_for_intersection(tls_id):
    """
    Build the 16-d state vector used during training:
      counts(4) normalized, queues(4) normalized, waits(4) normalized, phase_one_hot(4)
    Returns shape (1,16) dtype float32 ready for model.predict
    """
    counts, queues, waits = [], [], []
    for lane in LANES:
        counts.append(traci.lane.getLastStepVehicleNumber(lane))
        queues.append(traci.lane.getLastStepHaltingNumber(lane))
        # getWaitingTime returns total waiting time for lane; we will average later
        waits.append(traci.lane.getWaitingTime(lane))

    counts = np.array(counts, dtype=np.float32) / CNT_NORM
    queues = np.array(queues, dtype=np.float32) / QUEUE_NORM
    waits = np.array(waits, dtype=np.float32) / WAIT_NORM

    phase = traci.trafficlight.getPhase(tls_id)
    phase_vec = np.zeros(PHASE_VECTOR_SIZE, dtype=np.float32)
    # replicate training behavior: used phase % 4
    phase_vec[phase % PHASE_VECTOR_SIZE] = 1.0

    state = np.concatenate([counts, queues, waits, phase_vec]).astype(np.float32)
    return state.reshape(1, -1)  # shape (1,16)

# -------------------------------
# Utilities: map model action (0..3) to actual SUMO phase index robustly
# -------------------------------
def action_to_phase_index(tls_id, action):
    """
    Map action in {0,1,2,3} (used at training time) to a real SUMO phase index.
    Strategy:
      - Try to inspect program phases. If there are exactly 4 phases, map 1:1.
      - If more phases exist (e.g., green + yellow entries), assume green phases are at even indices:
          main_phases = [0, 2, 4, 6, ...] and map action->main_phases[action].
      - If that fails, fall back to action % num_phases (safe fallback).
    """
    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phases = program.phases
    n = len(phases)
    if n == PHASE_VECTOR_SIZE:
        return action
    # try even indices (common pattern green then yellow)
    even_indices = list(range(0, n, 2))
    if len(even_indices) >= PHASE_VECTOR_SIZE:
        return even_indices[action]
    # fallback: return modulo
    return action % n

# -------------------------------
# Metrics collector for diagnostics
# -------------------------------
metrics = defaultdict(float)
metrics_counts = 0

# -------------------------------
# Run simulation
# -------------------------------
traci.start([sumoBinary, "-c", sumo_cfg], numRetries=10, label="sim1")
tls_ids = traci.trafficlight.getIDList()
tls_phase_count = {tls: len(traci.trafficlight.getAllProgramLogics(tls)[0].phases) for tls in tls_ids}
phase_timer = {tls: 0 for tls in tls_ids}
step = 0

try:
    # replicate training semantics: for each step, observe -> choose -> set -> simulationStep
    while step < MAX_STEPS:
        # For every traffic light, decide an action (if allowed by min phase duration)
        for tls_id in tls_ids:
            if phase_timer[tls_id] < MIN_PHASE_DURATION:
                phase_timer[tls_id] += 1
                continue

            # build state identical to training
            state = get_state_for_intersection(tls_id)  # (1,16)

            # Prediction (catch shape/type issues)
            try:
                action_probs = model.predict(state, verbose=0)
                # model.predict returns shape (1, output_dim)
                best_action = int(np.argmax(action_probs[0]))
            except Exception as e:
                print(f"[Warning] model prediction failed at step {step}: {e}")
                # fallback: keep current phase
                best_action = traci.trafficlight.getPhase(tls_id) % PHASE_VECTOR_SIZE

            # epsilon-greedy exploration (small)
            if random.random() < EPSILON:
                best_action = random.randint(0, PHASE_VECTOR_SIZE - 1)

            # map to SUMO phase index (robust mapping)
            best_phase_index = action_to_phase_index(tls_id, best_action)

            # apply the chosen phase
            traci.trafficlight.setPhase(tls_id, int(best_phase_index))
            phase_timer[tls_id] = 0

        # step the simulation after applying phases (same order as training data collection)
        traci.simulationStep()
        step += 1

        # gather lightweight metrics (global)
        # sum queue length and average waiting time across lanes to monitor performance
        total_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES)
        total_wait = sum(traci.lane.getWaitingTime(l) for l in LANES)
        metrics["sum_queue"] += total_queue
        metrics["sum_wait"] += total_wait
        metrics_counts += 1

        if step % DEBUG_EVERY == 0:
            avg_q = metrics["sum_queue"] / metrics_counts
            avg_w = metrics["sum_wait"] / metrics_counts
            print(f"[Step {step}] avg_queue={avg_q:.3f}, avg_wait={avg_w:.3f}")
            # reset diagnostics summary to get windowed stats
            metrics = defaultdict(float)
            metrics_counts = 0

finally:
    traci.close()
    print("Simulation finished.")
