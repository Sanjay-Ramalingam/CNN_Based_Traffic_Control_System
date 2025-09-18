import os
import sys
import traci
import tensorflow as tf
import numpy as np
import random

# -------------------------------
# SUMO Setup
# -------------------------------
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

# Full paths to your files
sumo_cfg = r"C:\Users\Sanjay Ramalingam\Desktop\CNN_Based_Traffic_Control_System\simple_intersection.sumocfg"
model_path = r"C:\Users\Sanjay Ramalingam\Desktop\CNN_Based_Traffic_Control_System\traffic_model_4phases.h5"

# Load trained CNN model
model = tf.keras.models.load_model(model_path)

# Full path to SUMO GUI executable
sumoBinary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui.exe")

# Start SUMO via TraCI
traci.start([sumoBinary, "-c", sumo_cfg], numRetries=10, label="sim1")

# Get traffic light IDs and phase counts
tls_ids = traci.trafficlight.getIDList()
tls_phase_count = {tls: len(traci.trafficlight.getAllProgramLogics(tls)[0].phases) for tls in tls_ids}

# Simulation parameters
MIN_PHASE_DURATION = 10  # minimum steps to keep a phase active
phase_timer = {tls: 0 for tls in tls_ids}
MAX_STEPS = 3600
EPSILON = 0.1  # chance to explore random phase
step = 0

# Define main corridors (adjust according to your network)
lanes_NS = ["north_in_0", "south_in_0"]
lanes_EW = ["east_in_0", "west_in_0"]

try:
    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1

        for tls_id in tls_ids:
            # Skip phase change if minimum duration not reached
            if phase_timer[tls_id] < MIN_PHASE_DURATION:
                phase_timer[tls_id] += 1
                continue

            # Aggregate features per corridor for CNN
            features = []
            for corridor in [lanes_NS, lanes_EW]:
                vehicle_count = sum(traci.lane.getLastStepVehicleNumber(l) for l in corridor)
                queue_length = sum(traci.lane.getLastStepHaltingNumber(l) for l in corridor)
                avg_wait = np.mean([traci.lane.getWaitingTime(l) for l in corridor])
                features.extend([vehicle_count, queue_length, avg_wait])

            state = np.array(features).reshape(1, -1)

            try:
                # CNN predicts best phase
                action_probs = model.predict(state, verbose=0)
                best_phase = int(np.argmax(action_probs)) % tls_phase_count[tls_id]
            except Exception as e:
                print(f"Step {step}: CNN prediction error: {e}")
                best_phase = traci.trafficlight.getPhase(tls_id)

            # Epsilon-greedy exploration to prevent starvation
            if random.random() < EPSILON:
                best_phase = random.randint(0, tls_phase_count[tls_id]-1)

            # Apply phase using SUMO tlLogic
            traci.trafficlight.setPhase(tls_id, best_phase)
            phase_timer[tls_id] = 0

finally:
    traci.close()
    print("Simulation finished.")
