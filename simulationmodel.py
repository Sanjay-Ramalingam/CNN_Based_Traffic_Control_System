import os
import sys
import traci
import tensorflow as tf
import numpy as np
import random

# -------------------------------
# SUMO-GUI setup
# -------------------------------
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

sumo_cfg = r"C:\Users\Sanjay Ramalingam\Desktop\CNN_Based_Traffic_Control_System\simple_intersection.sumocfg"
model_path = r"C:\Users\Sanjay Ramalingam\Desktop\CNN_Based_Traffic_Control_System\traffic_model_ffnn.keras"
sumoBinary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui.exe")  # GUI mode

# -------------------------------
# Load trained FFNN model
# -------------------------------
model = tf.keras.models.load_model(model_path)
print("✅ Loaded model; input shape:", model.input_shape)

# -------------------------------
# Parameters
# -------------------------------
MIN_PHASE_DURATION = 5
MAX_STEPS = 1800
EPSILON = 0.05
LANES = [
    "north_in_0", "north_in_1", "north_in_2",
    "south_in_0", "south_in_1", "south_in_2",
    "east_in_0", "east_in_1", "east_in_2",
    "west_in_0", "west_in_1", "west_in_2",
]
NUM_PHASES = 4
CNT_NORM, QUEUE_NORM, WAIT_NORM = 20.0, 20.0, 100.0

# -------------------------------
# State vector function
# -------------------------------
def get_state(tls_id):
    counts = np.array([traci.lane.getLastStepVehicleNumber(l) for l in LANES], dtype=np.float32) / CNT_NORM
    queues = np.array([traci.lane.getLastStepHaltingNumber(l) for l in LANES], dtype=np.float32) / QUEUE_NORM
    waits  = np.array([traci.lane.getWaitingTime(l) for l in LANES], dtype=np.float32) / WAIT_NORM

    phase = traci.trafficlight.getPhase(tls_id)
    phase_vec = np.zeros(NUM_PHASES, dtype=np.float32)
    phase_vec[phase % NUM_PHASES] = 1.0

    return np.concatenate([counts, queues, waits, phase_vec]).reshape(1, -1)  # shape (1,40)

# -------------------------------
# Map model action → SUMO phase index
# -------------------------------
def action_to_phase_index(tls_id, action):
    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phases = program.phases
    n = len(phases)
    if n == NUM_PHASES:
        return action
    even_indices = list(range(0, n, 2))
    if len(even_indices) >= NUM_PHASES:
        return even_indices[action]
    return action % n

# -------------------------------
# Start SUMO-GUI
# -------------------------------
traci.start([sumoBinary, "-c", sumo_cfg,
             "--step-length", "1",
             "--quit-on-end", "false",
             "--duration-log.disable"])

# -------------------------------
# Prepare entry edges
# -------------------------------
entry_edges = ['north_in', 'south_in', 'east_in', 'west_in']
tls_ids = traci.trafficlight.getIDList()
phase_timer = {tls: 0 for tls in tls_ids}

vehicle_counter = 0
added_routes = set()
step = 0

# -------------------------------
# Simulation loop
# -------------------------------
try:
    while step < MAX_STEPS:
        # Add vehicles every 5 steps on each entry edge
        if step % 5 == 0:
            for edge in entry_edges:
                route_id = f"route_{edge}"
                if route_id not in added_routes:
                    traci.route.add(route_id, [edge])
                    added_routes.add(route_id)
                veh_id = f"veh_{vehicle_counter}"
                traci.vehicle.add(veh_id, route_id, typeID="car", depart=step)
                vehicle_counter += 1
                print(f"Added vehicle {veh_id} on edge {edge} at step {step}")

        # Traffic light control using trained model
        for tls_id in tls_ids:
            if phase_timer[tls_id] < MIN_PHASE_DURATION:
                phase_timer[tls_id] += 1
                continue

            state = get_state(tls_id)
            action_probs = model.predict(state, verbose=0)
            best_action = int(np.argmax(action_probs[0]))

            if random.random() < EPSILON:
                best_action = random.randint(0, NUM_PHASES - 1)

            best_phase_index = action_to_phase_index(tls_id, best_action)
            traci.trafficlight.setPhase(tls_id, int(best_phase_index))
            phase_timer[tls_id] = 0

        traci.simulationStep()
        step += 1

        # Debug info
        active_vehicles = traci.simulation.getMinExpectedNumber()
        print(f"Step {step}/{MAX_STEPS} — Active vehicles: {active_vehicles}")

finally:
    traci.close()
    print("✅ Simulation finished successfully")
