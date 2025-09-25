import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import traci
import xml.etree.ElementTree as ET

# ------------------ CONFIG ------------------
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "simple_intersection.sumocfg")
ROU_FILE = os.path.join(SCRIPT_DIR, "simple_intersection.rou.xml")
MODEL_FILE = os.path.join(SCRIPT_DIR, "traffic_model_ffnn.keras")
PLOTS_FOLDER = os.path.join(SCRIPT_DIR, "plots_validation_fixed")

LANES = [
    "north_in_0", "north_in_1", "north_in_2",
    "south_in_0", "south_in_1", "south_in_2",
    "east_in_0", "east_in_1", "east_in_2",
    "west_in_0", "west_in_1", "west_in_2",
]
NUM_PHASES = 4
os.makedirs(PLOTS_FOLDER, exist_ok=True)

MAX_STEPS = 600  # cap steps per timeframe for faster runs

# ------------------ Load Model ------------------
model = tf.keras.models.load_model(MODEL_FILE, compile=False)

# ------------------ Parse timeframes ------------------
def get_timeframes(rou_file):
    tree = ET.parse(rou_file)
    root = tree.getroot()
    frames = []
    for flow in root.findall("flow"):
        begin = int(flow.get("begin", 0))
        end = int(flow.get("end", 0))
        if end > begin:
            frames.append((begin, end))
    # merge overlapping flows
    merged = []
    for (b, e) in sorted(set(frames)):
        if not merged or b >= merged[-1][1]:
            merged.append([b, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(b, e) for b, e in merged]

# ------------------ Features ------------------
def get_state():
    counts = np.array([traci.lane.getLastStepVehicleNumber(l) for l in LANES], dtype=np.float32) / 20.0
    queues = np.array([traci.lane.getLastStepHaltingNumber(l) for l in LANES], dtype=np.float32) / 20.0
    waits  = np.array([traci.lane.getWaitingTime(l) for l in LANES], dtype=np.float32) / 100.0

    phase = traci.trafficlight.getPhase("center")
    phase_vec = np.zeros(NUM_PHASES, dtype=np.float32)
    phase_vec[phase % NUM_PHASES] = 1.0

    return np.concatenate([counts, queues, waits, phase_vec]).astype(np.float32), waits, queues, phase

# ------------------ Run Simulation ------------------
def run_mode(mode="model", steps=500, begin=0):
    traci.start([
        "sumo", "-c", CONFIG_FILE,
        "--begin", str(begin),
        "--end", str(begin + steps)  # force SUMO to stop at MAX_STEPS
    ])
    MIN_GREEN = 5
    last_switch_step = -MIN_GREEN

    # fixed-time params
    CYCLE = [0, 1, 2, 3]
    DURATION = [15, 5, 15, 5]
    phase_index, phase_timer = 0, 0

    all_waits, all_queues = [], []

    for t in range(steps):
        state, waits, queues, current_phase = get_state()

        if mode == "model":
            state_input = state.reshape(1, -1)
            action_probs = model.predict(state_input, verbose=0)
            action = int(np.argmax(action_probs[0]))
            if t - last_switch_step >= MIN_GREEN and action != (current_phase % NUM_PHASES):
                traci.trafficlight.setPhase("center", action)
                last_switch_step = t
        else:
            traci.trafficlight.setPhase("center", CYCLE[phase_index])
            phase_timer += 1
            if phase_timer >= DURATION[phase_index]:
                phase_index = (phase_index + 1) % len(CYCLE)
                phase_timer = 0

        all_waits.append(np.mean(waits))
        all_queues.append(np.mean(queues))
        traci.simulationStep()

    traci.close()
    return all_waits, all_queues

# ------------------ Plotting ------------------
def plot_comparison(steps, model_metrics, fixed_metrics, label):
    x = np.arange(steps)

    plt.figure()
    plt.plot(x, model_metrics[0], label="Model Waiting")
    plt.plot(x, fixed_metrics[0], label="Fixed Waiting", linestyle="--")
    plt.xlabel("Step"); plt.ylabel("Avg Waiting Time (s)")
    plt.title(f"Waiting Time Comparison - {label}")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(PLOTS_FOLDER, f"waiting_{label}.png")); plt.close()

    plt.figure()
    plt.plot(x, model_metrics[1], label="Model Queue")
    plt.plot(x, fixed_metrics[1], label="Fixed Queue", linestyle="--")
    plt.xlabel("Step"); plt.ylabel("Avg Queue Length")
    plt.title(f"Queue Length Comparison - {label}")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(PLOTS_FOLDER, f"queue_{label}.png")); plt.close()

# ------------------ Main ------------------
if __name__ == "__main__":
    timeframes = get_timeframes(ROU_FILE)
    print("Detected timeframes:", timeframes)

    for (begin, end) in timeframes:
        steps = min(end - begin, MAX_STEPS)
        label = f"{begin}_{end}"

        print(f"\n=== Validating {label} ({steps} steps) ===")

        print("Model-based control...")
        model_metrics = run_mode("model", steps=steps, begin=begin)
        print(" Avg Wait:", np.mean(model_metrics[0]), "Avg Queue:", np.mean(model_metrics[1]))

        print("Fixed-time control...")
        fixed_metrics = run_mode("fixed", steps=steps, begin=begin)
        print(" Avg Wait:", np.mean(fixed_metrics[0]), "Avg Queue:", np.mean(fixed_metrics[1]))

        plot_comparison(steps, model_metrics, fixed_metrics, label)

    print(f"\n✅ Validation done. Plots saved in {PLOTS_FOLDER}")
