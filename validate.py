import os
import sys
import numpy as np
import traci
import tensorflow as tf
import matplotlib.pyplot as plt

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

CONFIG_FILE = r"C:\Users\Sanjay Ramalingam\Desktop\Vit sem 3\CAO\projecy\project test\simple_intersection.sumocfg"
MODEL_FILE = "traffic_model_4phases.h5"

model = tf.keras.models.load_model(MODEL_FILE)

def get_features():
    lanes = ["north_in_0", "south_in_0", "east_in_0", "west_in_0"]
    counts, queues, waits = [], [], []
    for lane in lanes:
        counts.append(traci.lane.getLastStepVehicleNumber(lane))
        queues.append(traci.lane.getLastStepHaltingNumber(lane))
        waits.append(traci.lane.getWaitingTime(lane))

    counts_arr = np.array(counts) / 20.0
    queues_arr = np.array(queues) / 20.0
    waits_arr = np.array(waits) / 100.0

    current_phase = traci.trafficlight.getPhase("center")
    phase_vec = np.zeros(4)
    phase_vec[current_phase % 4] = 1

    features = np.concatenate([counts_arr, queues_arr, waits_arr, phase_vec]).astype(np.float32)
    return features, counts, queues, waits, phase_vec, current_phase

def run_mode(mode="model", steps=200):
    traci.start(["sumo", "-c", CONFIG_FILE])
    MIN_GREEN = 5
    last_switch_step = -MIN_GREEN

    CYCLE = [0, 1, 2, 3]
    DURATION = [15, 5, 15, 5]
    phase_index = 0
    phase_timer = 0

    all_waits, all_queues = [], []
    lane_counts_history = {lane: [] for lane in ["N","S","E","W"]}
    lane_queues_history = {lane: [] for lane in ["N","S","E","W"]}
    phases_history = []

    for t in range(steps):
        state, counts, queues, waits, phase_vec, current_phase = get_features()

        if mode == "model":
            state_input = state.reshape(1, -1)
            action_probs = model.predict(state_input, verbose=0)
            action = int(np.argmax(action_probs))
            current_phase_mod = current_phase % 4
            if t - last_switch_step >= MIN_GREEN and action != current_phase_mod:
                traci.trafficlight.setPhase("center", action)
                last_switch_step = t
            applied_phase = traci.trafficlight.getPhase("center")
        else:
            traci.trafficlight.setPhase("center", CYCLE[phase_index])
            applied_phase = CYCLE[phase_index]
            phase_timer += 1
            if phase_timer >= DURATION[phase_index]:
                phase_index = (phase_index + 1) % len(CYCLE)
                phase_timer = 0

        all_waits.append(np.mean(waits))
        all_queues.append(np.mean(queues))
        lane_counts_history["N"].append(counts[0])
        lane_counts_history["S"].append(counts[1])
        lane_counts_history["E"].append(counts[2])
        lane_counts_history["W"].append(counts[3])
        lane_queues_history["N"].append(queues[0])
        lane_queues_history["S"].append(queues[1])
        lane_queues_history["E"].append(queues[2])
        lane_queues_history["W"].append(queues[3])
        phases_history.append(applied_phase)

        traci.simulationStep()

    traci.close()
    return all_waits, all_queues, lane_counts_history, lane_queues_history, phases_history

def plot_metrics(steps, metrics, save_folder="plots", mode="model"):
    os.makedirs(save_folder, exist_ok=True)
    steps_range = list(range(steps))

    all_waits, all_queues, lane_counts, lane_queues, phases = metrics

    plt.figure()
    plt.plot(steps_range, all_waits, label="Avg Waiting Time")
    plt.xlabel("Step"); plt.ylabel("Waiting Time (s)")
    plt.title(f"Avg Waiting Time ({mode})"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_folder, f"avg_waiting_time_{mode}.png"))

    plt.figure()
    plt.plot(steps_range, all_queues, label="Avg Queue Length", color='orange')
    plt.xlabel("Step"); plt.ylabel("Queue Length")
    plt.title(f"Avg Queue Length ({mode})"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_folder, f"avg_queue_length_{mode}.png"))

    plt.figure()
    for lane, counts in lane_counts.items():
        plt.plot(steps_range, counts, label=f"Lane {lane}")
    plt.xlabel("Step"); plt.ylabel("Vehicle Count")
    plt.title(f"Vehicle Count per Lane ({mode})"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_folder, f"vehicle_counts_{mode}.png"))

    plt.figure()
    for lane, q in lane_queues.items():
        plt.plot(steps_range, q, label=f"Lane {lane}")
    plt.xlabel("Step"); plt.ylabel("Queue Length")
    plt.title(f"Queue Length per Lane ({mode})"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_folder, f"queue_lengths_{mode}.png"))

    plt.figure()
    plt.plot(steps_range, phases, label="Signal Phase", color='green')
    plt.xlabel("Step"); plt.ylabel("Phase")
    plt.title(f"Signal Phase Over Time ({mode})"); plt.grid(True)
    plt.savefig(os.path.join(save_folder, f"signal_phase_{mode}.png"))

def print_avg_metrics(metrics, mode="model"):
    all_waits, all_queues, lane_counts, lane_queues, _ = metrics
    print(f"\n{mode.upper()} SIGNAL AVERAGES")
    print("Avg Waiting Time:", np.mean(all_waits))
    print("Avg Queue Length:", np.mean(all_queues))
    for lane in ["N","S","E","W"]:
        print(f"Avg Vehicle Count {lane}:", np.mean(lane_counts[lane]))
        print(f"Avg Queue Length {lane}:", np.mean(lane_queues[lane]))

if __name__ == "__main__":
    steps = 200
    save_folder = "plots"

    print("Running model-based traffic signal...")
    model_metrics = run_mode(mode="model", steps=steps)
    plot_metrics(steps, model_metrics, save_folder, mode="model")
    print_avg_metrics(model_metrics, mode="model")

    print("Running fixed-time traffic signal...")
    fixed_metrics = run_mode(mode="fixed", steps=steps)
    plot_metrics(steps, fixed_metrics, save_folder, mode="fixed")
    print_avg_metrics(fixed_metrics, mode="fixed")

    print("Validation completed for both model and fixed-time control.")
