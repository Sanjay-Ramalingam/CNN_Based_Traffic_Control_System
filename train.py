import os
import sys
import numpy as np
import traci
import tensorflow as tf
from tensorflow.keras import layers, models


if "SUMO_HOME" not in os.environ:
    if os.name == "nt":  # Windows
        default_sumo = r"C:\Program Files (x86)\Eclipse\Sumo"
    else:  # Linux / Mac
        default_sumo = "/usr/share/sumo"
    os.environ["SUMO_HOME"] = default_sumo

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "simple_intersection.sumocfg")

def build_model(input_dim, output_dim):
    model = models.Sequential()
    model.add(layers.Dense(128, activation="relu", input_shape=(input_dim,)))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(output_dim, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def get_state():
    lanes = ["north_in_0", "south_in_0", "east_in_0", "west_in_0"]
    counts, queues, waits = [], [], []

    for lane in lanes:
        counts.append(traci.lane.getLastStepVehicleNumber(lane))
        queues.append(traci.lane.getLastStepHaltingNumber(lane))
        waits.append(traci.lane.getWaitingTime(lane))

    counts = np.array(counts) / 20.0
    queues = np.array(queues) / 20.0
    waits = np.array(waits) / 100.0

    phase = traci.trafficlight.getPhase("center")
    phase_vec = np.zeros(4)
    phase_vec[phase % 4] = 1

    return np.concatenate([counts, queues, waits, phase_vec]).astype(np.float32)

def collect_data(episodes=5, steps=200):
    X, y = [], []
    for ep in range(episodes):
        traci.start(["sumo", "-c", CONFIG_FILE])
        for t in range(steps):
            state = get_state()

            ns_total = np.sum(state[0:2]) + np.sum(state[4:6])
            ew_total = np.sum(state[2:4]) + np.sum(state[6:8])

            if ns_total >= ew_total:
                action = 0 if state[0] + state[4] >= state[1] + state[5] else 1
            else:
                action = 2 if state[2] + state[6] >= state[3] + state[7] else 3

            label = np.zeros(4)
            label[action] = 1

            X.append(state)
            y.append(label)

            traci.simulationStep()
        traci.close()
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = collect_data()
    print("Dataset shape:", X.shape, y.shape)

    model = build_model(input_dim=X.shape[1], output_dim=y.shape[1])
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

    model_path = os.path.join(SCRIPT_DIR, "traffic_model_4phases.h5")
    model.save(model_path)
    print(f"Model trained and saved as {model_path}")
