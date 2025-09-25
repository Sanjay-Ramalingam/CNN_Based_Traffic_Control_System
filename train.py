import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import traci

# ------------------ SUMO setup ------------------
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "simple_intersection.sumocfg")
MODEL_FILE = os.path.join(SCRIPT_DIR, "traffic_model_ffnn.keras")

# Incoming lanes (12 total: 3 per direction)
LANES = [
    "north_in_0", "north_in_1", "north_in_2",
    "south_in_0", "south_in_1", "south_in_2",
    "east_in_0", "east_in_1", "east_in_2",
    "west_in_0", "west_in_1", "west_in_2",
]

NUM_PHASES = 4  # N-S straight, N-S left, E-W straight, E-W left (assumed)

# ------------------ Build Feedforward Neural Network ------------------
def build_ffnn(input_dim, output_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(output_dim, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ------------------ Extract state features ------------------
def get_state():
    counts = np.array([traci.lane.getLastStepVehicleNumber(l) for l in LANES], dtype=np.float32) / 20.0
    queues = np.array([traci.lane.getLastStepHaltingNumber(l) for l in LANES], dtype=np.float32) / 20.0
    waits  = np.array([traci.lane.getWaitingTime(l) for l in LANES], dtype=np.float32) / 100.0

    phase = traci.trafficlight.getPhase("center")
    phase_vec = np.zeros(NUM_PHASES, dtype=np.float32)
    phase_vec[phase % NUM_PHASES] = 1.0

    return np.concatenate([counts, queues, waits, phase_vec]).astype(np.float32)

# ------------------ Collect training data ------------------
def collect_data(episodes=10, steps=300):
    X, y = [], []
    for ep in range(episodes):
        traci.start(["sumo", "-c", CONFIG_FILE])
        for t in range(steps):
            state = get_state()

            # Simple heuristic: compare total N/S vs E/W load
            ns_total = np.sum(state[0:6]) + np.sum(state[12:18])   # counts + queues N/S
            ew_total = np.sum(state[6:12]) + np.sum(state[18:24])  # counts + queues E/W

            if ns_total >= ew_total:
                action = 0  # keep N-S
            else:
                action = 2  # keep E-W

            label = np.zeros(NUM_PHASES, dtype=np.float32)
            label[action] = 1.0

            X.append(state)
            y.append(label)

            traci.simulationStep()
        traci.close()
        print(f" Episode {ep+1}/{episodes} finished.")
    return np.array(X), np.array(y)

# ------------------ Train FFNN ------------------
if __name__ == "__main__":
    print("Collecting training data...")
    X, y = collect_data(episodes=15, steps=300)
    print("Dataset shape:", X.shape, y.shape)

    model = build_ffnn(input_dim=X.shape[1], output_dim=y.shape[1])

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
    ]

    history = model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=cb,
        verbose=1
    )

    model.save(MODEL_FILE)
    print(f"✅ Feedforward NN trained and saved as {MODEL_FILE}")
