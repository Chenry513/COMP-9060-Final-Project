#load and preprocess EMAKI dataset
import numpy as np
import pandas as pd

#load emaki dataframe from pkl file
def load_emaki(pkl_path: str = "EMAKI_utt.pkl") -> pd.DataFrame:
    #read the pkl file
    df = pd.read_pickle(pkl_path)

    #sort by user, trial, timestamp so order of actions is correct
    df = df.sort_values(["user", "trial", "timestamp"]).reset_index(drop=True)

    return df


#helper to bucket mouse coordinates into a coarse grid
def bucket_coord(x, y, max_res_x, max_res_y, num_bins: int = 10) -> str:
    #handle nan values
    if np.isnan(x) or np.isnan(y):
        return "UNKPOS"

    #scale into [0, num_bins)
    bx = int(np.clip(x / (max_res_x + 1e-9) * num_bins, 0, num_bins - 1))
    by = int(np.clip(y / (max_res_y + 1e-9) * num_bins, 0, num_bins - 1))

    return f"{bx}_{by}"


#convert a single row (event) into a token string
def row_to_token(row, max_res_x, max_res_y, num_bins: int = 10) -> str:
    t = row["type"]

    #mouse moves get a bucketed position token
    if t == "mousemove":
        pos = bucket_coord(row["X"], row["Y"], max_res_x, max_res_y, num_bins)
        return f"MMOVE_{pos}"

    #mouse button actions
    if t in ["mousedown", "mouseup"]:
        button = "UNK"
        if row["value"] == 1:
            button = "LEFT"
        elif row["value"] == 3:
            button = "RIGHT"
        return f"{t.upper()}_{button}"

    #keyboard actions
    if t in ["keydown", "keyup"]:
        key = str(row["value"])
        #replace spaces and uppercase for consistency
        key = key.replace(" ", "_").upper()
        return f"{t.upper()}_{key}"

    #fallback token
    return "UNK"


#build token windows and labels for task recognition
def build_windows(
    df: pd.DataFrame,
    window_size: int = 50,
    stride: int = 25,
    num_bins: int = 10,
):
    #max resolution values across users
    max_res_x = df["resolutionX"].max()
    max_res_y = df["resolutionY"].max()

    #lists to store output
    all_sequences = []   #each element is a list of token strings
    all_labels = []      #task label for each window
    window_users = []    #user id for each window (for splitting later)

    #group by user and trial since each trial has a single task label
    for (user, trial), group in df.groupby(["user", "trial"]):
        #task label for this trial
        task_label = group["task"].iloc[0]

        #convert every row to a token
        tokens = [
            row_to_token(row, max_res_x, max_res_y, num_bins)
            for _, row in group.iterrows()
        ]

        #slide a window over the tokens
        for start in range(0, max(0, len(tokens) - window_size + 1), stride):
            window = tokens[start:start + window_size]

            #only keep full windows
            if len(window) == window_size:
                all_sequences.append(window)
                all_labels.append(task_label)
                window_users.append(user)

    #convert labels and users to numpy arrays
    all_labels = np.array(all_labels)
    window_users = np.array(window_users)

    return all_sequences, all_labels, window_users