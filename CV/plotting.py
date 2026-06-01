from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def compute_centers(csv_path, *, frame_col="frame", x1="x1", y1="y1", x2="x2", y2="y2",
                    save=False, out_path=None):

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = {frame_col, x1, y1, x2, y2}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    df = df.copy()
    df["cx"] = (df[x1] + df[x2]) / 2.0
    df["cy"] = (df[y1] + df[y2]) / 2.0

    if save:
        if out_path is None:
            out_path = csv_path.parent / f"preprocessed_{csv_path.stem}.csv"
        else:
            out_path = Path(out_path)

        df.to_csv(out_path, index=False)

    return df


def animate_centers(df, *, frame_col="frame", x_col="cx", y_col="cy",
                    frame_step=2, interval_ms=30, invert_y=False, s=15):

    required = {frame_col, x_col, y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {sorted(missing)}")

    df = df.sort_values(frame_col)
    frames = sorted(df[frame_col].unique())
    if frame_step and frame_step > 1:
        frames = frames[::frame_step]

    fig, ax = plt.subplots()
    ax.set_xlim(df[x_col].min() - 10, df[x_col].max() + 10)
    ax.set_ylim(df[y_col].min() - 10, df[y_col].max() + 10)
    if invert_y:
        ax.invert_yaxis()

    scat = ax.scatter([], [], s=s)
    title = ax.set_title("")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        title.set_text("")
        return scat, title

    def update(i):
        f = frames[i]
        g = df[df[frame_col] == f]
        scat.set_offsets(g[[x_col, y_col]].to_numpy())
        title.set_text(f"{frame_col}: {f} | points: {len(g)}")
        return scat, title

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init,
                        interval=interval_ms, blit=False)
    plt.show()
    return fig, ax, ani

#preprocess the csv and add speed column
def prepare_tracks_with_speed(
    csv_path,
    *,
    id_col="track_id",
    fps=30.0,                 # IMPORTANT: set this to your video/frame extraction FPS
    smooth_window=5,          # 0 or 1 = no smoothing; try 5, 7, 9 for smoother paths
    interp_method="linear",   # "linear" is usually fine
    max_interp_gap=None,      # e.g. 10 to avoid bridging long missing gaps; None = allow all
):

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    save_csv_path = csv_path.parent / f"preprocessed_{csv_path.stem}.csv"

    required = {"frame", "x1", "y1", "x2", "y2", id_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    df = df.copy()
    df["frame"] = df["frame"].astype(int)

    # Centers
    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0

    out = []

    for tid, g in df.sort_values(["frame"]).groupby(id_col, sort=False):
        g = g.sort_values("frame").set_index("frame")

        # Reindex to fill missing frames within this track's span
        full_idx = np.arange(g.index.min(), g.index.max() + 1)
        g2 = g.reindex(full_idx)

        # Keep the track id for newly created rows
        g2[id_col] = tid

        # Interpolate centers
        g2["cx"] = g2["cx"].interpolate(method=interp_method)
        g2["cy"] = g2["cy"].interpolate(method=interp_method)

        # Optional: prevent interpolation across very large gaps
        if max_interp_gap is not None:
            # Identify original observed frames
            observed = g.index.to_numpy()
            # Mark segments between observed frames that are too far apart
            # We will set interpolated points in those long gaps to NaN
            for a, b in zip(observed[:-1], observed[1:]):
                if (b - a) > max_interp_gap:
                    gap_idx = np.arange(a + 1, b)  # frames strictly between
                    g2.loc[gap_idx, ["cx", "cy"]] = np.nan

        # Smoothing (rolling mean on interpolated coordinates)
        if smooth_window is None or smooth_window <= 1:
            g2["cx_smooth"] = g2["cx"]
            g2["cy_smooth"] = g2["cy"]
        else:
            g2["cx_smooth"] = g2["cx"].rolling(smooth_window, center=True, min_periods=1).mean()
            g2["cy_smooth"] = g2["cy"].rolling(smooth_window, center=True, min_periods=1).mean()

        # Speed (px/s) from smoothed coords
        dx = g2["cx_smooth"].diff()
        dy = g2["cy_smooth"].diff()
        dist_px = np.sqrt(dx * dx + dy * dy)          # px per frame
        g2["speed_px_s"] = dist_px * float(fps)       # px/s

        g2 = g2.reset_index().rename(columns={"index": "frame"})
        out.append(g2[["frame", id_col, "cx", "cy", "cx_smooth", "cy_smooth", "speed_px_s"]])

    result = pd.concat(out, ignore_index=True)
    result.to_csv(save_csv_path, index=False)
    return result