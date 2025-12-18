## GybeLock Analyzer — Usage Tutorial

GybeLock’s Analyzer is designed for **beach-shot windsurf footage**: a camera (typically long-zoom/tele) filming riders from shore.

It is **not** designed for GoPro/action-cam POV footage.

### 1) Set the ingress folder

- Open the Analyzer.
- Click **Ingress** (bottom-right).
- Click **Select folder** and choose the folder where you will place your beach-shot MP4 videos.
- Make sure the status shows **Monitoring** (or uploading if you just added videos).

### 2) Drop MP4s into the folder

- Copy or move your `.mp4` files into the ingress folder (subfolders are fine).
- GybeLock will automatically detect new files and start uploading.
- You can open **Ingress** anytime to see upload progress and errors.

### 3) Processing

After upload, each video becomes a “job” and moves through processing stages (shown on the thumbnail), e.g.:
- Orienting video
- Stabilizing video
- Detecting surfers
- Surfer identification
- Tracking surfers

When processing finishes, the job becomes **Succeeded** and can be opened.

### 4) Open a finished video

- In **Analyzed Videos**, find a tile with status **Succeeded**.
- Click the tile to open the player.

If you see **VIDEO FILE NOT FOUND**, re-check that:
- You selected the correct ingress folder, and
- The video still exists in that folder at its expected path.

### 5) Review a track

- In the player, you start in **overview** mode.
- Move your mouse over the rider to highlight the detection.
- Click the rider to switch into a focused (cropped) view for that track.
- Use the timeline to seek.

### Tip: open the Shortcuts modal

In the Analyzer / Player, click **Shortcuts** to see all keyboard controls (play/pause, frame stepping, track navigation, and switching between videos).

### Export and Report

- **Export** lets you export a focused clip for the currently selected track.
- **Report** lets you report an analysis issue (include an approximate timestamp if possible).
