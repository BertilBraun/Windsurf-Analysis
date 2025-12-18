## GybeLock Analyzer — FAQ

### What is GybeLock supposed to do?

GybeLock helps you review **beach-shot windsurf footage** (shore camera, long zoom) by:
- Stabilizing shaky tele footage
- Detecting and tracking riders over time
- Letting you click a rider to get a smooth, focused view for review

### What footage is supported?

- GybeLock is designed for **beach-shot / shore-shot** videos (tele/zoom from land).
- Use **MP4**.
- GoPro/action-cam POV footage is **not supported / not the intended use case**.

### What is the ingress folder?

The ingress folder is a folder on your computer that GybeLock monitors. When you add a new MP4 video into that folder, GybeLock automatically uploads it for processing.

### Where do processed videos appear?

They appear on the Analyzer page under **Analyzed Videos**. When a job is finished, it shows as **Succeeded** and can be opened.

### Why can’t I open a video yet?

Only jobs with status **Succeeded** can be opened. If a video is still processing (e.g. tracking), wait until it completes.

### Why does the player say “VIDEO FILE NOT FOUND”?

The Analyzer opens videos from your local ingress folder. If the file can’t be located:
- Select the correct ingress folder again (Ingress → **Change folder**), and/or
- Make sure the video still exists in that folder (and wasn’t removed).

### I moved or renamed a video. What should I do?

If a moved/renamed file can’t be found, the player may show **VIDEO FILE NOT FOUND**.

Fix:
- Ensure you’ve selected the correct ingress folder, and
- Put the video back into the ingress folder (or the expected subfolder) so GybeLock can access it.

### Why does upload say “Video too long”?

There is a maximum supported video length. If your upload fails with “Video too long”, split the recording into shorter clips and upload those.

### Why was my upload skipped?

GybeLock deduplicates videos by checksum. If that exact video was already uploaded/processed, it may be skipped.

### Uploads are paused / stuck. What can I do?

- Open **Ingress** and check the error message.
- If uploads paused after a failure, use **Retry failed**.
- If you see a quota/free-job message, you’ve hit the current limit for your account.

### How do I see all keyboard controls?

In the Analyzer / Player, click **Shortcuts** to open the keyboard shortcuts modal.

### How do I delete my account?

- Open the Analyzer
- Click **Settings**
- Click **Delete account**

This will permanently delete your user and job mappings.
