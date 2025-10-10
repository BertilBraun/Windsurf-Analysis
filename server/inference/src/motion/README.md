![Pre](Screenshot%202025-09-23%20204439.png)

- Blue line: Global Camera Motion trajectory
- Pink arrows: Kalman Filter velocity vector
- Pink BBox: Last Kalman Filter bounding box
- Yellow BBox: Last Detection bounding box
- Red BBox: Current Detection bounding box
- Blue BBox: Current Kalman Filter bounding box
- Green BBox: Current Kalman Filter + Camera Motion compensated bounding box

![Post](Screenshot%202025-09-23%20204833.png)

Track 2 is now missing (no detection was found for it for some frames).
Kalman Filter predicted the track's trajectory and compensated for the camera motion. Because the prediction is uncertain, the bbox is inflated and is growing with increasing miss count.
