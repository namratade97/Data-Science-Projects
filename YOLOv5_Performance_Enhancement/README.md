# YOLOv5 Performance Enhancement

This project improves YOLOv5 for video-based action detection:

- Applied on a dataset curated for Fall Detection in humans.
- The algorithm uses classifications of the most recent frames in consideration while predicting a new class for the current frame.
- The original YOLOv5 model has a problem of label flickering between similar actions in a video frame (like "Standing" vs. "Walking"). Our approach reduces it significantly (Please refer to the attached video samples for example).
