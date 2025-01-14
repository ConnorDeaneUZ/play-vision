### TODO
1. gather clips of goals and no goals.

dataset/
├── goal/
│   ├── goal_1.mp4
│   ├── goal_2.mp4
└── no_goal/
    ├── no_goal_1.mp4
    ├── no_goal_2.mp4

2. Covert clips into frames
3. Extract features from frames
4. Pass features into training model.


## Main runs the project

## Editor
Provides the tools to splice full matches into clips, split audio.

## Extract
Pulls each frame from the clip provided and sticks them into a dir.

## Features
Extracts features from each frame provided to be passed into model.

## Detection
CNN_LSTM model.

## Predict
Decides if there is a goal or no goal (goal=1, noGoal=0)

## Train
Used on test data to train model before passing in real datasets.




#### High level flow
from editor import create_clips, detect_audio_spikes
from extract import extract_frames
from features import extract_frame_features
from predict import predict_goal

# Step 1: Detect audio spikes and create subclips
spike_times = detect_audio_spikes("match.mp4")
create_clips("match.mp4", "clips/", spike_times)

# Step 2: Extract frames from each clip
for clip in os.listdir("clips/"):
    extract_frames(f"clips/{clip}", f"frames/{clip}")

# Step 3: Extract features from frames
features = extract_frame_features("frames/")

# Step 4: Predict if a goal occurred
prediction = predict_goal(features)
print(f"Prediction: {prediction}")
