Play Vision POC - Football Goal Detection Project

Welcome to the Play Vision project! This repository is an end-to-end solution for detecting football goals in match footage using Computer Vision (CV) and Deep Learning (DL) techniques. The project combines audio spike detection, frame extraction, feature extraction using ResNet, and goal prediction using a CNN + LSTM model.

📂 Project Structure

play-vision/

├── main.py         # Main entry point for running the entire project

├── detection.py    # Contains the CNN_LSTM model architecture

├── editor.py       # Handles video editing and audio spike detection

├── extract.py      # Extracts frames from video clips

├── features.py     # Extracts ResNet features from frames

├── predict.py      # Uses the trained model to predict goals in new clips

├── train.py        # Handles the training process for the CNN_LSTM model
