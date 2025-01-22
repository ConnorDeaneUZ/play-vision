# Play Vision - Football Goal Detection Project

Welcome to the **Play Vision** project. This repository provides an innovative, end-to-end solution for detecting football goals in match footage using advanced Computer Vision (CV) and Deep Learning (DL) techniques. The project is designed to streamline the process of identifying goals in large volumes of video data by combining cutting-edge AI methods with efficient video processing.

---

## 🌟 Project Highlights

### Key Features:
1. **Audio Spike Detection**:
   - Analyses audio streams to identify moments of high crowd activity or commentary intensity, narrowing down key moments in the footage.
   
2. **Frame Extraction**:
   - Extracts relevant video frames from clips, focusing on moments of interest identified during audio analysis.

3. **Feature Extraction with ResNet**:
   - Leverages a pretrained ResNet model to extract spatial features from video frames, reducing the complexity of training from scratch.

4. **Goal Prediction with CNN + LSTM**:
   - Uses a hybrid deep learning model:
     - **CNN** for analyzing spatial data (video frames).
     - **LSTM** for capturing temporal sequences (patterns across multiple frames).

5. **End-to-End Automation**:
   - Seamlessly integrates the entire pipeline from raw video input to final goal predictions, making it scalable for large datasets.

---

## 📊 Goals of the Project:
- **Automation**: Replace manual video review with a fully automated pipeline.
- **Scalability**: Process thousands of hours of match footage efficiently.
- **Accuracy**: Achieve high precision in goal detection using robust feature extraction and sequence modeling techniques.
- **Real-World Applicability**: Adaptable for broadcasters, analysts, or sports organisations to streamline game analysis.

---

## 🚀 Workflow Overview:
1. **Input**:
   - Match footage in standard video formats.
   
2. **Audio Analysis**:
   - Detect key audio spikes (e.g., cheers, commentary) to identify potentially significant moments.
   
3. **Frame Processing**:
   - Extract frames around identified moments for further analysis.
   
4. **Feature Extraction**:
   - Use ResNet to identify visual features of the extracted frames.
   
5. **Goal Prediction**:
   - Process the extracted features through a CNN + LSTM model to predict whether a goal occurred.

6. **Output**:
   - Generate a report with timestamps or video segments for detected goals.

---

## 🛠️ Technologies Used:
- **Python**: For implementing the pipeline and models.
- **Deep Learning Frameworks**: PyTorch for building and training the CNN + LSTM model.
- **Computer Vision**: OpenCV for video and frame processing.
- **Pretrained Models**: ResNet for feature extraction to save on computational cost and improve accuracy.
- **Audio Processing**: MoviePy and Numpy for detecting audio spikes and analysing sound intensity.

---
