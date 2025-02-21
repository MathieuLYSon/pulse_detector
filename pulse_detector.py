import cv2  # OpenCV for video processing
import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for handling ECG data
import mediapipe as mp  # MediaPipe for face detection
import torch  # PyTorch for deep learning
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt  # Signal processing utilities
import matplotlib.pyplot as plt  # Visualization library
import os  # For file handling

# Set directories for data
VIDEO_DIR = "Video"  # Folder containing video files
DATA_DIR = "data"  # Folder containing ECG text files

# Initialize MediaPipe face detection
mp_face_mesh = mp.solutions.face_mesh  # Load the FaceMesh model from MediaPipe

def extract_roi(frame):
    """Extract the forehead region as the Region of Interest (ROI) from a video frame."""
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB for processing
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape  # Get frame dimensions
                forehead_x = int(face_landmarks.landmark[10].x * w)  # Forehead x position
                forehead_y = int(face_landmarks.landmark[10].y * h)  # Forehead y position
                roi = frame[forehead_y-10:forehead_y+10, forehead_x-10:forehead_x+10, 1]  # Extract green channel
                return np.mean(roi)  # Return average intensity in the ROI
    return None  # Return None if no face is detected


def process_video(video_path):
    """Extract ROI from each frame of a given video and return a processed time series."""
    cap = cv2.VideoCapture(video_path)  # Open the video file
    video_frames = []  # List to store extracted ROI values
    
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break  # Stop if no frame is read
        roi_value = extract_roi(frame)  # Extract ROI from the frame
        if roi_value is not None:
            video_frames.append(roi_value)  # Store extracted value
    
    cap.release()  # Release the video file
    return np.array(video_frames)  # Return extracted ROI values


def bandpass_filter(signal, lowcut=0.5, highcut=4.0, fs=30, order=3):
    """Apply a bandpass filter to remove noise from the extracted signal."""
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist  # Normalize low cutoff frequency
    high = highcut / nyquist  # Normalize high cutoff frequency
    b, a = butter(order, [low, high], btype='band')  # Design filter
    return filtfilt(b, a, signal)  # Apply filter


def load_ecg(file_path):
    """Load ECG data from a text file and align it with the extracted PPG signal."""
    data = pd.read_csv(file_path, delimiter=" ", skiprows=4, header=None).values.flatten()  # Read ECG data
    return data[:2100]  # Ensure correct length (21 sec * 100 Hz = 2100 samples)


# Define Dataset Class
class VideoECGDataset(Dataset):
    """Custom dataset class for handling video and ECG signal data."""
    def __init__(self, video_frames, ecg_signals):
        self.video_frames = torch.tensor(video_frames, dtype=torch.float32).unsqueeze(1)
        self.ecg_signals = torch.tensor(ecg_signals, dtype=torch.float32)
    
    def __len__(self):
        return len(self.video_frames)
    
    def __getitem__(self, idx):
        return self.video_frames[idx], self.ecg_signals[idx]

# Define Model
class PulseEstimationModel(nn.Module):
    """CNN-LSTM model for pulse estimation."""
    def __init__(self):
        super(PulseEstimationModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.lstm = nn.LSTM(input_size=16, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Define Training Function
def train_model(dataset, epochs=10, lr=0.001):
    model = PulseEstimationModel()
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for frames, labels in train_loader:
            frames = frames.unsqueeze(1)
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


def process_all_data():
    """Process all video and ECG files, match them by filename, and analyze pulse signals."""
    for filename in os.listdir(VIDEO_DIR):
        if filename.endswith(".mp4"):  # Process only MP4 files
            video_path = os.path.join(VIDEO_DIR, filename)
            ecg_path = os.path.join(DATA_DIR, filename.replace(".mp4", ".txt"))  # Match corresponding ECG file
            
            if os.path.exists(ecg_path):
                print(f"Processing {filename}...")
                video_signal = process_video(video_path)  # Extract PPG-like signal from video
                filtered_signal = bandpass_filter(video_signal)  # Apply bandpass filter
                ecg_signal = load_ecg(ecg_path)  # Load corresponding ECG data
                dataset = VideoECGDataset(filtered_signal, ecg_signal)
                train_model(dataset)  # Train model
                print(f"Finished processing {filename}")
            else:
                print(f"ECG file not found for {filename}")

# Run the processing pipeline
if __name__ == "__main__":
    process_all_data()  # Start processing
