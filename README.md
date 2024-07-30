# Gait Analysis to Detect Emotion

This project involves using gait analysis to detect human emotions. It leverages machine learning techniques to analyze human gait data and determine emotional states.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The goal of this project is to analyze human gait data and detect emotions based on the analysis. This can have applications in security, healthcare, and entertainment.

## Features
- Data extraction using OpenPose
- Data preprocessing and conversion
- Machine learning model training and evaluation
- Visualization of results

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sarojshakya01/gait-emotion.git
   cd gait-emotion
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Extract data using OpenPose:
   ```bash
   python extract_openpose.py
   ```

2. Convert data formats:
   ```bash
   python h5_to_csv.py
   python h5_to_npy.py
   ```

3. Train the model:
   ```bash
   python main.py
   ```

4. Visualize the results:
   ```bash
   python animate_data.py
   ```

## Directory Structure
- `net/`: Contains the neural network models.
- `test/`: Contains test scripts and datasets.
- `utils/`: Utility scripts for data processing and visualization.
- `animate_data.py`: Script for visualizing data.
- `extract_openpose.py`: Script for extracting data using OpenPose.
- `h5_to_csv.py`: Script for converting H5 data to CSV format.
- `h5_to_npy.py`: Script for converting H5 data to NPY format.
- `main.py`: Main script for training the model.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

This README provides an overview and guidance for using the repository effectively.
