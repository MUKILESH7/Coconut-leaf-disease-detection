# Coconut Leaf Disease Detection

![Project Banner](path/to/banner-image)

## Overview
This project focuses on detecting diseases in coconut leaves using deep learning. Leveraging the **Inception ResNetV2** architecture, the model can identify and classify different types of coconut leaf diseases from images. Early and accurate detection of these diseases helps mitigate their impact and supports sustainable coconut farming.

## Table of Contents
- Overview
- Features
- Dataset
- Methodology
- Installation
- Usage]
- Results
- Future Work
- Contributing
- License

## Features
- Automatic classification of coconut leaf diseases.
- High accuracy using the Inception ResNetV2 model.
- User-friendly interface for uploading and testing leaf images.
- Scalable solution for agriculture monitoring systems.

## Dataset
The dataset comprises:
- Images of healthy and diseased coconut leaves.
- Annotations for supervised learning.

You can use publicly available datasets or create your own dataset by collecting coconut leaf images in various conditions.

## Methodology
1. **Data Preprocessing**:
   - Resizing images to the input size required by Inception ResNetV2.
   - Augmenting images to improve model generalization.
   - Splitting data into training, validation, and test sets.

2. **Model Architecture**:
   - The deep learning model is based on **Inception ResNetV2**, a hybrid architecture combining the strengths of Inception and ResNet networks.

3. **Training**:
   - Fine-tuning the pre-trained Inception ResNetV2 model on the coconut leaf dataset.
   - Using loss functions such as categorical cross-entropy.
   - Optimizer: Adam with learning rate scheduling.

4. **Evaluation**:
   - Accuracy, precision, recall, and F1-score metrics.
   - Confusion matrix analysis for multi-class classification.

## Installation
### Prerequisites
- Python 3.8+
- TensorFlow 2.0+
- Other dependencies listed in `requirements.txt`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/MUKILESH7/coconut-leaf-disease-detection.git
   ```

2. Navigate to the project directory:
   ```bash
   cd coconut-leaf-disease-detection
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download or prepare the dataset and place it in the `data/` folder.

5. Train the model or use the pre-trained weights available in the `weights/` folder.

## Usage
### Training
To train the model on your dataset:
```bash
python train.py --dataset_path data/ --epochs 50 --batch_size 32
```

### Testing
To test the model on a single image:
```bash
python test.py --image_path path/to/image.jpg
```

### Running the Web Interface
A Flask-based web application is included:
```bash
python app.py
```
Access the interface at `http://127.0.0.1:5000`.

## Results
- Achieved **X% accuracy** on the test set.
- Confusion matrix and detailed performance metrics are available in the `results/` folder.
- Sample predictions:

| Image  | Predicted Label | Confidence |
|--------|-----------------|------------|
| 1| Grey leaf Spot    | 99%        |
| 2| Stem Bleeding     | 98%        |
|3|Leaf Blight         | 97%        |

## Future Work
- Incorporate additional leaf diseases.
- Deploy the model on mobile and edge devices for real-time monitoring.
- Integrate drone-based imaging for large-scale analysis.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Acknowledgments
- Inspired by state-of-the-art research in plant disease detection.
- Special thanks to the open-source community for providing pre-trained models and datasets.

---
Feel free to reach out for questions, suggestions, or collaborations!

