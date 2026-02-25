Cancer Image Classifier with PyTorch

This project implements a deep learning‑based image classification system for distinguishing between benign and malignant cancer images using PyTorch. It leverages transfer learning with a ResNet18 (or optionally MobileNetV2) backbone, and provides both training and batch prediction pipelines.
Features

    Train a custom classifier on your own dataset (requires folder‑structured data: benign/ and malignant/)

    Data augmentation for better generalization

    Automatic CPU/GPU support

    Model saving/loading (cancer_classifier.pth)

    Batch prediction on new images with CSV output

    Classification report and confusion matrix after training

Requirements

    Python 3.7 or higher

    PyTorch and torchvision

    Additional libraries: numpy, matplotlib, Pillow, scikit‑learn

Install the dependencies:
bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu   # for CPU only
# For CUDA support, follow the instructions at https://pytorch.org
pip install numpy matplotlib pillow scikit-learn

Project Structure

After cloning or creating the files, your project should look like this:
text

your_project/
│
├── images/                # Training data folder
│   ├── benign/            # Benign images
│   └── malignant/         # Malignant images
├── new_images/            # (Optional) Folder for images to predict
├── train.py               # Training script
├── predict.py             # Prediction script
├── cancer_classifier.pth  # Saved model (after training)
├── predictions.csv        # Prediction results (after running predict.py)
└── README.md              # This file

Preparing the Data for Training

Place your training images inside the images folder, with separate subfolders for each class:
text

images/
    benign/
        img1.jpg
        img2.jpg
        ...
    malignant/
        img1.jpg
        ...

The class names can be anything, but they will be mapped to labels in alphabetical order (typically benign → 0, malignant → 1).
Training the Model

Run the training script:
bash

python train.py

During training, you will see the loss and accuracy for each epoch. After completion:

    The best model is saved as cancer_classifier.pth

    Training curves (loss/accuracy) are displayed

    A classification report and confusion matrix for the validation set are printed

Note: If your images are unlabeled and you only want predictions, you must first train a model on a suitable labeled dataset.
Batch Prediction on New Images

To classify many images (e.g., those inside the new_images folder), use the predict.py script. Before running:

    Ensure the trained model file cancer_classifier.pth exists.

    Set the correct input folder path (INPUT_FOLDER) in predict.py.

    Verify that the class names list (CLASS_NAMES) matches your training order (default ['benign', 'malignant']).

Then execute:
bash

python predict.py

For each image, the predicted class and confidence are printed and saved to predictions.csv with the following format:
Filename	Predicted Class	Confidence
image1.jpg	benign	92.45%
image2.png	malignant	78.12%
...	...	...
Improving Performance

    More data: Aim for at least 100–200 images per class.

    Data augmentation: Adjust the augmentation parameters in train.py to better match your data.

    Regularization: Dropout and weight decay are already included; you can tune them.

    Alternative architecture: If your dataset is very small, consider using a smaller model like MobileNetV2 (commented code in train.py).

Troubleshooting
SSL certificate error when downloading pretrained weights

If you encounter an SSL error during the first run:

    Upgrade certifi: pip install --upgrade certifi

    Or manually download the weights from this link and place them in C:\Users\[username]\.cache\torch\hub\checkpoints\ (Windows) or ~/.cache/torch/hub/checkpoints/ (Linux/Mac).

Out‑of‑memory (GPU)

Reduce the BATCH_SIZE in train.py (e.g., to 8 or 16).
Model loading errors

Make sure the model architecture defined in predict.py exactly matches the one used during training. If you modified the architecture, update the create_model() function accordingly.
Contributing

Suggestions and improvements are welcome! Feel free to open an issue or submit a pull request.
License

This project is released under the MIT License. It is intended for educational and research purposes only and should not be used for clinical diagnosis without proper validation.

Developer: Hamidreza Hosseini
Last updated: 2026-02-25
