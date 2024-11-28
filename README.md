# Image Captioning with Hybrid CNN-RNN Model

This repository contains an image captioning system that utilizes a hybrid model combining a Convolutional Neural Network (CNN) for feature extraction and a Recurrent Neural Network (RNN), specifically an LSTM, for generating captions. The model is trained using the **Flickr8k dataset**, which consists of 8,000 images, each paired with multiple captions.

## Overview

The system works by first extracting features from images using a pre-trained VGG16 model, which serves as the CNN. These features are then passed to an LSTM-based RNN that generates captions word by word, based on the visual context provided by the CNN. The model uses tokenized captions for training and generates captions in natural language for unseen images.

### Key Features:
- **VGG16 CNN for feature extraction:** Pre-trained on ImageNet for robust feature representation.
- **LSTM RNN for caption generation:** Captions are generated sequentially, word by word.
- **Tokenization of captions:** Text processing with tokenization and padding for sequence alignment.
- **Flickr8k Dataset:** A dataset with 8,000 images, each having five captions.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Keras
- Matplotlib
- scikit-learn
- Pillow

## Dataset

The model uses the **Flickr8k dataset** which can be downloaded from [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k). After downloading the dataset, ensure that the images are stored in the `flickr8k/images/` folder and the captions in the `flickr8k/captions.txt` file.

## File Structure

```bash
image_captioning.ipynb       # Main Jupyter notebook for training and inference
flickr8k/
    images/                  # Folder containing the images
    captions.txt             # File containing captions for the images
```

## Usage

1. **Preprocess the data**: 
   - Load the images and captions, and preprocess them for training.

2. **Model Training**:
   - The hybrid CNN-RNN model is built using Keras and trained on the processed data.

3. **Generate Captions**:
   - Once trained, the model can be used to generate captions for new images by providing the image path.

### Example:

```python
test_image_id = 'dog'  # Example image ID
test_image_feature = preprocess_images('path_to_test_image')[test_image_id]
caption = generate_caption(model, tokenizer, test_image_feature, max_length)
print("Generated Caption:", caption)
```

## Model Architecture

The model consists of the following components:

- **CNN (VGG16)**: Used to extract features from images.
- **RNN (LSTM)**: Used to generate captions from the extracted features.
- **Dense Layer**: To predict the next word in the caption.
- **Embedding Layer**: Converts word indices into dense vector representations.

## Training the Model

Training is performed using the following parameters:

- **Epochs**: 20
- **Batch Size**: 64

```python
history = model.fit(
    [X1_train, X2_train], y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([X1_test, X2_test], y_test),
    verbose=1
)

---
