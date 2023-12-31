# Plant Disease Classification
<img width="697" alt="image" src="https://github.com/Yashsharma009/Plant_disease/assets/116294789/64a51279-984f-4ffa-bc16-d36900335cf5">

This project uses deep learning with TensorFlow to classify plant diseases. The dataset used is the PlantVillage dataset, which contains images of various plant diseases affecting tomatoes.

## Getting Started

### Prerequisites

- [Google Colab](https://colab.research.google.com/) (for running the notebook)
- [TensorFlow](https://www.tensorflow.org/install)
- [Matplotlib](https://matplotlib.org/stable/users/installing.html)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/plant-disease-classification.git
cd plant-disease-classification
```

2. Open the Jupyter notebook in Google Colab:

- Upload the notebook `plant_disease_classification.ipynb` to your Google Drive.
- Open it with Google Colab.

3. Run the notebook:

- Follow the instructions in the notebook to mount Google Drive, install dependencies, and train the model.

## Training the Model

The notebook contains code for training a convolutional neural network (CNN) on the PlantVillage dataset. It includes data preprocessing, model architecture, training, and evaluation.

```python
# Example training command
history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
)
```

## Model Evaluation

The trained model achieves an accuracy of approximately 80.4% on the test set.

```python
# Example model evaluation
scores = model.evaluate(test_ds)
print(f"Test Accuracy: {round(scores[1], 4) * 100}%")
```

## Results and Visualization

The notebook includes visualizations of training and validation accuracy and loss over epochs.

<img width="312" alt="image" src="https://github.com/Yashsharma009/Plant_disease_classification/assets/116294789/4ec69f34-63a1-48ff-99d6-d6286987c229">


## Predictions and Cure Information

The model can make predictions on test set images, providing predicted classes, confidence scores, and suggested cures.

<img width="592" alt="image" src="https://github.com/Yashsharma009/Plant_disease_classification/assets/116294789/73ea0764-9926-45df-921d-1b9a3ebaee3c">


