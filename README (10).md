# Environmental Sound Classification with a CNN (ESC-50)

A convolutional neural network that classifies 5-second environmental audio clips into **50 sound classes** (dog bark, rain, siren, keyboard typing, etc.) using log-mel spectrograms and PyTorch.

**Result: 70.00% test accuracy** (280 / 400 clips) on a held-out fold, against 2% chance level.

## Pipeline

```
audio clip (5 s) -> log-mel spectrogram (128 x 216) -> CNN -> 1 of 50 classes
```

## Dataset

[ESC-50](https://github.com/karolpiczak/ESC-50): 2,000 labelled clips, 50 classes, 40 clips per class, split into 5 official folds. Using the official folds keeps clips from the same original recording out of both train and test.

| Split | Folds | Clips |
|-------|-------|-------|
| Train | 1-3 | 1,200 |
| Validation | 4 | 400 |
| Test | 5 | 400 |

## Preprocessing

- Resample to 22,050 Hz; pad or cut each clip to exactly 5 seconds
- Mel spectrogram: `n_fft=1024`, `hop_length=512`, 128 mel bins, converted to decibels, giving a 128 x 216 array per clip
- Normalise with the training set's mean (-32.78) and std (21.51) only

![Example spectrograms](example_spectrograms.png)

## Data augmentation (training only)

Random circular time shift, frequency mask, time mask (SpecAugment-style) and light Gaussian noise. Validation and test data are never augmented.

## Model

Four blocks of *(Conv3x3 -> BatchNorm -> ReLU) x 2 -> MaxPool*, with channels 32 -> 64 -> 128 -> 256. Then global average pooling, Dropout (0.4) and a linear layer with 50 outputs.

**1,186,450 trainable parameters.**

## Training setup

| Setting | Value |
|---------|-------|
| Loss | Cross-entropy |
| Optimiser | Adam, lr = 0.001 |
| Schedule | Cosine annealing over 40 epochs |
| Batch size | 32 |
| Epochs | 40 (about 3.2 min on a Colab T4 GPU) |
| Model selection | Best validation accuracy (epoch 33) |
| Seed | 42 |

## Results

| Metric | Value |
|--------|-------|
| Best validation accuracy | 74.25% (epoch 33) |
| **Test accuracy** | **70.00%** (280 / 400) |
| Chance level | 2% |
| Human accuracy on ESC-50 | about 81% (reported by the dataset authors) |

The test fold was used once, only for the final evaluation.

![Training curves](training_curves.png)

Train accuracy reached about 83% and validation accuracy about 74%, so there is some overfitting, but validation loss kept falling to the end, and augmentation plus dropout kept the gap moderate.

### Per-class performance

![Per-class accuracy](per_class_accuracy.png)

- **Best classes (100%):** hand saw, car horn, clock alarm, glass breaking, toilet flush. These have distinctive, consistent acoustic signatures.
- **Worst classes:** clapping (0%), airplane (0%), water drops, pig and helicopter (12.5% each).

### Most common confusions (true -> predicted)

| True | Predicted | Count |
|------|-----------|-------|
| fireworks | footsteps | 6 |
| helicopter | engine | 4 |
| cat | rooster | 4 |
| airplane | thunderstorm | 4 |
| mouse click | crackling fire | 3 |
| clapping | sea waves | 3 |

Most errors involve sounds with similar spectral texture (helicopter and engine, airplane and thunderstorm) or short impulsive sounds (fireworks, clapping, mouse click) that look alike once converted to spectrograms.

![Confusion matrix](confusion_matrix.png)

## How to run

1. Open `ESC50_Sound_Classification.ipynb` in [Google Colab](https://colab.research.google.com).
2. Runtime -> Change runtime type -> **T4 GPU**.
3. Run all cells in order. The notebook downloads ESC-50, extracts features, trains, evaluates, and saves the figures and `best_model.pt`.

Dependencies: `torch`, `librosa`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`.

## Repository contents

| File | Description |
|------|-------------|
| `ESC50_Sound_Classification.ipynb` | Full notebook (data, model, training, evaluation) |
| `best_model.pt` | Trained weights (best validation epoch) |
| `example_spectrograms.png` | Sample log-mel spectrograms |
| `training_curves.png` | Loss and accuracy per epoch |
| `confusion_matrix.png` | Normalised 50x50 test confusion matrix |
| `per_class_accuracy.png` | Sorted per-class test accuracy |

## Limitations and future work

- Only 1,200 training clips; results come from a single run and one train/val/test split, so expect a few points of variation. A 5-fold cross-validation would give a more reliable estimate.
- Pretrained audio models (e.g. PANNs, AST) or ImageNet-pretrained CNNs fine-tuned on spectrograms would likely score much higher.
- Further augmentation (mixup, pitch shift, time stretch) could help the weak impulsive-sound classes.

## Acknowledgements

Dataset: K. J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia 2015.
