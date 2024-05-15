# Audio Classification using TensorFlow
 UrbanSound8K Audio Classifier: TensorFlow model

## Overview
This repository contains a TensorFlow model for classifying audio samples from the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset. The model achieves high accuracy through a streamlined workflow and comprehensive data analysis.

## Dataset
The [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset consists of 8,732 audio samples across 10 classes. Each sample is labeled with the corresponding sound class, making it suitable for supervised learning tasks.

## Features
- Utilizes [Mel-Frequency Cepstral Coefficients (MFCC)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum#:~:text=Mel%2Dfrequency%20cepstral%20coefficients%20(MFCCs,%2Da%2Dspectrum%22).) for feature extraction.
- TensorFlow model architecture optimized for audio classification tasks.
- Efficient training process with Adam optimizer and early stopping.


## Usage of [Librosa](https://librosa.org/doc/latest/index.html) for Audio Processing

This project heavily relies on the [Librosa](https://librosa.org/doc/latest/index.html) library for various audio processing tasks, including:

- Loading audio files in various formats.
- Extracting features using [Mel-Frequency Cepstral Coefficients (MFCC)](https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html#librosa.feature.mfcc) function.
- Visualizing audio waveforms and spectrograms.
- Performing advanced audio analysis by gauging the sampling rate of the loaded audio.
- Performing manipulations such as time-stretching, pitch-shifting, and noise injection using Librosa's built-in functions.




## Model Architecture

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Input

# Define the model
model = Sequential(
    layers=[

            Input(shape=X[0].shape),

            # Layer 1
            Dense(100),
            Activation('relu'),
            Dropout(0.5),
            
            # Layer 2
            Dense(200),
            Activation('relu'),
            Dropout(0.5),
            
            # Layer 3
            Dense(100),
            Activation('relu'),
            Dropout(0.5),
            
            # Output Layer
            Dense(num_labels),
            Activation('softmax')
        
        ]

)
model.build()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

```

## Model-Train Configuration

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime

EPOCHS = 1000
BACTH_SIZE = 32

checkPoint = ModelCheckpoint(filepath='checkpoints/model.h5', verbose=1, save_best_only=True)

start = datetime.now()

model.fit(X_train, y_train, batch_size=BACTH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), callbacks=[checkPoint])

duration = datetime.now() - start
print("Elapsed Time: ", duration)
```

## Results
- Test accuracy: close to 82% (Actual : 81.96%)
- Model checkpoint available for further experimentation.


------

