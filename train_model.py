import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Set the paths for training and testing datasets
train_spiral_dir = r'D:\mini\Parkinson’s Disease Detection Using Spiral Images (Hand Drawings)\Parkinson Dataset\Parkinson Dataset\dataset\spiral\training'
test_spiral_dir = r'D:\mini\Parkinson’s Disease Detection Using Spiral Images (Hand Drawings)\Parkinson Dataset\Parkinson Dataset\dataset\spiral\testing'
train_wave_dir = r'D:\mini\Parkinson’s Disease Detection Using Spiral Images (Hand Drawings)\Parkinson Dataset\Parkinson Dataset\dataset\wave\training'
test_wave_dir = r'D:\mini\Parkinson’s Disease Detection Using Spiral Images (Hand Drawings)\Parkinson Dataset\Parkinson Dataset\dataset\wave\testing'

# Data Generators for Spiral and Wave Images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create training and validation generators for spiral and wave images
train_spiral_generator = train_datagen.flow_from_directory(
    train_spiral_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_spiral_generator = test_datagen.flow_from_directory(
    test_spiral_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

train_wave_generator = train_datagen.flow_from_directory(
    train_wave_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_wave_generator = test_datagen.flow_from_directory(
    test_wave_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# DenseNet Model Definition
def create_densenet_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create models for spiral and wave images
spiral_model = create_densenet_model()
wave_model = create_densenet_model()

# Train the models separately
spiral_model.fit(
    train_spiral_generator,
    epochs=50,
    validation_data=test_spiral_generator
)
wave_model.fit(
    train_wave_generator,
    epochs=50,
    validation_data=test_wave_generator
)

# Save the trained models
spiral_model.save('spiral_detection_model_densenet.h5')
wave_model.save('wave_detection_model_densenet.h5')
