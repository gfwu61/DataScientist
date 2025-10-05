# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 17:28:03 2025

@author: WUG2SI
"""

from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
import os


seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
# für deterministische GPU (langsam!)
os.environ["TF_DETERMINISTIC_OPS"] = "1"


# Die Bilder herunterladen
data = fetch_lfw_people(data_home=None,
                        funneled=True,
                        resize=1,  # Im Falle von Arbeitsspeicherproblemen: 0.5
                        min_faces_per_person=50,
                        color=True,
                        slice_=(slice(50, 200, None), slice(50, 200, None)),
                        # slice_=(slice(0, 250, None), slice(0, 250, None)),

                        download_if_missing=True,
                        return_X_y=False)

# In x, y aufteilen
x = data.images
y = data.target
target_names = data.target_names
print("y: type:", type(y))
print("x.shape:, dtype: ", x.shape, x.dtype)
print("x[0,0,0,0:3]:", x[0,0,0,0:3])
# Bilder normalisieren
# Je nach scikit-learn-Version ist x.max() entweder 1 oder 255
# Wir geben uns erst mal die Max- und Min-Werte der Pixelintensitäten aus
print('Max and Min pixel values:', x.max(), x.min())


label_counts = pd.Series(y).value_counts()
print("\nLabel-Anzahl :")
print(label_counts.head(20))
print("target_names:", target_names)

# Die Pixel sollten am besten alle mit dem gleichen Faktor skaliert werden
# Mit dem MinMaxscaler würden wir pro Pixel einen unterschielichen Skalierungsfaktor
# bekommen und dadurch werden die Bilder sehr scheckig
# manche Ecken oder Kanten würden in andere Bilder einfließen
x = x/x.max()

# Einige Bilder darstellen
for i in range(12):
    plt.subplot(3, 4, i+1)  # Subplot mit 3 Zeilen und 4 Spalten
    plt.imshow((x[y == i][0]))
    plt.title(data.target_names[i], fontsize=8)
    plt.axis('off')  # Keine Achsenbeschriftung
plt.show()


# Labels encoden (LabelEncoder + One-Hot)
# ['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush''Gerhard Schroeder' 'Hugo Chavez' 'Jacques Chirac' 'Jean Chretien'
# 'John Ashcroft' 'Junichiro Koizumi' 'Serena Williams' 'Tony Blair']
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"num_classes: {num_classes}")
y_ohe = keras.utils.to_categorical(y_enc, num_classes=num_classes)
print(f"\nAnzahl Klassen: {num_classes}")

# Train/Val Split (stratifiziert)
X_train, X_val, y_train, y_val, y_train_idx, y_val_idx = train_test_split(
    x, y_ohe, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

print("X_train[0,0,0,0:3]:", X_train[0,0:150,0:150,0])
#print(f"x.shape:{x.shape}, x.type: { x.dtype}, \n X_train:{X_train} ")
print(f"x.shape:{x.shape}, x.type: { x.dtype} ")

print("y_val:", y_val)
# Ein Convolution Neuronales Netz designen
# Convolutional Neural Net design (Sequential-Style via Functional API)
input_shape = X_train.shape[1:]
inputs = keras.Input(shape=input_shape, name="input_layer")


x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

# Flatten oder GlobalAveragePooling
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x) #0.5
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)  #0.4

# num_classes=12
outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="lfw_cnn_small")

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


# Model summary anzeigen
model.summary()

epochs = 25 #25
batch_size = 64

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_val, y_val))



# Evaluation
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

# Plot Training history
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss Verlauf')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy Verlauf')
plt.xlabel('Epoche')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# y_val: OneHot encoded
y_val_true_classes = np.argmax(y_val, axis=1)
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_val_true_classes, y_val_pred_classes)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_val_true_classes, y_val_pred_classes)
print("Classification Report:")
print(report)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

