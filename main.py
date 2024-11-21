import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import kagglehub
import os

# Download latest version of the dataset
path = kagglehub.dataset_download("xainano/handwrittenmathsymbols")
print("Path to dataset files:", path)

path_plus = os.path.join(path, "extracted_images", "+")
path_minus = os.path.join(path, "extracted_images", "-")

images_plus = []
images_minus = []

for file in os.listdir(path_plus):
    img_path = os.path.join(path_plus, file)
    if not os.path.isfile(img_path):
        continue
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    images_plus.append(img_array)
    if len(images_plus) >= 30:
        break

print("Loaded all + images")

for file in os.listdir(path_minus):
    img_path = os.path.join(path_minus, file)
    if not os.path.isfile(img_path):
        continue
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    images_minus.append(img_array)
    if len(images_plus) >= 30:
        break

print("Loaded all - images")



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

# Arrays für Symbole laden
images_plus = np.vstack(images_plus)  # Stapeln der geladenen + Symbole
images_minus = np.vstack(images_minus)  # Stapeln der geladenen - Symbole

# Labels erstellen
labels_plus = np.ones(len(images_plus))  # Labels für + = 1
labels_minus = np.zeros(len(images_minus))  # Labels für - = 0

# Kombiniere die Daten
X = np.vstack((images_plus, images_minus))
y = np.concatenate((labels_plus, labels_minus))

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Daten normalisieren
X_train = X_train / 255.0
X_test = X_test / 255.0



def train_sequentially(X_train, y_train, epochs=1000):
    if os.path.isfile('models/model_sequentially.h5'):
        model = tf.keras.models.load_model('models/model_sequentially.h5')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    model = Sequential([
        Flatten(input_shape=(150, 150, 3)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    for i in range(len(X_train)):
        print(f"{i}/{len(X_train)}")
        model.fit(X_train[i:i+1], y_train[i:i+1], epochs=epochs)

    model.save('models/model_sequentially.h5')
    return model



def train_randomly(X_train, y_train, epochs=1000):
    if os.path.isfile('models/model_randomly.h5'):
        model = tf.keras.models.load_model('models/model_randomly.h5')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    model = Sequential([
        Flatten(input_shape=(150, 150, 3)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    indices = np.arange(len(X_train))
    for epoch in range(epochs):
        np.random.shuffle(indices)
        for i in indices:
            print(f"{epoch}/{epochs}")
            model.fit(X_train[i:i+1], y_train[i:i+1], epochs=1)

    model.save('models/model_randomly.h5')
    return model


def build_model_with_hidden_layers(hidden_layers):
    if os.path.isfile(f'models/model_layers{hidden_layers}.h5'):
        model = tf.keras.models.load_model(f'models/model_layers{hidden_layers}.h5')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    model = Sequential([Flatten(input_shape=(150, 150, 3))])
    for _ in range(hidden_layers):
        model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=1000, verbose=0)
    model.save(f'models/model_layers{hidden_layers}.h5')
    return model


def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    return accuracy


train_sequentially(X_train, y_train)
train_randomly(X_train, y_train)
