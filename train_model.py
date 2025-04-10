import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

gestures = os.listdir("gestures")
gestures = [g.replace(".csv", "") for g in gestures]

X, y = [], []
sequence_length = 4

for idx, gesture in enumerate(gestures):
    file_path = f'gestures/{gesture}.csv'
    df = pd.read_csv(file_path)

    if len(df) < sequence_length:
        print(f"[WARNING] Skipping '{gesture}' — only {len(df)} rows, needs at least {sequence_length}")
        continue

    for i in range(len(df) - sequence_length):
        window = df.iloc[i:i + sequence_length].values
        X.append(window)
        y.append(idx)

X = np.array(X)
y = to_categorical(np.array(y))

if len(X) == 0:
    print("[ERROR] No valid data found.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(gestures), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save('model.keras')
print("[INFO] Model saved as 'model.keras'")
