import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import confusion_matrix

regions = {
    'HU': '10YHU-MAVIR----U',
    'IT': '10YIT-GRTN-----B',
    'PO': '10YPL-AREA-----S',
    'SP': '10YES-REE------0',
    'UK': '10Y1001A1001A92E',
    'DE': '10Y1001A1001A83F',
    'DK': '10Y1001A1001A65H',
    'SE': '10YSE-1--------K',
    'NE': '10YNL----------L',
}

def compute_max_region(row):
    max_region = max(regions, key=lambda region: row[f'green_energy_{region}'] - row[f'{region}_Load'])
    return max_region

def compute_diffs(row):
    max_region = max([row[f'green_energy_{region}'] - row[f'{region}_Load'] for region in regions])
    return max_region

df = pd.read_parquet('../data/preprocessed/training_dataset.parquet', engine = 'pyarrow')
targets = df.apply(compute_max_region, axis=1)#.shift(-1)
print(targets)
# df = df.drop('Date', axis=1)

# targets = df['target']
df = df.select_dtypes(include=[np.number])
X_train, X_test, y_train, y_test = train_test_split(df, targets, test_size=0.2, random_state=42)
print(X_train.head(5))

# X_train = X_train.to_numpy()
# y_train = y_train.to_numpy().T

print(y_test.tail())

y_train = y_train.astype('category').cat.codes
y_test = y_test.astype('category').cat.codes

X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int8)

X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.int8)

n_input = 12
train_generator = TimeseriesGenerator(X_train, y_train, length=n_input, batch_size=1)
test_generator = TimeseriesGenerator(X_test, y_test, length=n_input, batch_size=1)

dense_neurons = len(np.unique(y_train))

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_input, 26), return_sequences=True))
model.add(LSTM(30, activation='relu', return_sequences=True))
model.add(LSTM(20, activation='relu'))
model.add(Dense(dense_neurons, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=2).history
# model.fit(train_generator, epochs=2)

print('Model evaluation:')
res = model.evaluate(test_generator)
print(res)
print('-----------------')

trainPredict = model.predict(train_generator)
testPredict = model.predict(test_generator)

print('Train:', len(trainPredict), len(y_train))
print('Test:', len(testPredict), len(y_test))

exit()

predicted_classes = np.argmax(testPredict, axis=1)
mtx = confusion_matrix(y_test, predicted_classes)
print(mtx)

print(trainPredict)
print(testPredict)

exit()



# X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

X_train=X_train.to_numpy().reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.to_numpy().reshape(X_test.shape[0],X_test.shape[1],1)

print(X_train)
# Crear el modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))  # Capa de salida

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f'Loss en el conjunto de prueba: {loss}')

# Hacer predicciones
predictions = model.predict(X_test)













exit()

# Assuming df_training and df_test contain your data
# Extract features and target variable
# X_train = df_train.drop(columns=['target']).values
# y_train = df_train['target'].values
# X_test = df_test.drop(columns=['target']).values
# y_test = df_test['target'].values

# X_train = X_train.select_dtypes(include=['number'])  # Select numeric columns
# X_test = X_test.select_dtypes(include=['number'])  # Select numeric columns
# X_train = X_train.select_dtypes(include=['number'])
# X_test = X_test.drop(X_test.select_dtypes(exclude=['number']), axis=1)

# Encoding categorical variables if needed
# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train)
# y_test_encoded = label_encoder.transform(y_test)
#
# # Scaling numerical features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM input (assuming 3D input with samples, time steps, and features)
# You may need to adjust these based on your data
n_steps = 1  # Number of time steps (hours)
n_features = X_train.shape[1]  # Number of features

X_train_reshaped = X_train.reshape((X_train.shape[0], n_steps, n_features))
X_test_reshaped = X_test.reshape((X_test.shape[0], n_steps, n_features))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(units=1, activation='sigmoid'))  # Adjust units and activation for your classification task

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use appropriate loss and metrics

# Train the model
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.1)  # Adjust epochs and batch_size

# Evaluate the model
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print(f'Test Accuracy: {accuracy}')

# Make predictions
predictions = model.predict(X_test_reshaped)


# import pmdarima as pm
# from pmdarima.model_selection import train_test_split
# import numpy as np
# import matplotlib.pyplot as plt
#
# train, test = train_test_split(co2_data.co2.values, train_size=2200)
#
# model = pm.auto_arima(train, seasonal=True, m=52)
# preds = model.predict(test.shape[0])
#
# plt.plot(co2_data.co2.values[:2200], train)
# plt.plot(co2_data.co2.values[2200:], preds)
# plt.show()