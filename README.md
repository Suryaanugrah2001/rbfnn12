# rbfnnforforecast

import pandas as pd

# Impor data dari file Excel
from google.colab import drive
drive.mount("/content/gdrive")
from google.colab import files
data = pd.read_excel('nama_file_excel.xlsx')
data = pd.DataFrame(data)
data

#spitdata
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Inisialisasi model RBFNN
rbfnn = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons,), activation='logistic', solver='lbfgs')

# Latih model
rbfnn.fit(X_train, y_train)

# Prediksi
predictions = rbfnn.predict(X_test)

# Hitung MAPE
mape = mean_absolute_percentage_error(y_test, predictions)
print(f'MAPE: {mape}')

import matplotlib.pyplot as plt

# Visualisasi hasil
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title('Actual vs. Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.show()



