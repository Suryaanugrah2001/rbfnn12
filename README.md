Radial Basis Function NN, Time Series Data
# import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyrbf import RBF
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Impor data dari file Excel
from google.colab import drive
drive.mount("/content/gdrive")
from google.colab import files
data = pd.read_excel('nama_file_excel.xlsx')
data = pd.DataFrame(data)
data
# Pra-pemrosesan Data: Pisahkan variabel independen (fitur) dan variabel dependen (target), lalu bagi data menjadi set pelatihan dan pengujian
X = data[['fitur1', 'fitur2', 'fitur3']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pemrosesan Skala: Skalakan fitur-fitur Anda menggunakan StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membangun Model
rbf = RBF(hidden_shape=10)  # Jumlah neuron tersembunyi, sesuaikan dengan kebutuhan Anda
rbf.fit(X_train, y_train)
y_pred = rbf.predict(X_test)

# Evaluasi Model
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = calculate_mape(y_test, y_pred)
print(f'MAPE: {mape:.2f}%')

# Visualisasi Grafik
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()

