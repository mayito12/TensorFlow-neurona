import math
import numpy as np
from ventana_ui import *
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    w = []
    error = 0
    X = []
    Y = []

    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.cant_gen
        self.ejec_tensor.clicked.connect(self.tensor)
        self.abalone_features = []
        self.abalone_labels = []
        self.aprendizaje = 0.002
        self.error_permisible = 0.01
        self.pesos

    def leer_datos(self):
        df = pd.read_csv(f'data.csv',
                         header=None,
                         names=['x1', 'x2', 'x3', 'x4', 'y'])
        x = df[['x1', 'x2', 'x3', 'x4']].values
        y = df['y'].values
        print("X: ",x)
        print("Y : ",y)
        sesgo = []
        for i in range(len(x)):
            sesgo.append([1, x[i][0], x[i][1], x[i][2], x[i][3]])
        
        self.X = sesgo
        self.Y = y
        self.w = np.random.rand(len(self.X[0]))

    def calcular_error(self):
        wt = np.transpose(self.w)
        u = (np.dot(self.X, wt))
        e = []
        for i in range(len(u)):
            e.append(self.Y[i] - u[i])
        et = np.transpose(e)
        delta_w = []
        for i in range(len(self.w)):
            delta_w = (np.dot(et, self.X) * self.aprendizaje)
        w_nuevo = self.w + delta_w
        self.w = w_nuevo
        return w_nuevo
        
    def calcular_e(self, error):
        e = 0
        for i in range(len(error)):
            e = e + error[i]**2
        return math.sqrt(e)

    def perceptron(self):
        self.pesos.setText("")
        his_error = []
        self.leer_datos()
        for i in range(int(self.cant_gen.text())):
            w_nuevo = self.calcular_error()
            his_error.append(self.calcular_e(w_nuevo))
            print(self.calcular_e(w_nuevo))
            if self.calcular_e(w_nuevo) < self.error_permisible:
                break
        texto = "Pesos Finales : "+str(self.w) 
        self.pesos.setText(texto)
        print(self.w)
        plt.plot(his_error)
        plt.legend()
        plt.xlabel("Generaciones")
        plt.ylabel("Rango error")
        plt.show()

    def tensor(self):
        self.pesos.setText("")
        self.leer_Archivo()
        n = 0.1
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.Adam(n))
        historial = model.fit(self.abalone_features,
                              self.abalone_labels, epochs=int(self.cant_gen.text()))
        result = model.predict(self.abalone_features)
        print(f'PESOS: {model.get_weights()}')
        texto = "Pesos Finales : "+str(model.get_weights()[0]) 
        self.pesos.setText(texto)
        plt.plot(historial.history['loss'], label=f'n={n}')
        print(f'RESULT: {result}')
        plt.legend()
        plt.xlabel("Generaciones")
        plt.ylabel("Rango error")
        plt.show()

    def leer_Archivo(self):
        abalone_train = pd.read_csv(
            "data.csv", names=["x1", "x2", "x3", "x4", "y"])
        self.abalone_features = abalone_train.copy()
        self.abalone_labels = self.abalone_features.pop("y")
        self.abalone_features = np.array(self.abalone_features)
        print("lista de x: ", self.abalone_features)
        print("lista de y: ", self.abalone_labels)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()