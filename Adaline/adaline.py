import numpy as np
import csv
from random import uniform
import time

EPOCAS = 50


class Adaline:

    def __init__(self, n_entradas, bias=None, alfa=0.1):
        self.pesos_sinapticos = np.random.uniform(
            low=-0.05, high=0.05, size=(n_entradas,))
        self.n_entradas = n_entradas
        self.bias = bias if bias else uniform(0,0.1)
        self.alfa = alfa

    def calcular(self, estimulo):
        y = np.dot(self.pesos_sinapticos, estimulo)
        y += self.bias
        self.y = self.lineal(y)
        return self.y

    def entrenar_estocastico(self, estimulo, resultado_esperado):
        """
            LMS Estocastico
        """
        self.calcular(estimulo)
        error = (resultado_esperado - self.y)
        delta = (self.alfa * error) * estimulo
        # print(type(error), type(self.bias), type(estimulo), type(delta))
        
        # Actualizacion
        self.pesos_sinapticos = self.pesos_sinapticos + delta
        self.bias = self.bias + error * self.alfa
        return self.pesos_sinapticos

    def entrenar_batch(self, estimulos, resultados_esperados):
        """
            LMS Batch
        """
        delta = np.zeros(self.n_entradas)
        for estimulo, respuesta in zip(estimulos, resultados_esperados):
            self.calcular(estimulo)
            error = (respuesta - self.y)
            delta += error * estimulo
            self.bias += error * self.alfa
        self.pesos_sinapticos = self.pesos_sinapticos + self.alfa * delta

    # Por seguir con el algoritmo
    def lineal(self, a):
        """
            Funcion de transferencia lineal
        """
        return a

    def describir(self):
        print("PESOS SINAPTICOS: ", self.pesos_sinapticos)
        print("RESPUESTA: ", self.y)


class Capa:

    def __init__(self, n_adaline, n_entradas, alfa=0.1):
        self.n_adaline = n_adaline
        self.n_entradas = n_entradas
        self.adalines = [Adaline(n_entradas, alfa=alfa)
                             for i in range(n_adaline)]

    def entrenar(self, entrada, respuesta):
        """
            Entrenamiento estocastico de las neuronas
        """
        for i in range(self.n_adaline):
            if respuesta == i:
                self.adalines[i].entrenar_estocastico(entrada, 1)
            else:
                self.adalines[i].entrenar_estocastico(entrada, 0)

    def calcular(self, estimulo):
        """
            Calcular el resultado dado un estimulo a las neuronas
        """
        respuesta = []
        for i in range(self.n_adaline):
            respuesta.append(self.adalines[i].calcular(estimulo))

        return respuesta


def entrenar(capa, epocas=1):
    """
    Entrenar las neuronas de una capa
    """
    with open('../mnist_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',',
                                quoting=csv.QUOTE_NONNUMERIC)
        start = time.time()
        leido = 0
        for _ in range(epocas):
            leido = 0
            for row in csv_reader:
                # Dividimos los datos por 255 para tenerlos 
                # en un rango entre 0 y 1
                datos = np.array(row[1:]) / 255
                capa.entrenar(datos, row[0])
                leido += 1
                
            # Devolvemos el apuntador del archivo al inicio
            csv_file.seek(0)
        print(f"Tiempo: {time.time() - start}seg")
        print(f"No de datos usados para el entrenamiento: {leido}")


def verificar(capa):
    """
    Verificar si las neuronas de una capa dada estan entrenadas
    """
    with open('../mnist_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',',
                                quoting=csv.QUOTE_NONNUMERIC)
        line_count = 0
        resultados_buenos = 0
        resultados_malos = 0
        for row in csv_reader:
            resultado = row[0]
            datos = np.array(row[1:])
            res = capa.calcular(datos)
            rep = [0,res[0]]
            for i, j in enumerate(res):
                if j > rep[1] :
                    rep[0] = i
                    rep[1] = j

            if rep[0] == resultado:
                resultados_buenos += 1
            else:
                resultados_malos += 1
            line_count += 1

        
        print(f"Datos leido: {line_count}")
        print(
            f"Porcentaje de resultados buenos: {resultados_buenos / line_count * 100}%")
        print(
            f"Porcentaje de resultados malos: {resultados_malos / line_count * 100}%")


def script_mistico():
    # ENTRENAMIENTO CON TASA DE APRENDIZAJE = 0.1
    c = Capa(10, 784, 0.1)
    entrenar(c,EPOCAS)
    print("Resultados de entrenar con taza de aprendizaje 0.1.")
    verificar(c)
    print("\n#######################\n")


    # ENTRENAMIENTO CON TASA DE APRENDIZAJE = 0.01
    c = Capa(10, 784, 0.01)
    entrenar(c, EPOCAS)
    print("Resultados de entrenar con taza de aprendizaje 0.01.")
    verificar(c)
    print("\n#######################\n")

    # ENTRENAMIENTO CON TASA DE APRENDIZAJE = 0.001
    c = Capa(10, 784, 0.001)
    entrenar(c, EPOCAS)
    print("Resultados de entrenar con taza de aprendizaje 0.001.")
    verificar(c)
