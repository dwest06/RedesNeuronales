import numpy as np
import csv
import time

EPOCAS = 50


class Perceptron:

    def __init__(self, n_entradas, bias=0, alfa=0.1):
        self.pesos_sinapticos = np.random.uniform(
            low=-0.05, high=0.05, size=(n_entradas,))
        self.n_entradas = n_entradas
        self.bias = bias
        self.alfa = alfa

    def calcular(self, estimulo):
        # Realizar operaciones vectoriales de numpy
        y = np.dot(self.pesos_sinapticos, estimulo)
        y += self.bias
        self.y = self.umbral(y)
        return self.y

    def entrenar(self, estimulo, resultado_esperado):
        # alfa siendo la tasa de aprendizaje
        self.calcular(estimulo)
        delta = self.alfa * (resultado_esperado - self.y) * estimulo
        self.pesos_sinapticos = self.pesos_sinapticos + delta
        return self.pesos_sinapticos

    def umbral(self, a, threshold=0):
        return 1 if a >= threshold else 0

    def describir(self):
        print("PESOS SINAPTICOS: ", self.pesos_sinapticos)
        print("RESPUESTA: ", self.y)


class Capa:

    def __init__(self, n_perceptron, n_entradas, alfa=0.1):
        self.n_perceptron = n_perceptron
        self.n_entradas = n_entradas
        self.perceptrones = [Perceptron(n_entradas, alfa=alfa)
                             for i in range(n_perceptron)]

    def entrenar(self, entrada, respuesta):
        for i in range(self.n_perceptron):
            if respuesta == i:
                self.perceptrones[i].entrenar(entrada, 1)
            else:
                self.perceptrones[i].entrenar(entrada, 0)

    def calcular(self, estimulo):
        respuesta = []
        for i in range(self.n_perceptron):
            respuesta.append(self.perceptrones[i].calcular(estimulo))

        return respuesta


# Funcion de epocas, util para entrenar el perceptron
def epocas(p, es, r, iteraciones=10):
    for j in range(iteraciones):
        for i in range(len(es)):
            print(es[i], r[i])
            p.entrenar(r[i])
            p.describir()
            print("---------------------")

    print("Numero de iteraciones: ", iteraciones)


def entrenar(capa, epocas=1):
    """
    Entrenar una capa 
    """
    with open('mnist_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',',
                                quoting=csv.QUOTE_NONNUMERIC)

        for _ in range(epocas):
            for row in csv_reader:
                capa.entrenar(np.array(row[1:]), row[0])
            csv_file.seek(0)

def verificar(capa):
    """
    Verificar si las neuronas estan entrenadas
    """
    with open('mnist_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',',
                                quoting=csv.QUOTE_NONNUMERIC)
        line_count = 0
        resultados_buenos = 0
        resultados_malos = 0
        for row in csv_reader:
            resultado = row[0]
            datos = np.array(row[1:])
            res = capa.calcular(datos)
            # print(resultado, res)
            if res[int(resultado)] == 1:
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
