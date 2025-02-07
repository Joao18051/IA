import numpy as np

sampleSize = 100
trainingSize = 0.8

x = np.random.uniform(-10, 10, size=sampleSize)
trainingSize = int(len(x)* trainingSize)
#Ruído gaussiano media 0 desvio padrão 2 (Draw random samples from a normal (Gaussian) distribution.)
epsilon = np.random.normal(loc=0, scale=2, size=sampleSize) 

y = 3 *x +5 +epsilon

xTrainingData = x[trainingSize:]
yTrainingData = y[trainingSize:]

xTestingData = x[:trainingSize]
yTestingData = y[:trainingSize]

del x
del y

#Objetivo achar beta0 e beta1​
#Adiciona coluna de 1s já que é a versão simplificada
xTrainingData = np.vstack((np.ones(len(xTrainingData)), xTrainingData)).T

#Função que calcula a pseudo inversa da matriz "beta = (X^T X)^(-1) X^T y"
beta = np.linalg.pinv(xTrainingData) @ yTrainingData

# Exibindo os coeficientes encontrados
print(f"Beta 0 (intercepto): {beta[0]:.4f}")
print(f"Beta 1 (inclinação de x): {beta[1]:.4f}")

#Adiciona coluna de 1s na matriz de testes já que é a versão simplificada
xTestingData = np.vstack((np.ones(len(xTestingData)), xTestingData)).T
#A previsão é feita utilizando y = vetor x * vetor Beta
yPrediction = xTestingData @ beta
#Erro Quadrático Médio
mse = np.mean((yPrediction -yTestingData) ** 2)
print(f"Erro Quadrático médio: {mse:.4f}")