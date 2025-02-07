import numpy as np

np.random.seed(50)

sampleSize = 100
trainingSize = int(0.8 * sampleSize)

x = np.random.uniform(-10, 10, size=sampleSize)
#Ruído gaussiano media 0 desvio padrão 2 (Draw random samples from a normal (Gaussian) distribution.)
epsilon = np.random.normal(loc=0, scale=2, size=sampleSize) 

y = 3 *x +5 +epsilon

xTrainingData = x[:trainingSize]
yTrainingData = y[:trainingSize]

xTestingData = x[trainingSize:]
yTestingData = y[trainingSize:]

del x
del y

xTrainingData = np.vstack((np.ones(len(xTrainingData)), xTrainingData)).T
xTestingData = np.vstack((np.ones(len(xTestingData)), xTestingData)).T

#Pesos e bias
weights = 2 *np.random.random((2, 1)) - 1
learningRate = 0.001

def mseLoss(yPred, yTrue):
    return np.mean((yPred -yTrue) **2)

for epoch in range(1000):
    yPrediction = xTrainingData @ weights

    loss = mseLoss(yPrediction, yTrainingData.reshape(-1, 1))
    gradientWeight = -2 *xTrainingData.T @ (yTrainingData.reshape(-1, 1) -yPrediction) /trainingSize
    weights -= learningRate *gradientWeight

yPrediction = xTestingData @ weights  
mse = mseLoss(yPrediction, yTestingData.reshape(-1, 1))
print(f"Erro Quadrático médio: {mse:.4f}")