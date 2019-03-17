# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:20:20 2019

@author: ZULFA
"""
import math
import random
import csv
import matplotlib.pyplot as plt

with open('iris.csv') as csv_file:
    dataset = list(csv.reader(csv_file))

# Change string value to numeric
for row in dataset:
    row[:4] = [float(row[j]) for j in range(len(row)-1)]
    row[4] = ["setosa", "versicolor", "virginica"].index(row[4])

# Split x and y (feature and target)
random.shuffle(dataset)
datatrain = dataset[:120]
datatest = dataset[120:]
train_X = [data[:4] for data in datatrain]
train_y = [data[4] for data in datatrain]
test_X = [data[:4] for data in datatest]
test_y = [data[4] for data in datatest]

def matrix_mul_bias(A, B, bias): # Matrix multiplication (for Testing)
    C = [[0 for i in range(len(B[0]))] for i in range(len(A))]    
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C

def vec_mat_bias(A, B, bias): # Vector (A) x matrix (B) multiplication
    C = [0 for i in range(len(B[0]))]
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    return C

def mat_vec(A, B): # Matrix (A) x vector (B) multipilicatoin (for backprop)
    C = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C

def sigmoid(A, deriv=False):
    if deriv: # derivation of sigmoid (for backprop)
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

def forward_propagation(x,weight,bias,bias_2,weight_2):
    # Forward propagation
    h_1 = vec_mat_bias(x, weight, bias)
    X_1 = sigmoid(h_1)
    h_2 = vec_mat_bias(X_1, weight_2, bias_2)
    X_2 = sigmoid(h_2)
    return h_1,X_1,h_2,X_2

def backward_propagation(X):
    cost_total = 0  
    for idx, x in enumerate(X): # Update for each data; SGD
        h_1,X_1,h_2,X_2=forward_propagation(x,weight,bias,bias_2,weight_2)
        
        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(train_y[idx])] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in range(3):
            eror +=  (target[i] - X_2[i]) ** 2 
        eror=0.5*eror
        cost_total += eror

        # Backward propagation Training
        # Update weight_2 and bias_2 (layer 2)
        delta_2 = []
        for j in range(neuron[2]):
            delta_2.append(-1*(target[j]-X_2[j]) * X_2[j] * (1-X_2[j]))

        for i in range(neuron[1]):
            for j in range(neuron[2]):
                weight_2[i][j] -= alfa * (delta_2[j] * X_1[i])
                bias_2[j] -= alfa * delta_2[j]
        
        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_2)
        for j in range(neuron[1]):
            delta_1[j] = delta_1[j] * X_1[j] * (1-X_1[j])
        
        for i in range(neuron[0]):
            for j in range(neuron[1]):
                weight[i][j] -=  alfa * (delta_1[j] * x[i])
                bias[j] -= alfa * delta_1[j]
    return cost_total

# Define parameter
alfa = 0.8
epoch = 100
neuron = [4, 3, 3] # number of neuron each layer
error = []
error_t=[]
accuracy = []
accuracy_t = []

# Initiate weight and bias with 0 value
weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
weight_2 = [[0 for j in range(neuron[2])] for i in range(neuron[1])]
bias = [0 for i in range(neuron[0])]
bias_2 = [0 for i in range(neuron[1])]

# Initiate weight with random between -1.0 ... 1.0
for i in range(neuron[0]):
    for j in range(neuron[1]):
        weight[i][j] = 2 * random.random() - 1

for i in range(neuron[1]):
    for j in range(neuron[2]):
        weight_2[i][j] = 2 * random.random() - 1
        
for e in range(epoch):
    #Training
    cost_total = 0
    cost_total=backward_propagation(train_X)
    cost_total /= len(train_X)
    error.append(cost_total)
    
    res = matrix_mul_bias(train_X, weight, bias)
    res_2 = matrix_mul_bias(res, weight_2, bias)
    preds = []
    # Get prediction
    for r in res_2:
        preds.append(max(enumerate(r), key=lambda x:x[1])[0])
    acc = 0.0
    for i in range(len(preds)):
        if preds[i] == int(train_y[i]):
            acc += 1        
    acc = acc / len(preds) * 100
    accuracy.append(acc)
    
    #Validasi
    cost_total = 0
    for idx, x in enumerate(test_X): # Update for each data; SGD
        h_1,X_1,h_2,X_2=forward_propagation(x,weight,bias,bias_2,weight_2)
        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(train_y[idx])] = 1
        eror = 0
        for i in range(3):
            eror +=  (target[i] - X_2[i]) ** 2 
        eror=0.5*eror
        cost_total += eror
    cost_total /= len(test_X)
    error_t.append(cost_total)
    
    res = matrix_mul_bias(test_X, weight, bias)
    res_2 = matrix_mul_bias(res, weight_2, bias)
    preds_t = []
    # Get prediction
    for r in res_2:
        preds_t.append(max(enumerate(r), key=lambda x:x[1])[0])
    acc = 0.0
    for i in range(len(preds_t)):
        if preds_t[i] == int(test_y[i]):
            acc += 1        
    acc = acc / len(preds_t) * 100
    accuracy_t.append(acc)
            

# Plot Total Loss vs Epoch
plt.plot(range(epoch),error)
plt.plot(range(epoch),error_t,color='red')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()

plt.plot(range(epoch),accuracy)
plt.plot(range(epoch),accuracy_t,color='red')
plt.ylabel('Accuracy %')
plt.xlabel('Epoch')
plt.show()
