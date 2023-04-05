import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
#%%
def aco(model, x_train, y_train, num_ants=10, num_iterations=100, evaporation_rate=0.5, alpha=1.0, beta=1.0):
    num_weights = sum(len(layer.get_weights()[i].flatten()) for i in range(len(model.layers)) for layer in model.layers)
    pheromone = np.random.rand(num_weights)
    best_solution = None
    best_fitness = float('-inf')

    for i in range(num_iterations):
        solutions = []
        for j in range(num_ants):
            solution = []
            for layer in model.layers:
                weights = []
                for w in layer.get_weights():
                    flattened_w = w.flatten()
                    for k in range(len(flattened_w)):
                        pheromone_index = np.ravel_multi_index((layer.output_shape[1:], (len(flattened_w),), k), pheromone.shape)
                        weight = np.random.normal(loc=pheromone[pheromone_index], scale=1.0)
                        weights.append(weight)
                    weights = np.array(weights).reshape(w.shape)
                solution.append(weights)
            solutions.append(solution)
        fitness = []
        for solution in solutions:
            model.set_weights(solution)
            fitness.append(model.evaluate(x_train, y_train, verbose=0)[1])
        for solution, fit in zip(solutions, fitness):
            for layer_index, layer in enumerate(model.layers):
                for weight_index, w in enumerate(layer.get_weights()):
                    flattened_w = w.flatten()
                    for k in range(len(flattened_w)):
                        pheromone_index = np.ravel_multi_index((layer.output_shape[1:], (len(flattened_w),), k), pheromone.shape)
                        pheromone[pheromone_index] = (1 - evaporation_rate) * pheromone[pheromone_index] + alpha * fit * solution[layer_index][weight_index][k]
        if max(fitness) > best_fitness:
            best_solution = solutions[np.argmax(fitness)]
            best_fitness = max(fitness)

    return best_solution
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('D:/DATASETS/Bank_Personal_Loan_Modelling.csv')
df = df.drop(['ZIP Code','ID'],axis = 1)
df.columns = [ 'Age', 'Experience', 'Income', 'Family', 'CCAvg','Education', 'Mortgage','Securities Account','CD Account', 'Online', 'CreditCard','Personal Loan']
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

#%%
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (11,)),
    keras.layers.Dense(16,activation = tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
opt = aco(model, x_train, y_train,)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
#%%

