import random
import numpy as np
import pandas as pd
#%%
def ga(model, x_train, y_train, population_size=10, mutation_rate=0.01, generations=100):
    population = []
    for i in range(population_size):
        weights = []
        for layer in model.layers:
            layer_weights = []
            for w in layer.get_weights():
                layer_weights.append(np.random.uniform(low=-1.0, high=1.0, size=w.shape))
            weights.append(layer_weights)
        population.append(weights)

    for gen in range(generations):
        fitness_scores = []
        for p in population:
            model.set_weights(p)
            fitness_scores.append(model.evaluate(x_train, y_train, verbose=0)[1])

        print(f"Generation {gen} | Best fitness: {max(fitness_scores)}")

        elite_indices = np.argsort(fitness_scores)[-population_size//2:]
        elites = [population[i] for i in elite_indices]
        offspring = []
        for i in range(population_size - len(elites)):
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            child = []
            for j in range(len(parent1)):
                layer1 = parent1[j]
                layer2 = parent2[j]
                layer_child = []
                for k in range(len(layer1)):
                    w1 = layer1[k]
                    w2 = layer2[k]
                    crossover_point = random.randint(0, len(w1) - 1)
                    child_weights = np.concatenate((w1[:crossover_point], w2[crossover_point:]))
                    layer_child.append(child_weights)
                child.append(layer_child)
            offspring.append(child)
        for child in offspring:
            for i in range(len(child)):
                layer = child[i]
                for j in range(len(layer)):
                    w = layer[j]
                    for k in range(len(w)):
                        if random.random() < mutation_rate:
                            w[k] += np.random.normal(scale=0.1)
        population = elites + offspring
    best_weights = []
    for i in range(len(model.layers)):
        layer_weights = []
        for w in population[0][i]:
            layer_weights.append(w)
        best_weights.append(np.array(layer_weights))
    return best_weights

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
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (11,)),
    keras.layers.Dense(16,activation = tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
best_weights = ga(model, x_train, y_train, population_size=10, mutation_rate=0.01, generations=100)
model.set_weights(best_weights)
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)









