import numpy as np
import pandas as pd
#%%
def pso(func, lb, ub, max_iter=100, swarm_size=100, c1=2.0, c2=2.0, w=0.7):
    
    swarm = np.random.uniform(low=lb, high=ub, size=(swarm_size, len(lb)))
    velocity = np.zeros_like(swarm)
    best_swarm_position = swarm.copy()
    best_swarm_fitness = np.full((swarm_size,), np.inf)
    best_particle_position = swarm.copy()
    best_particle_fitness = np.full((swarm_size,), np.inf)
    
    
    for i in range(max_iter):
        swarm_fitness = func(swarm)
        update = swarm_fitness < best_particle_fitness
        best_particle_position[update] = swarm[update]
        best_particle_fitness[update] = swarm_fitness[update]
        
        best_index = np.argmin(best_particle_fitness)
        
        if best_particle_fitness[best_index] < best_swarm_fitness:
            best_swarm_position = best_particle_position[best_index].copy()
            best_swarm_fitness = best_particle_fitness[best_index].copy()
            
        
        r1 = np.random.uniform(size=(swarm_size, len(lb)))
        r2 = np.random.uniform(size=(swarm_size, len(lb)))
        cognitive = c1 * r1 * (best_particle_position - swarm)
        social = c2 * r2 * (best_swarm_position - swarm)
        velocity = w * velocity + cognitive + social
        swarm = swarm + velocity
        
        swarm = np.clip(swarm, lb, ub)
    
    
    return best_swarm_position, best_swarm_fitness
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
best_weights = pso(model, x_train, y_train)
model.set_weights(best_weights)
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

