import keras.backend as K
import keras
import numpy as np
import h5py
import copy
import math

//give the path to train dataset
file=h5py.File("path/to/train-features/and/train-labels","r")
features=file.get("features").value
labels=file.get("labels").value


//give the path to test dataset
file=h5py.File("path/to/test-features/and/test-labels","r")
test_features=file.get("features").value
test_labels=file.get("labels").value

sol_per_pop = 6                //change it as per your need
num_generations = 8            //change it as per your need
num_weights = features.shape[1]
pop_size = (sol_per_pop, num_weights)


// random initialization of population
new_population = np.random.randint(0,2,pop_size)
for i in range(sol_per_pop):
    for j in range(num_weights):
        prob = np.random.uniform(0.0, 1.0)
        if prob <= 0.4:
            new_population[i][j] = 0
        else:
            new_population[i][j] = 1
            

// change the model as per your need, but keep the input shape same
def cal_fitness(new_population, features):
   fitness = []
   for i in range(new_population.shape[0]):
        print(i+1)
        cur_features = copy.deepcopy(features)
        cur_features = np.take(features, np.where(new_population[i]), axis=1)
        temp = cur_features.shape[2]
        cur_features = cur_features.reshape((features.shape[0], temp))
        model = keras.Sequential(
        [
         keras.layers.Dense(512, input_shape=(cur_features.shape[1],)),
         keras.layers.Activation("relu"),
         keras.layers.Dense(256, kernel_regularizer = keras.regularizers.l2(0.0004)),
         keras.layers.Activation("sigmoid"),
         keras.layers.Dense(8, kernel_regularizer = keras.regularizers.l2(0.0004))
        ]
        )
        model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.002), 
                  loss = mse,
                  metrics = ['accuracy'])
        model.fit(cur_features, labels, epochs=100)
        cur_features_test = copy.deepcopy(test_features)
        cur_features_test = np.take(test_features, np.where(new_population[i]), axis=1)
        temp_test = cur_features_test.shape[2]
        cur_features_test = cur_features_test.reshape((test_features[0], temp_test))
        test_loss, test_accuracy = model.evaluate(cur_features_test, test_labels, verbose = 2)
        fitness.append(-1*test_loss)
   tot_fitness.append(fitness)
   return fitness
   
 
 def select_parents(fitness):
    max1 = -100
    max2 = -100
    ind1, ind2 = -1, -1
    for i in range(len(fitness)):
        if(fitness[i] > max1):
            max2 = max1
            max1 = fitness[i]
            ind2 = ind1
            ind1 = i
        elif(fitness[i] > max2):
            ind2 = i
            max2 = fitness[i]
    parents = [ind1, ind2]
    return parents
    
    
def select_bad_individuals(fitness):
    min1 = math.inf 
    min2 = math.inf
    ind1, ind2 = -1, -1
    for i in range(len(fitness)):
        if(fitness[i] < min1):
            min2 = min1
            min1 = fitness[i]
            ind2 = ind1
            ind1 = i
        elif(fitness[i] < min2):
            ind2 = i
            min2 = fitness[i]
    bad_pop = [ind1, ind2]
    return bad_pop
    
    
for generation in range(num_generations):
  print("Generation :" , generation+1)
  fitness = cal_fitness(new_population, features)
  # selection of parents
  parents = select_parents(fitness)

  # crossover
  child1 = []
  child2 = []
  for i in range(num_weights):
      prob = np.random.randint(0, 2)
      if prob == 0:
        child1.append(new_population[parents[0]][i])
        child2.append(new_population[parents[1]][i])
      else:
        child1.append(new_population[parents[0]][i])
        child2.append(new_population[parents[0]][i])

  # mutation
  for i in range(num_weights):
      prob = np.random.uniform(0.0, 1.0)
      if prob <= 0.1:
          if child1[i] == 0:
            child1[i] = 1
          else:
            child1[i] = 0
  for i in range(num_weights):
      prob = np.random.uniform(0.0, 1.0)
      if prob <= 0.1:
          if child2[i] == 0:
            child2[i] = 1
          else:
            child2[i] = 0
  # selecting individuals with low fitness
  bad_pop = select_bad_individuals(fitness)

  # updating the population
  new_population[bad_pop[0], :] = child1[:]
  new_population[bad_pop[1], :] = child2[:]
  pathCur = '/path/to/save/the/results/after/each/generation/' + str(generation+1)
  np.save(pathCur, new_population)
  
  print("The most optimized fitness value is : ", fitness[parents[0]])
