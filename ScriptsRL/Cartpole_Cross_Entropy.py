import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras  import Sequential
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pickle as pkl
import random
import time
from PickleSaver import PickleSaver

SEED = 0
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)



def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=SEED)

HIDDEN_SIZE = 128

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#tf.debugging.set_log_device_placement(True)

data_to_dump = {}
data_to_dump["Mean Return"] = []
pklsave = PickleSaver("Cartpole_Cross_Entropy")
# Neuronal Net
model = Sequential([
                    Dense(HIDDEN_SIZE,activation="relu"),
                    Dense(HIDDEN_SIZE,activation="relu"),
                    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy", # loss function here
    optimizer="adam",
    metrics=["accuracy"])

env=gym.make("CartPole-v1")
env.reset()[0]

def RandomAgent(obs):
  return env.action_space.sample()

def CrossEntropyAgent(obs):
  obs= obs.reshape(1,4)
  prob=model.predict(obs,verbose=0)[0][0] # Probability that the action to be 1
  a=np.random.choice([1,0],p=[prob,1-prob])
  return a

S=set()
n=0
start = time.time()
for j in range(5):

  # Collect data
  trajectories=[]
  episodes=10
  for e in range(episodes):
    done=False
    trunc=False
    obs=env.reset()[0]
    S.add(tuple(obs)) 
    n+=1
    trajectory=[]
    while not done and not trunc:
      action=CrossEntropyAgent(obs)
      n_obs,r,done,trunc,info = env.step(action)
      trajectory.append((obs,action,r))
      obs=n_obs
      S.add(tuple(obs))
      n+=1
      trajectories.append(trajectory)

  # Looking index for trayectory wiht maximun return R
  Ri=np.zeros(len(trajectories))
  for i,trajectory in enumerate(trajectories):
   Ri[i]=len(trajectory)
  i=np.argmax(Ri)
  print('iteration',j,'states visited',len(S),'Mean Return =',Ri.mean(),)

  # Training
  X=np.array([s for s,a,r in trajectories[i]])
  y=np.array([a for s,a,r in trajectories[i]])
  history = model.fit(X, y, epochs=100, verbose=0)
  data_to_dump["Mean Return"].append(Ri.mean())
  pklsave.save_data("Mean Return", Ri.mean())
  #with open('Mean_returns.pickle', 'wb') as handle:
  #  pkl.dump(data_to_dump, handle, protocol=pkl.HIGHEST_PROTOCOL)

print(n,len(S))
pklsave.save_int("Execution Time", time.time()-start)
model.save("my_model.keras")

#with open('Mean_returns.pickle', 'wb') as handle:
#    pkl.dump(data_to_dump, handle, protocol=pkl.HIGHEST_PROTOCOL)

