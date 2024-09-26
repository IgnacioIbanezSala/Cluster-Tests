import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras  import Sequential
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


HIDDEN_SIZE = 128

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)

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
  prob=model.predict(obs, verbose=0)# Probability that the action to be 1
  a=np.random.choice([1,0],p=[prob[0][0],1-prob[0][0]])
  return a

n=0
for j in range(50):
  # Collect data
  trajectories=[]
  episodes=10
  for e in range(episodes):
    done=False
    trunc=False
    obs=env.reset()[0]
  
    n+=1
    trajectory=[]
    while not done and not trunc:
      obs = np.reshape(obs, [1,env.observation_space.shape[0]])
      action=CrossEntropyAgent(obs)
      n_obs,r,done,trunc,info = env.step(action)
      trajectory.append((obs,action,r))
      
      

      n+=1
      trajectories.append(trajectory)
    #print('iteration',j,'episode',e,'Return =',len(trajectory))
  # Looking index for trayectory wiht maximun return R
  Ri=np.zeros(len(trajectories))
  for i,trajectory in enumerate(trajectories):
   Ri[i]=len(trajectory)
  i=np.argmax(Ri)
  print('iteration',j, 'Mean Return =',Ri.mean(),)
  # Training
  X=np.array([s for s,a,r in trajectories[i]])
  y=np.array([a for s,a,r in trajectories[i]])

#print(n,len(S))

one=False
trunc=True

not done and not trunc, done ,trunc