import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras  import Sequential
import numpy as np
import matplotlib.pyplot as plt
import gym

HIDDEN_SIZE = 128

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

env=gym.make("CartPole-v1",new_step_api=True)
env.reset()

def RandomAgent(obs):
  return env.action_space.sample()

def CrossEntropyAgent(obs):
  prob=model.predict(np.array([obs]),verbose=0)[0][0] # Probability that the action to be 1
  a=np.random.choice([1,0],p=[prob,1-prob])
  return a

S=set()
n=0
for j in range(50):
  # Collect data
  trajectories=[]
  episodes=10
  for e in range(episodes):
    done=False
    trunc=False
    obs=env.reset()
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
    #print('iteration',j,'episode',e,'Return =',len(trajectory))
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

print(n,len(S))

one=False
trunc=True

not done and not trunc, done ,trunc