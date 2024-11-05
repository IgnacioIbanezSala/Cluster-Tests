import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras  import Sequential
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pickle as pkl

env=gym.make("CartPole-v1")

def RandomAgent(obs):
  return env.action_space.sample()

def ReinforceAgent(obs):
  prob=Reinforce_Net(obs.reshape(1,4))
  #print("--",prob,prob.numpy())
  a=np.random.choice(2,p=prob.numpy().reshape(2))
  return a




HIDDEN_SIZE = 128
Reinforce_Net = Sequential([
                    Dense(HIDDEN_SIZE,activation="relu"),
                    Dense(HIDDEN_SIZE,activation="relu"),
                    Dense(2,activation='softmax')
])

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#tf.debugging.set_log_device_placement(True)

data_to_dump = {}
data_to_dump["Mean Return"] = []

n=0
for it in range(10000):
    # Collect data
    trajectories=[]
    episodes=10
    if n >=3 : break
    for _ in range(episodes):
      done=False
      obs=env.reset()[0]
      trajectory=[]
      while(not done):
        action=ReinforceAgent(obs)
        new_obs,r,done,trunc,info = env.step(action)
        trajectory.append((obs,action,r))
        obs=new_obs
        if trunc:
          print('500 Steps!')
          n+=1
          break
      trajectories.append(trajectory)

    # Looking index for trayectory wiht maximun return R
    Ri=np.zeros(len(trajectories))
    for i,trajectory in enumerate(trajectories):
      Ri[i]=len(trajectory)
    i=np.argmax(Ri)

    X=np.array([s for s,a,r in trajectories[i]])
    actions=np.array([a for s,a,r in trajectories[i]])
    rewards=np.array([r for s,a,r in trajectories[i]])

    # training
    lr=0.001
    gamma=1
    for t,x  in enumerate(X):
      old_step=Reinforce_Net(x.reshape(1,4))

      #for t in range(0,len(rewards)):
      G=0
      for k in range(t,len(rewards)):
        G+=gamma**(k-t)*rewards[k]

      with tf.GradientTape() as tape:
        # Forward pass
        y = Reinforce_Net(x.reshape(1,4))[0][actions[t]]
        loss = tf.math.log(y)

      # Calculate gradients with respect to every trainable variable
      grad = tape.gradient(loss,Reinforce_Net.trainable_variables)
      w=[]

      for w_old, g in zip(Reinforce_Net.get_weights(), grad):
        w.append(w_old + G*lr*g)
      Reinforce_Net.set_weights(w)
      #print('G=',G,', a=',actions[t],', old_step=',old_step.numpy()[0],', new_step=',Reinforce_Net(x.reshape(1,4)).numpy()[0])
    #print(len(trajectories[i]))
    data_to_dump["Mean Return"].append(Ri.mean())

Reinforce_Net.save("Reinforce_model.keras")

with open('Mean_returns_Reinforce.pickle', 'wb') as handle:
    pkl.dump(data_to_dump, handle, protocol=pkl.HIGHEST_PROTOCOL)


