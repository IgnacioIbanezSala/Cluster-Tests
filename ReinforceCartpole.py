import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras  import Sequential
import numpy as np
import matplotlib.pyplot as plt
import gym



env=gym.make("CartPole-v1",new_step_api=True)

def RandomAgent(obs):
  return env.action_space.sample()

def ReinforceAgent(obs):
  prob=Reinforce_Net(obs.reshape(1,4))
  a=np.random.choice(2,p=prob.numpy().reshape(2))
  return a




HIDDEN_SIZE = 128
Reinforce_Net = Sequential([
                    Dense(HIDDEN_SIZE,activation="relu"),
                    Dense(HIDDEN_SIZE,activation="relu"),
                    Dense(2,activation='softmax')
])

# Collect data
trajectories=[]
episodes=10
for _ in range(episodes):
  done=False
  obs=env.reset()
  trajectory=[]
  while(not done):
    action=ReinforceAgent(obs)
    n_obs,r,done,trunc,info = env.step(action)
    trajectory.append((obs,action,r))
    obs=n_obs
    if trunc:
      print('error : anomaly output')
      break
  trajectories.append(trajectory)

# Looking index for trayectory wiht maximun return R
Ri=np.zeros(len(trajectories))
for i,trajectory in enumerate(trajectories):
  Ri[i]=len(trajectory)
i=np.argmax(Ri)

Ri,Ri[i]

X=np.array([s for s,a,r in trajectories[i]])
actions=np.array([a for s,a,r in trajectories[i]])
rewards=np.array([r for s,a,r in trajectories[i]])



G_acc=[]
for it in range(100):
    # Collect data
    trajectories=[]
    episodes=10
    for _ in range(episodes):
      done=False
      obs=env.reset()
      trajectory=[]
      while(not done):
        action=ReinforceAgent(obs)
        n_obs,r,done,trunc,info = env.step(action)
        trajectory.append((obs,action,r))
        obs=n_obs
        if trunc:
          print('error : anomaly output')
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
    G_acc.append(len(trajectories[i]))
plt.plot(G_acc)
plt.show()

