import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.models import clone_model
from tensorflow.keras  import Sequential
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Neuronal Net Pi(a/s)
HIDDEN_SIZE = 128
policy = Sequential([
                    Dense(HIDDEN_SIZE,activation="relu"),
                    Dense(HIDDEN_SIZE,activation="relu"),
                    Dense(2, activation="softmax")
])
old_policy=clone_model(policy)
#old_policy.set_weights(policy.get_weights())

# Neuronal Net V(s)
HIDDEN_SIZE=128
V_Net = Sequential([
                    Dense(HIDDEN_SIZE,activation="relu"),
                    Dense(HIDDEN_SIZE,activation="relu"),
                    Dense(1, activation="linear")
                   ])


env=gym.make("CartPole-v1")

def RandomAgent(obs):
  return env.action_space.sample()

def CrossEntropyAgent(obs):
  prob=model.predict(np.array([obs]),verbose=0)[0][0] # Probability that the action to be 1
  a=np.random.choice([1,0],p=[prob,1-prob])
  return a

def ReinforceAgent(obs):
  prob=policy(np.reshape(obs, [1,env.observation_space.shape[0]]))
  a=np.random.choice(2,p=prob.numpy().reshape(2))
  return a

# Collect data
trajectories=[]
episodes=10
for _ in range(episodes):
  done=False
  obs=env.reset()[0]
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

gamma=1
lr=0.01

# V network update
with tf.GradientTape() as tape:
    r = tf.Variable(rewards[:-1], dtype=tf.float32) # Change the type of r to float32
    Q= tf.reshape(r,[r.shape[0],1]) + gamma*V_Net(X[1:]) # Q(s,a)
    A= Q - V_Net(X[:1]) # Advantage
    loss = tf.reduce_mean((A)**2)

grad = tape.gradient(loss,V_Net.trainable_variables)
w=[]
for w_old, g in zip(V_Net.get_weights(), grad):
  w.append(w_old - lr*g)
V_Net.set_weights(w)





# policy update

epsilon=0.2
lr=0.01
# Policy network update

with tf.GradientTape() as tape:
  policy_probs = tf.gather(policy(X)[0], actions)
  old_policy_probs = tf.gather(old_policy(X)[0], actions)
  policy_ratio = policy_probs / old_policy_probs
  clip=tf.clip_by_value(policy_ratio, 1-epsilon, 1+epsilon)
  L=tf.math.minimum(policy_ratio*A,clip*A)
  loss=tf.reduce_mean(L)
grad = tape.gradient(loss,policy.trainable_variables)

old_policy.set_weights(policy.get_weights()) # policy <-- old_policy

# update weights
w=[]
for w_old, g in zip(policy.get_weights(), grad):
  w.append(w_old + lr*g)
policy.set_weights(w)

G_acc=[]
for _ in range(100):

  # Collect data
  trajectories=[]
  episodes=10
  for _ in range(episodes):
    done=False
    obs=env.reset()[0]
    trajectory=[]
    while(not done):
      action=ReinforceAgent(obs)
      n_obs,r,done,trunc,info = env.step(action)
      trajectory.append((obs,action,r))
      obs=n_obs
      if trunc:
        print('error : anomaly output',len(trajectory))
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

  gamma=1
  lr=0.01
  # V network update
  with tf.GradientTape() as tape:
      r = tf.Variable(rewards[:-1], dtype=tf.float32) # Change the type of r to float32
      Q= tf.reshape(r,[r.shape[0],1]) + gamma*V_Net(X[1:]) # Q(s,a)
      A= Q - V_Net(X[:1]) # Advantage
      loss = tf.reduce_mean((A)**2)

  grad = tape.gradient(loss,V_Net.trainable_variables)
  w=[]
  for w_old, g in zip(V_Net.get_weights(), grad):
    w.append(w_old - lr*g)
  V_Net.set_weights(w)

  # Policy network update

  with tf.GradientTape() as tape:
    policy_probs = tf.gather(policy(X)[0], actions)
    old_policy_probs = tf.gather(old_policy(X)[0], actions)
    policy_ratio = policy_probs / old_policy_probs
    clip=tf.clip_by_value(policy_ratio, 1-epsilon, 1+epsilon)
    L=tf.math.minimum(policy_ratio*A,clip*A)
    loss=tf.reduce_mean(L)

  grad = tape.gradient(loss,policy.trainable_variables)

  old_policy.set_weights(policy.get_weights()) # policy <-- old_policy

  w=[]
  for w_old, g in zip(policy.get_weights(), grad):
    w.append(w_old + lr*g)
  policy.set_weights(w)

  G_acc.append(Ri[i])
plt.plot(G_acc)

