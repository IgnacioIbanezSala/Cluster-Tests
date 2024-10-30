import pickle
import matplotlib.pyplot as plot


with open('Pickle Files/PPO_data.pickle', 'rb') as handle:
    r_data = pickle.load(handle)


plot.plot(r_data["Mean Return"])
plot.show()
print(r_data)