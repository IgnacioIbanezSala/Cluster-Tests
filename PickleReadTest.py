import pickle
import matplotlib.pyplot as plot


with open('Cartpole_Cross_Entropy_20241114_19464.pickle', 'rb') as handle:
    r_data = pickle.load(handle)


print(r_data)
print(len(r_data["Mean Return"]))

plot.plot(r_data["Mean Return"])
plot.show()