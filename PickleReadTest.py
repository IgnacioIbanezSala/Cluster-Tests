import pickle
import matplotlib.pyplot as plot


with open('ScriptsRL\Mean_returns_Reinforce.pickle', 'rb') as handle:
    r_data = pickle.load(handle)


plot.plot(r_data["Mean Return"])
plot.show()
print(r_data)