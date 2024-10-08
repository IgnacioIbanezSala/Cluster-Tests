import pickle

with open('Mean_returns.pickle', 'rb') as handle:
    r_data = pickle.load(handle)

print(r_data)