import pickle as pkl
from datetime import datetime

class PickleSaver:
    def __init__(self, name = "savedata", path = "./"):
        self.name = name
        self.path = path
        self.date = str(datetime.now().year) + str(datetime.now().month) + str(datetime.now().day) + "_" + str(datetime.now().hour) + str(datetime.now().minute) + str(datetime.now().second)
        self.filename = self.path + self.name + "_" + self.date + ".pickle"
        self.info = {}
        self.info["Mean Return"] = []
        self.info["Net Params"] = {}
        self.info["Execution Time"] = 0
        self.info["Learning Rate"] = 0

    
    def save_data(self, key, data):
        self.info[key].append(data)
        with open(self.filename, 'wb+') as handle:
            pkl.dump(self.info, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def save_int(self, key, data):
        self.info[key] = (data)
        with open(self.filename, 'wb+') as handle:
            pkl.dump(self.info, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    def save_time_mark(self, time):
        self.info["Time Mark"] = time
        with open(self.filename, 'wb+') as handle:
            pkl.dump(self.info, handle, protocol=pkl.HIGHEST_PROTOCOL)
        