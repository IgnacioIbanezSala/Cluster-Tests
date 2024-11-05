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

    
    def save_data(self, key, data):
        self.info[key].append(data)
        with open(self.filename, 'wb+') as handle:
            pkl.dump(self.info, handle, protocol=pkl.HIGHEST_PROTOCOL)
    

        