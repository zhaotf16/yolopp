import os

class StarData():
    def __init__(self):
        self.content = []

def read_star(path):
    with open(path) as f:
        while True:
            line = f.readline()

def read_all_star(path):
    if not os.path.isdir(path):
        print(path, " is not a valid directory")
        return
    if not path.endswith('/'):
        path += '/'
        mrc_data = []
    for file in os.listdir(path):
        if file.endswith('.star'):
            print("Loading %s ..." % (file))
            content = read_star(file)
