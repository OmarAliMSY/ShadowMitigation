
from utills.processors import DataPreprocessor,\
    DataPostprocessor
from utills import FileHandler


class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def run(self):
        self.view.run(self.model)
