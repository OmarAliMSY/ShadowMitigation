
from abc import ABC, abstractmethod

class Processor(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def process(self, data):
        pass
