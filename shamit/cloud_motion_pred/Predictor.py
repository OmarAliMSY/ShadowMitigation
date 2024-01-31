from abc import ABC, abstractmethod

class Predictor(ABC):
    
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self.iamge = None
        self.tensor = None
        self.pred_frame = None
        


    @abstractmethod
    def inference(self):
        print("Moin")
