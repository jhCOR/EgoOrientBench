import json
from abc import ABC, abstractmethod

class ModelWrapper(ABC):

    @abstractmethod
    def load_model(self):
        raise NotImplementedError("[load_model] is not yet implemented")

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError("[predict] is not yet implemented")



    

