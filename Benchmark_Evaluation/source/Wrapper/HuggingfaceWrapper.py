from ModelWrapper import ModelWrapper
from transformers import AutoModel, AutoTokenizer

class HuggingfaceModelWrapper(ModelWrapper):
    def __init__(self, model_name, configuration, **kwargs):
        pass

    def load_model(self):
        pass

    def predict(self, x):
        pass