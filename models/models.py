from base import BaseModel
from sklearn.pipeline import Pipeline

class Model(BaseModel):
    def __init__(self, pipeline):
        steps = self.create_steps(pipeline)
        self.model = Pipeline(steps=steps)

    def get_model(self):
        return self.model

