from base import BaseModel
from sklearn.pipeline import Pipeline


class Model(BaseModel):
    def __init__(self, pipeline, unions):
        steps = self.create_steps(pipeline, unions)
        self.model = Pipeline(steps=steps)

    def created_model(self):
        return self.model
