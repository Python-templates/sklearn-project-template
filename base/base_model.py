from abc import abstractmethod
from models import methods_dict
from sklearn.pipeline import FeatureUnion

class BaseModel():

    @abstractmethod
    def created_model(self):
        '''should return created model'''
        return NotImplementedError

    def create_steps(self, pipeline, unions):
        steps = list()
        for model_name in pipeline:
            # add features from pipeline
            if model_name in methods_dict.keys():
                step = self._make_step(model_name)
                steps.append(step)

            # add combined features
            elif model_name in unions.keys():
                steps_cf = list()
                for model_name_cf in unions[model_name]:
                    if model_name_cf in methods_dict.keys():
                        step = self._make_step(model_name_cf)
                        steps_cf.append(step)
                if steps_cf:
                    steps.append([model_name, FeatureUnion(steps_cf)])

            else:
                # if method not found
                steps.append([model_name, None])
        return steps

    def change_step(self, model_name, model_instance, steps):
        for idx in range(len((steps))):
            if steps[idx][0] == model_name:
                steps[idx][1] = model_instance
                break
        return steps

    def _make_step(self, model_name):
        if isinstance(methods_dict[model_name], type):
            step = [model_name, methods_dict[model_name]()]
        else:
            # if already initialized
            step = [model_name, methods_dict[model_name]]
        return step



