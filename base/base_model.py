from abc import abstractmethod
from models import methods_dict

class BaseModel():

    @abstractmethod
    def created_model(self):
        '''should return created model'''
        return NotImplementedError

    def create_steps(self, pipeline):
        steps = list()
        for model_name in pipeline:
            try:
                if isinstance(methods_dict[model_name], type):
                    steps.append((model_name, methods_dict[model_name]()))
                else:
                    # if already initialized
                    steps.append((model_name, methods_dict[model_name]))
            except:
                steps.append([model_name, None])
                # print(f"Key: {mod} not defined in methods_dict.")
        return steps

    def change_step(self, model_name, model_instance, steps):
        for idx in range(len((steps))):
            if steps[idx][0] == model_name:
                steps[idx][1] = model_instance
                break
        return steps
