from sys import version_info

import mlflow.pyfunc

class ModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import pickle
        self.model = pickle.load(open(context.artifacts["model_path"], 'rb'))
    
    def predict(self, data):
        return self.model.predict(data)