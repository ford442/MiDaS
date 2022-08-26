import torch
class BaseModel(torch.nn.Module):
    def load(self, path):
        parameters = torch.load(path,map_location={'cpu':'cuda:0'})
        if "optimizer" in parameters:
            parameters = parameters["model"]
        self.load_state_dict(parameters)
