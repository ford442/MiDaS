import torch
class BaseModel(torch.nn.Module):
    def load(self, path):
        device=torch.device('cuda:0')
        parameters = torch.load(path)
        if "optimizer" in parameters:
            parameters = parameters["model"]
        self.load_state_dict(parameters)
