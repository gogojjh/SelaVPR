dependencies = ['torch']

import torch
import network

def trained_model(domain):
    model = network.GeoLocalizationNet()
    model = torch.nn.DataParallel(model)
    if domain == "pitts30k":
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                f'https://github.com/gogojjh/SelaVPR/releases/download/v1.0.0/SelaVPR_{domain}.pth', 
                map_location=torch.device('cpu'))["model_state_dict"]
        )
    elif domain == "msls":
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                f'https://github.com/gogojjh/SelaVPR/releases/download/v1.0.0/SelaVPR_{domain}.pth',
                map_location=torch.device('cpu'))["model_state_dict"]
        )
    else:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                f'https://github.com/gogojjh/SelaVPR/releases/download/v1.0.0/SelaVPR_{domain}.pth', 
                map_location=torch.device('cpu'))["model_state_dict"]
        )

    return model