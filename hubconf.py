dependencies = ['torch']

import torch
import network
import argparse

def trained_model(domain):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    args.registers = False
    args.foundation_model_path = None
    args.features_dim = 1024
    if domain == "pitts30k":
        args.resume = 'https://github.com/gogojjh/SelaVPR/releases/download/v2.0.0/SelaVPR_pitts30k.pth'
    elif domain == "msls":
        args.resume = 'https://github.com/gogojjh/SelaVPR/releases/download/v2.0.0/SelaVPR_msls.pth'
    elif domain == "msls":
        args.resume = 'https://github.com/gogojjh/SelaVPR/releases/download/v2.0.0/SelaVPR_msls.pth'

    # Load model
    model = network.GeoLocalizationNet(args)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            args.resume, 
            map_location=torch.device('cpu'))["model_state_dict"]
        )
    return model

if __name__ == "__main__":
    model = trained_model("pitts30k")
    print(model)