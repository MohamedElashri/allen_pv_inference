import torch
import sys

class Model(torch.nn.Module):
    def __init__(self, nOut1=20, nOut2=20, nOut3=20, nOut4=20, nOut5=20):
        super(Model, self).__init__()
        self.layer1 = torch.nn.Linear(9, nOut1)
        self.layer2 = torch.nn.Linear(nOut1, nOut2)
        self.layer3 = torch.nn.Linear(nOut2, nOut3)
        self.layer4 = torch.nn.Linear(nOut3, nOut4)
        self.layer5 = torch.nn.Linear(nOut4, nOut5)
        self.layer6 = torch.nn.Linear(nOut5, 100)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.layer1(x), 0.01)
        x = torch.nn.functional.leaky_relu(self.layer2(x), 0.01)
        x = torch.nn.functional.leaky_relu(self.layer3(x), 0.01)
        x = torch.nn.functional.leaky_relu(self.layer4(x), 0.01)
        x = torch.nn.functional.leaky_relu(self.layer5(x), 0.01)
        x = torch.nn.functional.softplus(self.layer6(x))
        return x

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_model.py <input_pyt_file> <output_pt_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Load the state dict
    state_dict = torch.load(input_file, map_location=torch.device('cpu'))

    # Create a new model and load the state dict
    model = Model()
    model.load_state_dict(state_dict)

    # Convert to TorchScript
    script_module = torch.jit.script(model)

    # Save the TorchScript model
    torch.jit.save(script_module, output_file)

    print(f"Converted model saved to {output_file}")
