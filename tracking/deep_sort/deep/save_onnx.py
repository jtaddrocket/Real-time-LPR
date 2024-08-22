import torch
import torch.onnx
from model import Net  # Make sure this import works correctly

# Load the checkpoint
checkpoint = torch.load("./checkpoint/ckpt.t7", map_location=torch.device('cpu'))
net_dict = checkpoint['net_dict']

# Create an instance of your model
model = Net(reid=True)

# Load the state dict into your model
model.load_state_dict(net_dict, strict=False)

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor
# Adjust the shape according to your model's expected input
dummy_input = torch.randn(1, 3, 128, 64)

# Export the model to ONNX format
torch.onnx.export(model,               # model being run
                  dummy_input,         # model input (or a tuple for multiple inputs)
                  "reid_model.onnx",   # where to save the model
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

print("Model has been converted to ONNX")