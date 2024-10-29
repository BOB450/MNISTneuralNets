import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. PyTorch will use the GPU.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print("cuDNN Version:", torch.backends.cudnn.version())
    print("Is cuDNN enabled:", torch.backends.cudnn.enabled)

    # Perform a simple operation on the GPU
    x = torch.rand(3, 3).to("cuda")
    y = torch.rand(3, 3).to("cuda")
    z = x + y
    print("Tensor operation result on GPU:", z)
else:
    print("CUDA is not available. PyTorch will use the CPU.")
