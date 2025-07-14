import torch
import kagglehub

path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")

print("Path to dataset files:", path)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.xpu.is_available():
        return torch.device('xpu')
    return torch.device('cpu')

def main():
    pass

if __name__ == "__main__":
    main()