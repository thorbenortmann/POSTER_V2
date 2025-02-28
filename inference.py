import os
import platform
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from models.PosterV2_7cls import pyramid_trans_expr2
from main import RecorderMeter, RecorderMeter1

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


def print_system_report(device):
    print(f'\n---System Info---')
    print(f'Operating System: {platform.system()} {platform.version()}')
    print(f'Python Version: {platform.python_version()}')
    print(f'PyTorch Version: {torch.__version__}')
    if device.type == 'cuda':
        print('\n---CUDA info---')
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'cuDNN Version: {torch.backends.cudnn.version()}')
        print(f'CUDA Available: {torch.cuda.is_available()}')
        print(f'GPU(s) available: {torch.cuda.device_count()}')
        print(f'Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('\n---CPU info---')
        print('Processor: ', platform.processor())
        print('CPU count: ', os.cpu_count())

    print('\n')


def load_model(model_path, device):
    model = pyramid_trans_expr2(img_size=224, num_classes=7)
    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if device.type == 'cpu' or torch.cuda.device_count() <= 1:
        new_state_dict = OrderedDict({k[7:] if k.startswith('module.') else k: v
                                      for k, v in checkpoint['state_dict'].items()})
        model.load_state_dict(new_state_dict)
    else:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    return model


def load_image(image_path, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = Variable(image.unsqueeze(0)).to(device)
    return image


def predict(model, image):
    with torch.no_grad():
        output = model(image)
    probabilities = F.softmax(output, dim=1)
    return probabilities.cpu().detach().numpy()


if __name__ == '__main__':
    model_path = 'path/to/checkpoint.pth'
    image_path = 'path/to/image.jpg'
    number_of_inferences = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_system_report(device)

    model = load_model(model_path, device)
    image = load_image(image_path, device)

    print(f'\nStarting performance test for {number_of_inferences} inferences...')
    start_time = time.time()
    for j in range(number_of_inferences):
        prediction = predict(model, image)
    end_time = time.time()
    print(f'Processed {number_of_inferences} images in  {(end_time - start_time):.3f} seconds')
    print(f'One inference takes about  {(end_time - start_time) / number_of_inferences * 1000:.2f} milli seconds')
