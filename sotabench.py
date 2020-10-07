import gc
import math

import torch
from torchbench.datasets.utils import download_file_from_google_drive
from torchbench.image_classification import ImageNet
from torchvision.transforms import transforms

import GENet

# GENet-large
file_id = '1xuyW2GB_kUfJNf2G146rk1sdKuYGxWlE'
destination = './GENet_params/'
filename = 'GENet_large.pth'
download_file_from_google_drive(file_id, destination, filename=filename)

input_image_size = 256
model = GENet.genet_large(pretrained=True, root='./GENet_params/')
model = GENet.fuse_bn(model)

input_image_crop = 0.875
resize_image_size = int(math.ceil(input_image_size / input_image_crop))
transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_list = [transforms.Resize(resize_image_size),
                  transforms.CenterCrop(input_image_size), transforms.ToTensor(), transforms_normalize]
transformer = transforms.Compose(transform_list)
# load model
model = model.cuda().half()
model.eval()

def send_data(input, target, device, dtype=torch.float16, non_blocking: bool = True):
    input = input.to(device=device, dtype=torch.float16, non_blocking=non_blocking)

    if target is not None:
        target = target.to(device=device, dtype=torch.float16, non_blocking=non_blocking)

    return input, target


print('Benchmarking GENet-large-pro')
# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='GENet-large-pro',
    paper_arxiv_id='2006.14090',
    input_transform=transformer,
    send_data_to_device=send_data,
    batch_size=256,
    num_workers=8,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.813},
    model_description="GENet-large-pro"
)

del model
gc.collect()
torch.cuda.empty_cache()

# GENet-normal
file_id = '1rpL0BKI_l5Xg4vN5fHGXPzTna5kW9hfs'
destination = './GENet_params/'
filename = 'GENet_normal.pth'
download_file_from_google_drive(file_id, destination, filename=filename)
input_image_size = 192
model = GENet.genet_normal(pretrained=True, root='./GENet_params/')
model = GENet.fuse_bn(model)

input_image_crop = 0.875
resize_image_size = int(math.ceil(input_image_size / input_image_crop))
transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_list = [transforms.Resize(resize_image_size),
                  transforms.CenterCrop(input_image_size), transforms.ToTensor(), transforms_normalize]
transformer = transforms.Compose(transform_list)
# load model
model = model.cuda().half()
model.eval()

print('Benchmarking GENet-normal')
# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='GENet-normal-pro',
    paper_arxiv_id='2006.14090',
    input_transform=transformer,
    send_data_to_device=send_data,
    batch_size=256,
    num_workers=8,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.800},
    model_description="GENet-normal-pro"
)

del model
gc.collect()
torch.cuda.empty_cache()

# GENet-light
file_id = '1jAkklQlQFPZi4odKUvbKEsNPYSS76GAv'
destination = './GENet_params/'
filename = 'GENet_small.pth'
download_file_from_google_drive(file_id, destination, filename=filename)
input_image_size = 192
model = GENet.genet_small(pretrained=True, root='./GENet_params/')
model = GENet.fuse_bn(model)

input_image_crop = 0.875
resize_image_size = int(math.ceil(input_image_size / input_image_crop))
transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_list = [transforms.Resize(resize_image_size),
                  transforms.CenterCrop(input_image_size), transforms.ToTensor(), transforms_normalize]
transformer = transforms.Compose(transform_list)
# load model
model = model.cuda().half()
model.eval()

print('Benchmarking GENet-light')
# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='GENet-light-pro',
    paper_arxiv_id='2006.14090',
    input_transform=transformer,
    send_data_to_device=send_data,
    batch_size=256,
    num_workers=8,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.757},
    model_description="GENet-light-pro"
)

del model
gc.collect()
torch.cuda.empty_cache()
