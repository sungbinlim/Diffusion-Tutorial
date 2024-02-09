import os
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image

class ImageFolder(data.Dataset):
    """Custom Dataset compatible with prebuilt DataLoader."""
    def __init__(self, root, transform=None):
        """Initialize image paths and preprocessing module."""
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform
    
    def __len__(self):
        """Return the total number of image files."""
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """Read an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image    

def get_loader(image_path, image_size, batch_size, num_workers=2):
    """Create and return Dataloader."""
    
    transform = basic_transform(image_size)
    
    dataset = ImageFolder(image_path, transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader

def basic_transform(image_size):
    transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t * 2) - 1),
                    ])
    return transform

def reverse_transform():
    transform = transforms.Compose([
     transforms.Lambda(lambda t: (t + 1) / 2),
     transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     transforms.Lambda(lambda t: t * 255.),
     transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
     transforms.ToPILImage(),
    ])
    return transform