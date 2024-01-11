import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils_ import *

from PIL import Image
import torchvision.transforms as transforms
from UniformAugment import UniformAugment

## Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img_path']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(self.dataframe.iloc[idx][list(self.dataframe.columns)[2:]], dtype=torch.int8) if 'airplane' in self.dataframe.columns else 0
        
        return img, label


import random
def expand_image_by_duplicating_opposite_pixels(img, scale, randomflip):
    """
    Load an image using PIL and expand its resolution by duplicating opposite pixels.
    :param image_path: Path to the image file.
    :return: Expanded resolution image.
    """
    # Get original size
    width, height = img.size
    angles = [0, 90, 180, 270]
    # Create a new image with double the width and height
    new_img = Image.new('RGB', (width * scale, height * scale))
    # Copy original image into each quadrant of the new image
    if randomflip is False:
        for i in range(scale):
            for j in range(scale):
                new_img.paste(img, (width*i, width*j))
    else:
        for i in range(scale):
            for j in range(scale):
                img = img.rotate(random.choice(angles))
                new_img.paste(img, (width*i, width*j))
    return new_img
def crop_center(img, output_size):
    """
    Crop the center part of an image.
    :param image_path: Path to the image file.
    :param output_size: A tuple (width, height) for the size of the cropped area.
    :return: Cropped image.
    """
    # Load the image
    width, height = img.size
    # Calculate coordinates for the center crop
    left = (width - output_size[0]) / 2
    top = (height - output_size[1]) / 2
    right = (width + output_size[0]) / 2
    bottom = (height + output_size[1]) / 2
    # Crop the center of the image
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img

class RandomRotationWithPadding(transforms.RandomRotation):
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number): Pixel fill value for the area outside the rotated
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
    """
    def __init__(self, degrees, interpolation=transforms.InterpolationMode.NEAREST, expand=True, center=None, fill=0):
        super().__init__(degrees, interpolation, expand, center, fill)
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.
        Returns:
            PIL Image or Tensor: Rotated image.
        """
        original_width, original_height = img.size
        img = expand_image_by_duplicating_opposite_pixels(img, 3, True)
        img = super().forward(img)
        img = crop_center(img, (original_width, original_height))
        return img


def My_DataLoader(train_data, args, val_data=None, test_data=None, num_workers=1):
    
    train_transform = transforms.Compose([
        transforms.Resize(size=args.img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomRotationWithPadding(degrees=(0, 180)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_transform.transforms.insert(2, UniformAugment())
    
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)), 
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CustomDataset(train_data, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    if val_data is not None:
        val_dataset = CustomDataset(val_data, test_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    else:
        val_loader = None
    
    test_dataset = CustomDataset(test_data, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader