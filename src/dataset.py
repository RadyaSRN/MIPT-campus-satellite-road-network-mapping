from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepGlobeDataset(Dataset):
  """DeepGlobe road extraction dataset"""
  def __init__(self, images_list, masks_list, transform=None):
    self.images_list = images_list
    self.masks_list = masks_list
    self.transform = transform

  def __len__(self):
    return len(self.images_list)

  def __getitem__(self, idx):
    image = self.images_list[idx]
    mask = self.masks_list[idx]

    if self.transform:
      transformed = self.transform(image=image, mask=mask)
      return transformed['image'], transformed['mask']

    return image, mask