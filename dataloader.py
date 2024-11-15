import torch 

__all__ = [
    
    'ImageDataset'
]


class ImageDataset(torch.utils.data.Dataset):
     
    def __init__(self, data, transform=None): 
        self.data = data
        self.transform = transform 
        print(f'Transforming the Images with the Transformations:\n{transform}')
  
    # Defining the length of the dataset 
    def __len__(self): 
        return len(self.data) 
  
    # Defining the method to get an item from the dataset 
    def __getitem__(self, index):
        image, label = self.data[index]
        # Applying the transform 
        if self.transform: 
            image = self.transform(image) 
          
        return image, label