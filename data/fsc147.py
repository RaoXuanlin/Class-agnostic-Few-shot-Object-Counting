import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import os
import cv2
import scipy.ndimage
import json

imgs_mean = [0.485, 0.456, 0.406]
imgs_std = [0.229, 0.224, 0.225]

cv2.setNumThreads(0) # disable multithread to avoid deadlocks

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class FSC147Dataset(Dataset):
    """
    Dataset loader for FSC-147 Few-Shot Counting dataset.
    
    FSC-147 is specifically designed for few-shot object counting with:
    - Point annotations for object locations
    - 3 exemplar bounding boxes per image as references
    - Train/val/test splits pre-defined
    """

    def __init__(self, config, dataset_type='train', num_shots=3):
        """
        Args:
            config: Configuration object containing dataset paths
            dataset_type: 'train', 'val', or 'test'
            num_shots: Number of reference images to use (default 3 for FSC-147)
        """
        super(FSC147Dataset, self).__init__()
        self.config = config
        self.dataset_type = dataset_type
        self.num_shots = num_shots
        
        # Load annotations
        anno_file = os.path.join(config.data.data_path, 'annotation_FSC147_384.json')
        with open(anno_file) as f:
            self.annotations = json.load(f)
        
        # Load train/val/test splits
        split_file = os.path.join(config.data.data_path, 'Train_Test_Val_FSC_147.json')
        with open(split_file) as f:
            splits = json.load(f)
        
        # Get image list for current split
        if dataset_type == 'train':
            self.image_list = splits['train']
        elif dataset_type == 'val':
            self.image_list = splits['val']
        elif dataset_type == 'test':
            self.image_list = splits['test']
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'train', 'val', or 'test'")
        
        # Filter images that exist in annotations
        self.image_list = [img for img in self.image_list if img in self.annotations]
        
        # Image directory
        self.image_dir = os.path.join(config.data.data_path, 'images_384_VarV2')
        
        # Transforms
        self.query_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        self.reference_transform = transforms.ToTensor()
        
        self.length = len(self.image_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Generate query tensor
        query_tensor = self.generate_query_tensor(img)
        
        # Get annotation for this image
        anno = self.annotations[img_name]
        
        # Generate reference tensors from exemplar boxes
        references_tensor = self.generate_references_tensor(img, anno)
        
        # Generate density map from point annotations
        target_densemap_tensor = self.get_target_densemap(img, anno)
        
        return query_tensor, references_tensor, target_densemap_tensor

    def generate_query_tensor(self, img):
        """
        Generate query image tensor with padding to maintain aspect ratio.
        Resizes to 256x256 similar to COCO implementation.
        """
        img_size = img.shape
        max_img_size = max(img_size[0], img_size[1])
        ph = int((max_img_size - img_size[0]) / 2)
        pw = int((max_img_size - img_size[1]) / 2)
        
        pad_img = np.pad(img, ((ph, ph), (pw, pw), (0, 0)))
        pad_img = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB)
        query_tensor = self.query_transforms(pad_img)
        
        return query_tensor

    def generate_references_tensor(self, img, anno):
        """
        Generate reference image tensors from exemplar bounding boxes.
        
        Args:
            img: Original image (BGR format from cv2)
            anno: Annotation dictionary containing 'box_examples_coordinates'
        
        Returns:
            Tensor of shape (k, C, H, W) where k is number of shots
        """
        box_examples = anno['box_examples_coordinates']
        
        # FSC-147 provides 3 exemplar boxes, but we can sample k of them
        num_available = len(box_examples)
        
        if self.num_shots > num_available:
            # If requested more shots than available, sample with replacement
            selected_indices = np.random.choice(num_available, self.num_shots, replace=True)
        else:
            # Sample without replacement
            selected_indices = np.random.choice(num_available, self.num_shots, replace=False)
        
        references_tensor = []
        
        for idx in selected_indices:
            box = box_examples[idx]
            # FSC-147 box format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Ensure valid box coordinates
            x1, x2 = max(0, x1), min(img.shape[1], x2)
            y1, y2 = max(0, y1), min(img.shape[0], y2)
            
            # Extract crop
            crop_img = img[y1:y2, x1:x2]
            
            if crop_img.size == 0:
                # If crop is empty, use a small patch from center
                h, w = img.shape[:2]
                crop_img = img[h//2-32:h//2+32, w//2-32:w//2+32]
            
            # Resize crop to 64x64 (similar to COCO reference images)
            w = x2 - x1
            h = y2 - y1
            
            if w > 0 and h > 0:
                img_size = max(w, h)
                h_pad = int((img_size - h) // 2)
                w_pad = int((img_size - w) // 2)
                crop_img = np.pad(crop_img, ((h_pad, h_pad), (w_pad, w_pad), (0, 0)))
                crop_img = cv2.resize(crop_img, (64, 64))
            else:
                crop_img = cv2.resize(crop_img, (64, 64))
            
            # Convert to RGB
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            crop_tensor = self.reference_transform(crop_img)
            references_tensor.append(crop_tensor)
        
        reference_concat = torch.stack(references_tensor, 0)
        return reference_concat

    def get_target_densemap(self, img, anno):
        """
        Generate density map from point annotations.
        
        Args:
            img: Original image
            anno: Annotation dictionary containing 'points'
        
        Returns:
            Density map tensor of shape (1, 256, 256)
        """
        img_h, img_w = img.shape[:2]
        
        # Calculate padding and scaling for 256x256 output
        max_img_size = max(img_h, img_w)
        scale_ratio = 256 / max_img_size
        ph = (max_img_size - img_h) / 2
        ph = ph * scale_ratio
        pw = (max_img_size - img_w) / 2
        pw = pw * scale_ratio
        
        density = np.zeros((256, 256))
        
        # Get point annotations
        points = anno['points']
        
        ptx = []
        pty = []
        
        for point in points:
            x, y = point
            
            # Scale and translate point coordinates
            x = x * scale_ratio + pw
            y = y * scale_ratio + ph
            
            center_x = int(x)
            center_y = int(y)
            
            # Ensure points are within bounds
            if 0 <= center_x < 256 and 0 <= center_y < 256:
                ptx.append(center_x)
                pty.append(center_y)
        
        # Place points in density map
        if len(ptx) > 0:
            density[pty, ptx] = 1
        
        # Apply Gaussian filter to create smooth density map
        density = scipy.ndimage.gaussian_filter(density, sigma=(5, 5), mode='constant')
        density = torch.from_numpy(density.astype('float32'))
        
        return density.unsqueeze(0)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse
    import matplotlib.pyplot as plt
    import torchvision

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    config = argparse.Namespace()
    config.data = argparse.Namespace()
    config.data.data_path = '/path/to/FSC147'  # Update this path
    
    d = FSC147Dataset(config, 'train', num_shots=3)
    data_loader = DataLoader(d, 1, shuffle=True)

    print("Total:", len(data_loader))
    
    # Test loading a few samples
    it = iter(data_loader)
    for i in range(min(3, len(data_loader))):
        q, r, den = next(it)
        
        print(f"Sample {i}:")
        print(f"  Query shape: {q.shape}")
        print(f"  References shape: {r.shape}")
        print(f"  Density map shape: {den.shape}")
        print(f"  Object count: {torch.sum(den).item():.2f}")
