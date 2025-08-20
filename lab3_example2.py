"""
Lab 6 Example 2: Image Preprocessing และ Augmentation
เรียนรู้การประมวลผลภาพและเพิ่มความหลากหลายของข้อมูลภาพ
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import os
import random
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

print("=" * 60)
print("Lab 6 Example 2: Image Preprocessing และ Augmentation")
print("=" * 60)

# 1. สร้างข้อมูลภาพตัวอย่าง
def create_sample_images(num_images=100, save_dir="sample_images"):
    """สร้างภาพตัวอย่างสำหรับทดสอบ"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Creating {num_images} sample images...")
    
    # สร้างโฟลเดอร์สำหรับแต่ละ class
    classes = ['circles', 'squares', 'triangles']
    for class_name in classes:
        class_dir = os.path.join(save_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    images_per_class = num_images // len(classes)
    
    for class_idx, class_name in enumerate(classes):
        for i in range(images_per_class):
            # สร้างภาพ 64x64 pixels
            img_array = np.zeros((64, 64, 3), dtype=np.uint8)
            
            if class_name == 'circles':
                # วาดวงกลม
                center = (32, 32)
                radius = random.randint(15, 25)
                y, x = np.ogrid[:64, :64]
                mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
                color = [random.randint(100, 255) for _ in range(3)]
                img_array[mask] = color
                
            elif class_name == 'squares':
                # วาดสี่เหลี่ยม
                size = random.randint(20, 30)
                start_x = random.randint(10, 64 - size - 10)
                start_y = random.randint(10, 64 - size - 10)
                color = [random.randint(100, 255) for _ in range(3)]
                img_array[start_y:start_y+size, start_x:start_x+size] = color
                
            else:  # triangles
                # วาดสามเหลี่ยม (แบบง่าย)
                center_x, center_y = 32, 32
                size = random.randint(15, 25)
                color = [random.randint(100, 255) for _ in range(3)]
                
                # สร้างสามเหลี่ยมแบบง่าย
                for y in range(center_y - size, center_y + size):
                    for x in range(center_x - size, center_x + size):
                        if (0 <= y < 64 and 0 <= x < 64 and 
                            abs(x - center_x) <= (size - abs(y - center_y))):
                            img_array[y, x] = color
            
            # บันทึกภาพ
            img = Image.fromarray(img_array)
            img_path = os.path.join(save_dir, class_name, f"{class_name}_{i:03d}.png")
            img.save(img_path)
    
    print(f"Sample images created in {save_dir}/")
    return save_dir

# 2. Basic Image Transforms
def demo_basic_transforms():
    """Demo การใช้ basic transforms"""
    
    print("\n1. Basic Image Transforms")
    print("-" * 30)
    
    # สร้างภาพตัวอย่าง
    sample_image = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
    
    # แปลงเป็น PIL Image เพื่อแสดงผล
    to_pil = transforms.ToPILImage()
    sample_pil = to_pil(sample_image)
    
    # Basic transforms
    basic_transforms = {
        'Original': transforms.Compose([]),
        'Resize': transforms.Resize((32, 32)),
        'CenterCrop': transforms.CenterCrop(48),
        'RandomCrop': transforms.RandomCrop(48),
        'RandomHorizontalFlip': transforms.RandomHorizontalFlip(p=1.0),
        'RandomVerticalFlip': transforms.RandomVerticalFlip(p=1.0),
        'RandomRotation': transforms.RandomRotation(45),
        'ColorJitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
    }
    
    # แสดงผลการ transform
    print("Transform effects (เปลี่ยนแปลงภาพ):")
    for name, transform in basic_transforms.items():
        try:
            if name == 'Original':
                transformed = sample_pil
            else:
                transformed = transform(sample_pil)
            print(f"  {name}: ใช้งานได้")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    return basic_transforms

# 3. Advanced Image Augmentation
class AdvancedImageAugmentation:
    """Class สำหรับ advanced image augmentation"""
    
    def __init__(self):
        self.transforms = {
            'geometric': self._get_geometric_transforms(),
            'color': self._get_color_transforms(),
            'noise': self._get_noise_transforms(),
            'blur': self._get_blur_transforms()
        }
    
    def _get_geometric_transforms(self):
        """Geometric transformations"""
        return {
            'RandomRotation': transforms.RandomRotation(degrees=30),
            'RandomAffine': transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                shear=10
            ),
            'RandomPerspective': transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            'RandomResizedCrop': transforms.RandomResizedCrop(
                size=(64, 64),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33)
            )
        }
    
    def _get_color_transforms(self):
        """Color transformations"""
        return {
            'ColorJitter': transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            'RandomGrayscale': transforms.RandomGrayscale(p=0.2),
            'RandomAutocontrast': transforms.RandomAutocontrast(p=0.5),
            'RandomEqualize': transforms.RandomEqualize(p=0.5)
        }
    
    def _get_noise_transforms(self):
        """Noise transformations"""
        return {
            'GaussianNoise': self._add_gaussian_noise,
            'SaltPepperNoise': self._add_salt_pepper_noise,
            'SpeckleNoise': self._add_speckle_noise
        }
    
    def _get_blur_transforms(self):
        """Blur transformations"""
        return {
            'GaussianBlur': transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            'MotionBlur': self._motion_blur,
            'RandomBlur': self._random_blur
        }
    
    def _add_gaussian_noise(self, img):
        """เพิ่ม Gaussian noise"""
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = np.array(img)
        
        noise = np.random.normal(0, 25, img_array.shape).astype(np.int16)
        noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)
    
    def _add_salt_pepper_noise(self, img):
        """เพิ่ม Salt and Pepper noise"""
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = np.array(img)
        
        noise_ratio = 0.05
        h, w = img_array.shape[:2]
        
        # Salt noise (white pixels)
        salt_coords = [np.random.randint(0, i - 1, int(noise_ratio * h * w / 2)) 
                      for i in img_array.shape[:2]]
        img_array[salt_coords[0], salt_coords[1]] = 255
        
        # Pepper noise (black pixels)
        pepper_coords = [np.random.randint(0, i - 1, int(noise_ratio * h * w / 2)) 
                        for i in img_array.shape[:2]]
        img_array[pepper_coords[0], pepper_coords[1]] = 0
        
        return Image.fromarray(img_array)
    
    def _add_speckle_noise(self, img):
        """เพิ่ม Speckle noise"""
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = np.array(img)
        
        noise = np.random.randn(*img_array.shape) * 0.1
        speckle = img_array + img_array * noise
        speckle = np.clip(speckle, 0, 255).astype(np.uint8)
        
        return Image.fromarray(speckle)
    
    def _motion_blur(self, img):
        """Motion blur effect"""
        if isinstance(img, Image.Image):
            # สร้าง motion blur kernel
            kernel_size = random.randint(3, 7)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = 1 / kernel_size
            
            img_array = np.array(img)
            from scipy import ndimage
            
            if len(img_array.shape) == 3:
                blurred = np.zeros_like(img_array)
                for i in range(3):
                    blurred[:, :, i] = ndimage.convolve(img_array[:, :, i], kernel)
            else:
                blurred = ndimage.convolve(img_array, kernel)
            
            return Image.fromarray(blurred.astype(np.uint8))
        return img
    
    def _random_blur(self, img):
        """Random blur effect"""
        if random.random() > 0.5:
            radius = random.uniform(0.5, 2.0)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

# 4. Custom Image Dataset
class CustomImageDataset(Dataset):
    """Custom dataset สำหรับภาพ"""
    
    def __init__(self, image_dir, transform=None, target_transform=None):
        """
        Args:
            image_dir: โฟลเดอร์ที่มีโฟลเดอร์ย่อยแต่ละ class
            transform: การประมวลผลภาพ
            target_transform: การประมวลผล label
        """
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # หา classes และสร้าง mapping
        self.classes = sorted([d for d in os.listdir(image_dir) 
                              if os.path.isdir(os.path.join(image_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # สร้างรายการไฟล์ภาพ
        self.samples = self._make_dataset()
        
    def _make_dataset(self):
        """สร้างรายการ (image_path, class_index)"""
        samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.image_dir, class_name)
            class_index = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, filename)
                    samples.append((path, class_index))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, target = self.samples[idx]
        
        # โหลดภาพ
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target
    
    def get_class_distribution(self):
        """ดูการกระจายของ classes"""
        targets = [sample[1] for sample in self.samples]
        from collections import Counter
        return Counter(targets)

# 5. Transform Compositions สำหรับ Training และ Validation
def get_transforms(image_size=64, augment=True):
    """สร้าง transform compositions"""
    
    if augment:
        # Training transforms (มี augmentation)
        train_transforms = transforms.Compose([
            transforms.Resize((image_size + 8, image_size + 8)),  # ขยายนิดหน่อย
            transforms.RandomCrop(image_size),                     # Crop แบบสุ่ม
            transforms.RandomHorizontalFlip(p=0.5),               # Flip 50%
            transforms.RandomRotation(degrees=15),                # หมุน ±15 องศา
            transforms.ColorJitter(                               # เปลี่ยนสี
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),                    # Grayscale 10%
            transforms.ToTensor(),                                # แปลงเป็น tensor
            transforms.Normalize(                                 # Normalize
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Validation/Test transforms (ไม่มี augmentation)
        train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # Validation transforms (เหมือนกันเสมอ)
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transforms, val_transforms

# 6. Visualization Functions
def visualize_transforms(dataset, indices=[0, 1, 2, 3], num_augmentations=4):
    """แสดงผลการ transform"""
    
    print("\n6. Transform Visualization")
    print("-" * 30)
    
    # สร้าง transform สำหรับ augmentation
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])
    
    # ไม่ใช้ transform
    no_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    print(f"Showing {len(indices)} original images with {num_augmentations} augmentations each")
    
    for idx in indices:
        original_image, label = dataset.samples[idx]
        original_pil = Image.open(original_image).convert('RGB')
        
        print(f"\nImage {idx}: {os.path.basename(original_image)}, Class: {dataset.classes[label]}")
        
        # Original
        original_tensor = no_transform(original_pil)
        print(f"  Original shape: {original_tensor.shape}")
        
        # Augmentations
        augmented_images = []
        for i in range(num_augmentations):
            aug_tensor = augment_transform(original_pil)
            augmented_images.append(aug_tensor)
            print(f"  Augmentation {i+1} shape: {aug_tensor.shape}")

def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """แปลง normalized tensor กลับเป็นภาพปกติ"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    denormalized = tensor * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized

# 7. Batch Visualization
def visualize_batch(dataloader, num_images=8, denormalize=True):
    """แสดงผล batch ของภาพ"""
    
    print("\n7. Batch Visualization")
    print("-" * 25)
    
    # ดึง batch แรก
    batch_images, batch_labels = next(iter(dataloader))
    
    print(f"Batch shape: {batch_images.shape}")
    print(f"Labels shape: {batch_labels.shape}")
    print(f"Unique labels in batch: {torch.unique(batch_labels).tolist()}")
    
    # เลือกภาพที่จะแสดง
    num_images = min(num_images, len(batch_images))
    selected_images = batch_images[:num_images]
    selected_labels = batch_labels[:num_images]
    
    if denormalize:
        selected_images = denormalize_tensor(selected_images)
    
    print(f"Showing first {num_images} images from batch")
    print(f"Image tensor range: [{selected_images.min():.3f}, {selected_images.max():.3f}]")
    
    return selected_images, selected_labels

# 8. การทดสอบ Performance ของ Augmentation
def benchmark_augmentation_performance():
    """ทดสอบประสิทธิภาพของ augmentation"""
    
    print("\n8. Augmentation Performance Benchmark")
    print("-" * 40)
    
    # สร้างภาพตัวอย่าง
    sample_image = Image.new('RGB', (224, 224), color='red')
    
    # Transform sets ที่แตกต่างกัน
    transform_sets = {
        'Basic': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ]),
        'Light Augmentation': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ]),
        'Heavy Augmentation': transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    # ทดสอบเวลา
    num_iterations = 100
    
    for name, transform in transform_sets.items():
        import time
        start_time = time.time()
        
        for _ in range(num_iterations):
            _ = transform(sample_image)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations * 1000  # milliseconds
        
        print(f"{name:20}: {avg_time:.2f} ms per transform")

# ทดสอบการใช้งาน
print("เริ่มการทดสอบ Image Preprocessing และ Augmentation")

# 1. สร้างภาพตัวอย่าง
sample_dir = create_sample_images(num_images=60)

# 2. Basic Transforms
basic_transforms = demo_basic_transforms()

# 3. Advanced Augmentation
print("\n3. Advanced Augmentation Demo")
print("-" * 35)
augmentor = AdvancedImageAugmentation()

print("Available augmentation categories:")
for category, transforms_dict in augmentor.transforms.items():
    print(f"  {category}: {list(transforms_dict.keys())}")

# 4. Custom Dataset
print("\n4. Custom Image Dataset")
print("-" * 25)

# สร้าง transforms
train_transform, val_transform = get_transforms(image_size=64, augment=True)

# สร้าง datasets
train_dataset = CustomImageDataset(sample_dir, transform=train_transform)
val_dataset = CustomImageDataset(sample_dir, transform=val_transform)

print(f"Train dataset: {len(train_dataset)} images")
print(f"Val dataset: {len(val_dataset)} images")
print(f"Classes: {train_dataset.classes}")
print(f"Class distribution: {train_dataset.get_class_distribution()}")

# ทดสอบการโหลดข้อมูล
sample_image, sample_label = train_dataset[0]
print(f"Sample image shape: {sample_image.shape}")
print(f"Sample label: {sample_label} ({train_dataset.classes[sample_label]})")

# 5. DataLoaders
print("\n5. Creating DataLoaders")
print("-" * 25)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,  # Set to 0 untuk avoid multiprocessing issues
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print(f"Train loader: {len(train_loader)} batches")
print(f"Val loader: {len(val_loader)} batches")

# 6. Transform Visualization
visualize_transforms(train_dataset, indices=[0, 1, 2], num_augmentations=3)

# 7. Batch Visualization
selected_images, selected_labels = visualize_batch(train_loader, num_images=6)

# 8. เปรียบเทียบ Augmented vs Non-augmented
print("\n8. Augmented vs Non-augmented Comparison")
print("-" * 45)

# สร้าง dataset ไม่มี augmentation
no_aug_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

no_aug_dataset = CustomImageDataset(sample_dir, transform=no_aug_transform)
no_aug_loader = DataLoader(no_aug_dataset, batch_size=16, shuffle=False)

# เปรียบเทียบ
aug_batch, aug_labels = next(iter(train_loader))
no_aug_batch, no_aug_labels = next(iter(no_aug_loader))

print(f"Augmented batch stats:")
print(f"  Shape: {aug_batch.shape}")
print(f"  Range: [{aug_batch.min():.3f}, {aug_batch.max():.3f}]")
print(f"  Mean: {aug_batch.mean():.3f}")

print(f"Non-augmented batch stats:")
print(f"  Shape: {no_aug_batch.shape}")
print(f"  Range: [{no_aug_batch.min():.3f}, {no_aug_batch.max():.3f}]")
print(f"  Mean: {no_aug_batch.mean():.3f}")

# 9. Performance Benchmark
benchmark_augmentation_performance()

# 10. Memory Usage Analysis
print("\n10. Memory Usage Analysis")
print("-" * 30)

# วิเคราะห์การใช้หน่วยความจำ
def analyze_memory_usage(dataloader, num_batches=5):
    """วิเคราะห์การใช้หน่วยความจำ"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    batches = []
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
        batches.append((images, labels))
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = memory_after - memory_before
    
    return memory_used, len(batches)

# วิเคราะห์หน่วยความจำ
memory_used, num_batches = analyze_memory_usage(train_loader, num_batches=3)
print(f"Memory used for {num_batches} batches: {memory_used:.2f} MB")
print(f"Average memory per batch: {memory_used/num_batches:.2f} MB")

print("\n" + "=" * 60)
print("สรุป Example 2: Image Preprocessing และ Augmentation")
print("=" * 60)
print("✓ การสร้างภาพตัวอย่างสำหรับทดสอบ")
print("✓ Basic image transforms (resize, crop, flip, rotation)")
print("✓ Advanced augmentation techniques (noise, blur, color)")
print("✓ Custom Image Dataset implementation")
print("✓ Transform compositions สำหรับ train/val")
print("✓ Visualization tools สำหรับ transforms และ batches")
print("✓ Performance benchmarking")
print("✓ Memory usage analysis")
print("=" * 60)