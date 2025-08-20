"""
Lab 6 Example 1: torch.utils.data.Dataset และ DataLoader พื้นฐาน
เรียนรู้การสร้างและใช้งาน Dataset และ DataLoader สำหรับจัดการข้อมูล
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler
import pandas as pd
from collections import Counter
import time

print("=" * 60)
print("Lab 6 Example 1: Dataset และ DataLoader พื้นฐาน")
print("=" * 60)

# 1. การใช้ TensorDataset (วิธีง่ายที่สุด)
def demo_tensor_dataset():
    """Demo การใช้ TensorDataset"""
    
    print("\n1. TensorDataset Demo")
    print("-" * 25)
    
    # สร้างข้อมูลตัวอย่าง
    X = torch.randn(1000, 10)  # 1000 samples, 10 features
    y = torch.randint(0, 3, (1000,))  # 3 classes
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {torch.unique(y).tolist()}")
    
    # สร้าง TensorDataset
    dataset = TensorDataset(X, y)
    print(f"Dataset length: {len(dataset)}")
    
    # ดูตัวอย่างข้อมูล
    sample_x, sample_y = dataset[0]
    print(f"First sample - X shape: {sample_x.shape}, y: {sample_y}")
    
    return dataset

# 2. การสร้าง Custom Dataset
class SimpleCustomDataset(Dataset):
    """Custom Dataset สำหรับข้อมูลง่ายๆ"""
    
    def __init__(self, num_samples=1000, num_features=20, num_classes=4, add_noise=True):
        """
        สร้างข้อมูลสังเคราะห์
        Args:
            num_samples: จำนวน samples
            num_features: จำนวน features
            num_classes: จำนวน classes
            add_noise: เพิ่ม noise หรือไม่
        """
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        
        # สร้างข้อมูล
        self.data, self.labels = self._generate_data(add_noise)
        
    def _generate_data(self, add_noise):
        """สร้างข้อมูลสังเคราะห์"""
        
        torch.manual_seed(42)  # เพื่อผลลัพธ์ที่เหมือนกัน
        
        # สร้างข้อมูลแต่ละ class
        data_list = []
        labels_list = []
        
        samples_per_class = self.num_samples // self.num_classes
        
        for class_id in range(self.num_classes):
            # สร้าง pattern ที่แตกต่างกันแต่ละ class
            if class_id == 0:
                # Class 0: ค่าเฉลี่ยสูงในครึ่งแรก
                class_data = torch.cat([
                    torch.randn(samples_per_class, self.num_features // 2) + 2,
                    torch.randn(samples_per_class, self.num_features // 2)
                ], dim=1)
            elif class_id == 1:
                # Class 1: ค่าเฉลี่ยสูงในครึ่งหลัง
                class_data = torch.cat([
                    torch.randn(samples_per_class, self.num_features // 2),
                    torch.randn(samples_per_class, self.num_features // 2) + 2
                ], dim=1)
            elif class_id == 2:
                # Class 2: pattern สลับกัน
                class_data = torch.randn(samples_per_class, self.num_features)
                class_data[:, ::2] += 1.5  # even indices
            else:
                # Class 3: ข้อมูลแบบ random
                class_data = torch.randn(samples_per_class, self.num_features) * 0.5
            
            data_list.append(class_data)
            labels_list.append(torch.full((samples_per_class,), class_id))
        
        # รวมข้อมูล
        data = torch.cat(data_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # เพิ่ม noise
        if add_noise:
            noise = torch.randn_like(data) * 0.1
            data += noise
        
        # สุ่มลำดับ
        perm = torch.randperm(len(data))
        data = data[perm]
        labels = labels[perm]
        
        return data, labels
    
    def __len__(self):
        """ความยาวของ dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """ดึงข้อมูล 1 sample"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_data = self.data[idx]
        sample_label = self.labels[idx]
        
        return sample_data, sample_label
    
    def get_class_distribution(self):
        """ดูการกระจายของ classes"""
        unique, counts = torch.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_statistics(self):
        """สถิติของข้อมูล"""
        return {
            'mean': torch.mean(self.data, dim=0),
            'std': torch.std(self.data, dim=0),
            'min': torch.min(self.data, dim=0)[0],
            'max': torch.max(self.data, dim=0)[0]
        }

# 3. DataLoader Configuration Demo
def demo_dataloader_configurations():
    """Demo การตั้งค่า DataLoader ต่างๆ"""
    
    print("\n3. DataLoader Configurations")
    print("-" * 35)
    
    # สร้าง dataset
    dataset = SimpleCustomDataset(num_samples=800, num_features=15)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # ทดสอบ DataLoader ต่างๆ
    configs = [
        {"batch_size": 32, "shuffle": True, "drop_last": False},
        {"batch_size": 64, "shuffle": True, "drop_last": True},
        {"batch_size": 16, "shuffle": False, "drop_last": False},
        {"batch_size": 128, "shuffle": True, "drop_last": False}
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        
        dataloader = DataLoader(dataset, **config)
        
        # ข้อมูลเกี่ยวกับ DataLoader
        print(f"  Number of batches: {len(dataloader)}")
        
        # ดูตัวอย่าง batch แรก
        first_batch = next(iter(dataloader))
        batch_x, batch_y = first_batch
        
        print(f"  Batch X shape: {batch_x.shape}")
        print(f"  Batch y shape: {batch_y.shape}")
        print(f"  Classes in first batch: {torch.unique(batch_y).tolist()}")

# 4. การแบ่งข้อมูล Train/Validation/Test
def demo_data_splitting():
    """Demo การแบ่งข้อมูล"""
    
    print("\n4. Data Splitting Demo")
    print("-" * 25)
    
    # สร้าง dataset
    dataset = SimpleCustomDataset(num_samples=1000)
    
    # วิธีที่ 1: ใช้ random_split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Original dataset: {len(dataset)}")
    print(f"Train set: {len(train_dataset)}")
    print(f"Validation set: {len(val_dataset)}")
    print(f"Test set: {len(test_dataset)}")
    
    # สร้าง DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

# 5. การใช้ WeightedRandomSampler
def demo_weighted_sampling():
    """Demo การ sampling แบบมี weight"""
    
    print("\n5. Weighted Random Sampling")
    print("-" * 30)
    
    # สร้าง imbalanced dataset
    X = torch.randn(1000, 10)
    
    # สร้าง imbalanced labels
    y = torch.cat([
        torch.zeros(700),  # Class 0: 70%
        torch.ones(200),   # Class 1: 20%
        torch.full((100,), 2)  # Class 2: 10%
    ]).long()
    
    # สุ่มลำดับ
    perm = torch.randperm(len(y))
    X, y = X[perm], y[perm]
    
    dataset = TensorDataset(X, y)
    
    # แสดงการกระจายเดิม
    class_counts = Counter(y.numpy())
    print(f"Original class distribution: {dict(class_counts)}")
    
    # คำนวณ weights สำหรับแต่ละ sample
    class_weights = {}
    total_samples = len(y)
    
    for class_id, count in class_counts.items():
        class_weights[class_id] = total_samples / (len(class_counts) * count)
    
    print(f"Class weights: {class_weights}")
    
    # สร้าง sample weights
    sample_weights = [class_weights[int(label)] for label in y]
    
    # สร้าง WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # สร้าง DataLoader ด้วย sampler
    balanced_loader = DataLoader(
        dataset, 
        batch_size=64, 
        sampler=sampler
    )
    
    # ทดสอบการกระจายใน batches
    print("\nTesting balanced sampling...")
    class_distribution_in_batches = Counter()
    
    for i, (batch_x, batch_y) in enumerate(balanced_loader):
        if i >= 10:  # ทดสอบ 10 batches แรก
            break
        
        batch_counts = Counter(batch_y.numpy())
        for class_id, count in batch_counts.items():
            class_distribution_in_batches[class_id] += count
    
    print(f"Distribution in first 10 batches: {dict(class_distribution_in_batches)}")

# 6. การใช้ SubsetRandomSampler
def demo_subset_sampling():
    """Demo การใช้ SubsetRandomSampler"""
    
    print("\n6. Subset Random Sampling")
    print("-" * 30)
    
    dataset = SimpleCustomDataset(num_samples=500)
    
    # สร้าง indices สำหรับแต่ละ subset
    all_indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(all_indices)
    
    # แบ่ง indices
    train_indices = all_indices[:350]
    val_indices = all_indices[350:450]
    test_indices = all_indices[450:]
    
    print(f"Train indices: {len(train_indices)}")
    print(f"Val indices: {len(val_indices)}")
    print(f"Test indices: {len(test_indices)}")
    
    # สร้าง samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # สร้าง DataLoaders
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # ทดสอบว่าไม่มีข้อมูลซ้ำกัน
    train_samples = set()
    for batch_idx, (data, target) in enumerate(train_loader):
        for i in range(len(data)):
            sample_hash = hash(tuple(data[i].numpy().flatten()))
            train_samples.add(sample_hash)
    
    print(f"Unique samples in train set: {len(train_samples)}")

# 7. Performance Comparison
def demo_performance_comparison():
    """เปรียบเทียบประสิทธิภาพของ DataLoader"""
    
    print("\n7. Performance Comparison")
    print("-" * 30)
    
    dataset = SimpleCustomDataset(num_samples=5000, num_features=50)
    
    # ทดสอบ num_workers ต่างๆ
    num_workers_list = [0, 1, 2, 4]
    batch_sizes = [32, 64, 128]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 20)
        
        for num_workers in num_workers_list:
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            # วัดเวลา
            start_time = time.time()
            
            for i, (data, target) in enumerate(dataloader):
                if i >= 50:  # ทดสอบ 50 batches
                    break
                # จำลองการประมวลผล
                _ = data.mean()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print(f"  num_workers={num_workers}: {elapsed_time:.3f} seconds")

# 8. DataLoader Iterator และ Advanced Usage
def demo_advanced_dataloader():
    """Demo การใช้ DataLoader แบบ advanced"""
    
    print("\n8. Advanced DataLoader Usage")
    print("-" * 35)
    
    dataset = SimpleCustomDataset(num_samples=200)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # การใช้ iterator
    print("Manual iteration:")
    dataloader_iter = iter(dataloader)
    
    # ดึง batch ที่ 1
    batch1_data, batch1_labels = next(dataloader_iter)
    print(f"Batch 1 shape: {batch1_data.shape}")
    
    # ดึง batch ที่ 2
    batch2_data, batch2_labels = next(dataloader_iter)
    print(f"Batch 2 shape: {batch2_data.shape}")
    
    # การใช้ enumerate
    print("\nUsing enumerate:")
    for batch_idx, (data, target) in enumerate(dataloader):
        print(f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
        if batch_idx >= 2:  # แสดงแค่ 3 batches แรก
            break
    
    # การ reset iterator
    print("\nResetting iterator:")
    dataloader_iter = iter(dataloader)
    first_batch_again = next(dataloader_iter)
    print(f"First batch again shape: {first_batch_again[0].shape}")

# ทดสอบการใช้งาน
print("เริ่มการทดสอบ Dataset และ DataLoader")

# 1. TensorDataset
dataset1 = demo_tensor_dataset()

# 2. Custom Dataset
print("\n2. Custom Dataset Demo")
print("-" * 25)
custom_dataset = SimpleCustomDataset(num_samples=500, num_features=12)
print(f"Custom dataset length: {len(custom_dataset)}")
print(f"Class distribution: {custom_dataset.get_class_distribution()}")

# ดูข้อมูลตัวอย่าง
sample_data, sample_label = custom_dataset[0]
print(f"Sample data shape: {sample_data.shape}")
print(f"Sample label: {sample_label}")

# สถิติของข้อมูล
stats = custom_dataset.get_statistics()
print(f"Data statistics:")
print(f"  Mean (first 5): {stats['mean'][:5]}")
print(f"  Std (first 5): {stats['std'][:5]}")

# 3. DataLoader Configurations
demo_dataloader_configurations()

# 4. Data Splitting
train_loader, val_loader, test_loader = demo_data_splitting()

# 5. Weighted Sampling
demo_weighted_sampling()

# 6. Subset Sampling
demo_subset_sampling()

# 7. Performance Comparison
demo_performance_comparison()

# 8. Advanced DataLoader
demo_advanced_dataloader()

print("\n" + "=" * 60)
print("สรุป Example 1: Dataset และ DataLoader พื้นฐาน")
print("=" * 60)
print("✓ TensorDataset สำหรับข้อมูลง่าย")
print("✓ Custom Dataset สำหรับความยืดหยุ่น")
print("✓ DataLoader configurations ต่างๆ")
print("✓ การแบ่งข้อมูล train/val/test")
print("✓ WeightedRandomSampler สำหรับ imbalanced data")
print("✓ SubsetRandomSampler สำหรับ custom splitting")
print("✓ Performance optimization")
print("✓ Advanced DataLoader usage patterns")
print("=" * 60)