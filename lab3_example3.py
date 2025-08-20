"""
Lab 6 Example 3: Batch Processing และ Shuffling
เรียนรู้การจัดการ batch และ shuffling แบบมีประสิทธิภาพ
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import time
import random

print("=" * 60)
print("Lab 6 Example 3: Batch Processing และ Shuffling")
print("=" * 60)

# 1. Custom Dataset สำหรับทดสอบ Batch Processing
class BatchTestDataset(Dataset):
    """Dataset สำหรับทดสอบ batch processing"""
    
    def __init__(self, num_samples=1000, feature_dim=50, num_classes=5, 
                 add_sample_weight=False, imbalanced=False):
        """
        Args:
            num_samples: จำนวน samples
            feature_dim: จำนวน features
            num_classes: จำนวน classes
            add_sample_weight: เพิ่ม sample weight หรือไม่
            imbalanced: สร้าง imbalanced dataset หรือไม่
        """
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # สร้างข้อมูล
        torch.manual_seed(42)
        self.data = torch.randn(num_samples, feature_dim)
        
        if imbalanced:
            # สร้าง imbalanced labels
            self.labels = self._create_imbalanced_labels()
        else:
            # สร้าง balanced labels
            self.labels = torch.randint(0, num_classes, (num_samples,))
        
        # Sample weights
        if add_sample_weight:
            self.sample_weights = self._calculate_sample_weights()
        else:
            self.sample_weights = None
        
        # เพิ่ม metadata
        self.sample_ids = torch.arange(num_samples)
        
    def _create_imbalanced_labels(self):
        """สร้าง imbalanced labels"""
        # Class distribution: [50%, 25%, 15%, 7%, 3%]
        proportions = [0.5, 0.25, 0.15, 0.07, 0.03]
        labels = []
        
        for class_id, prop in enumerate(proportions):
            num_samples_class = int(self.num_samples * prop)
            labels.extend([class_id] * num_samples_class)
        
        # เติมที่เหลือด้วย class สุดท้าย
        while len(labels) < self.num_samples:
            labels.append(self.num_classes - 1)
        
        # สุ่มลำดับ
        random.shuffle(labels)
        
        return torch.tensor(labels[:self.num_samples])
    
    def _calculate_sample_weights(self):
        """คำนวณ sample weights สำหรับ balanced sampling"""
        class_counts = Counter(self.labels.numpy())
        total_samples = len(self.labels)
        
        # คำนวณ weight แต่ละ class
        class_weights = {}
        for class_id in range(self.num_classes):
            if class_id in class_counts:
                class_weights[class_id] = total_samples / (self.num_classes * class_counts[class_id])
            else:
                class_weights[class_id] = 1.0
        
        # กำหนด weight แต่ละ sample
        sample_weights = torch.zeros(self.num_samples)
        for i, label in enumerate(self.labels):
            sample_weights[i] = class_weights[label.item()]
        
        return sample_weights
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx],
            'sample_id': self.sample_ids[idx]
        }
        
        if self.sample_weights is not None:
            sample['weight'] = self.sample_weights[idx]
        
        return sample
    
    def get_class_distribution(self):
        """ดูการกระจายของ classes"""
        return Counter(self.labels.numpy())

# 2. Custom Collate Functions
def custom_collate_fn(batch):
    """Custom collate function สำหรับ batch processing"""
    
    # แยกข้อมูลแต่ละส่วน
    data = torch.stack([sample['data'] for sample in batch])
    labels = torch.stack([sample['label'] for sample in batch])
    sample_ids = torch.stack([sample['sample_id'] for sample in batch])
    
    # ตรวจสอบว่ามี weights หรือไม่
    if 'weight' in batch[0]:
        weights = torch.stack([sample['weight'] for sample in batch])
        return {
            'data': data,
            'labels': labels,
            'sample_ids': sample_ids,
            'weights': weights,
            'batch_size': len(batch)
        }
    else:
        return {
            'data': data,
            'labels': labels,
            'sample_ids': sample_ids,
            'batch_size': len(batch)
        }

def variable_length_collate_fn(batch):
    """Collate function สำหรับข้อมูลที่มีความยาวต่างกัน"""
    
    # เรียงตามความยาวจากมากไปน้อย
    batch = sorted(batch, key=lambda x: x['data'].shape[0], reverse=True)
    
    # Padding
    max_length = batch[0]['data'].shape[0]
    feature_dim = batch[0]['data'].shape[1]
    
    padded_data = torch.zeros(len(batch), max_length, feature_dim)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    
    for i, sample in enumerate(batch):
        seq_len = sample['data'].shape[0]
        padded_data[i, :seq_len] = sample['data']
        lengths[i] = seq_len
        labels[i] = sample['label']
    
    return {
        'data': padded_data,
        'labels': labels,
        'lengths': lengths,
        'batch_size': len(batch)
    }

# 3. Custom Batch Samplers
class BalancedBatchSampler(BatchSampler):
    """Batch Sampler ที่รับประกันการกระจายของ classes ใน batch"""
    
    def __init__(self, dataset, batch_size, samples_per_class=None):
        """
        Args:
            dataset: dataset ที่มี labels
            batch_size: ขนาด batch
            samples_per_class: จำนวน samples ต่อ class ใน batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # จัดกลุ่ม indices ตาม class
        self.class_indices = defaultdict(list)
        for idx, sample in enumerate(dataset):
            if isinstance(sample, dict):
                label = sample['label'].item()
            else:
                label = sample[1].item() if torch.is_tensor(sample[1]) else sample[1]
            self.class_indices[label].append(idx)
        
        self.num_classes = len(self.class_indices)
        
        if samples_per_class is None:
            self.samples_per_class = batch_size // self.num_classes
        else:
            self.samples_per_class = samples_per_class
        
        # ตรวจสอบความเป็นไปได้
        min_samples = min(len(indices) for indices in self.class_indices.values())
        if self.samples_per_class > min_samples:
            self.samples_per_class = min_samples
            print(f"Warning: Adjusted samples_per_class to {self.samples_per_class}")
    
    def __iter__(self):
        # สุ่ม indices แต่ละ class
        class_iterators = {}
        for class_id, indices in self.class_indices.items():
            shuffled_indices = indices.copy()
            random.shuffle(shuffled_indices)
            class_iterators[class_id] = iter(shuffled_indices)
        
        # สร้าง batches
        while True:
            batch = []
            
            # เลือก samples จากแต่ละ class
            for class_id in self.class_indices:
                try:
                    for _ in range(self.samples_per_class):
                        batch.append(next(class_iterators[class_id]))
                except StopIteration:
                    # Class นี้หมดแล้ว
                    return
            
            # เติม batch ให้เต็มถ้าจำเป็น
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                # เลือก class ที่ยังมี samples
                available_classes = [cid for cid in self.class_indices 
                                   if len(self.class_indices[cid]) > 0]
                if not available_classes:
                    break
                
                for _ in range(remaining):
                    class_id = random.choice(available_classes)
                    try:
                        batch.append(next(class_iterators[class_id]))
                    except StopIteration:
                        continue
            
            if len(batch) == self.batch_size:
                yield batch
            else:
                break
    
    def __len__(self):
        min_samples = min(len(indices) for indices in self.class_indices.values())
        return min_samples // self.samples_per_class

class DynamicBatchSampler(BatchSampler):
    """Batch Sampler ที่ปรับขนาด batch ตามความซับซ้อนของข้อมูล"""
    
    def __init__(self, dataset, min_batch_size=8, max_batch_size=64, 
                 complexity_fn=None):
        """
        Args:
            dataset: dataset
            min_batch_size: ขนาด batch ต่ำสุด
            max_batch_size: ขนาด batch สูงสุด
            complexity_fn: function ที่คำนวณความซับซ้อน
        """
        self.dataset = dataset
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        if complexity_fn is None:
            self.complexity_fn = self._default_complexity_fn
        else:
            self.complexity_fn = complexity_fn
        
        # คำนวณความซับซ้อนของแต่ละ sample
        self.complexities = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            complexity = self.complexity_fn(sample)
            self.complexities.append((idx, complexity))
        
        # เรียงตามความซับซ้อน
        self.complexities.sort(key=lambda x: x[1])
    
    def _default_complexity_fn(self, sample):
        """คำนวณความซับซ้อนแบบง่าย (variance ของข้อมูล)"""
        if isinstance(sample, dict):
            data = sample['data']
        else:
            data = sample[0]
        
        return torch.var(data).item()
    
    def __iter__(self):
        current_idx = 0
        
        while current_idx < len(self.complexities):
            # กำหนดขนาด batch ตามความซับซ้อน
            _, complexity = self.complexities[current_idx]
            
            if complexity < 0.5:  # ข้อมูลง่าย
                batch_size = self.max_batch_size
            elif complexity < 1.0:  # ข้อมูลปานกลาง
                batch_size = (self.min_batch_size + self.max_batch_size) // 2
            else:  # ข้อมูลซับซ้อน
                batch_size = self.min_batch_size
            
            # สร้าง batch
            batch = []
            for i in range(batch_size):
                if current_idx + i < len(self.complexities):
                    idx, _ = self.complexities[current_idx + i]
                    batch.append(idx)
            
            current_idx += batch_size
            
            if batch:
                yield batch
    
    def __len__(self):
        return (len(self.complexities) + self.max_batch_size - 1) // self.max_batch_size

# 4. Shuffling Strategies
class ShufflingAnalyzer:
    """Class สำหรับวิเคราะห์ shuffling strategies"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def analyze_shuffling_effect(self, batch_size=32, num_epochs=3):
        """วิเคราะห์ผลของ shuffling"""
        
        print("\n4. Shuffling Analysis")
        print("-" * 25)
        
        strategies = {
            'No Shuffling': False,
            'Random Shuffling': True
        }
        
        results = {}
        
        for strategy_name, shuffle in strategies.items():
            print(f"\nTesting {strategy_name}:")
            
            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=custom_collate_fn
            )
            
            # เก็บลำดับการเกิดของ samples
            sample_orders = []
            class_distributions = []
            
            for epoch in range(num_epochs):
                epoch_samples = []
                epoch_classes = []
                
                for batch in dataloader:
                    sample_ids = batch['sample_ids'].tolist()
                    labels = batch['labels'].tolist()
                    
                    epoch_samples.extend(sample_ids)
                    epoch_classes.extend(labels)
                
                sample_orders.append(epoch_samples)
                class_distributions.append(Counter(epoch_classes))
            
            results[strategy_name] = {
                'sample_orders': sample_orders,
                'class_distributions': class_distributions
            }
            
            # วิเคราะห์ความสม่ำเสมอ
            first_10_samples = [order[:10] for order in sample_orders]
            print(f"  First 10 samples across epochs:")
            for i, samples in enumerate(first_10_samples):
                print(f"    Epoch {i+1}: {samples}")
            
            # วิเคราะห์การกระจายของ class
            print(f"  Class distribution consistency:")
            for i, dist in enumerate(class_distributions):
                print(f"    Epoch {i+1}: {dict(dist)}")
        
        return results

# 5. Memory-Efficient Batch Processing
class MemoryEfficientDataLoader:
    """DataLoader ที่ใช้หน่วยความจำอย่างมีประสิทธิภาพ"""
    
    def __init__(self, dataset, batch_size, shuffle=True, 
                 prefetch_factor=2, persistent_workers=False):
        """
        Args:
            dataset: dataset
            batch_size: ขนาด batch
            shuffle: shuffling หรือไม่
            prefetch_factor: จำนวน batch ที่ prefetch
            persistent_workers: รักษา workers ไว้หรือไม่
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        
    def create_dataloader(self, num_workers=0):
        """สร้าง DataLoader ที่ optimize แล้ว"""
        
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor if num_workers > 0 else 2,
            persistent_workers=self.persistent_workers and num_workers > 0,
            collate_fn=custom_collate_fn
        )
    
    def benchmark_memory_usage(self, num_workers_list=[0, 1, 2]):
        """Benchmark การใช้หน่วยความจำ"""
        
        print("\n5. Memory Usage Benchmark")
        print("-" * 30)
        
        import psutil
        import os
        
        for num_workers in num_workers_list:
            print(f"\nTesting with {num_workers} workers:")
            
            # วัดหน่วยความจำก่อน
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # สร้าง DataLoader
            dataloader = self.create_dataloader(num_workers=num_workers)
            
            # โหลดข้อมูล 10 batches
            batches_loaded = 0
            start_time = time.time()
            
            for i, batch in enumerate(dataloader):
                if i >= 10:
                    break
                batches_loaded += 1
                # จำลองการประมวลผล
                _ = batch['data'].mean()
            
            end_time = time.time()
            
            # วัดหน่วยความจำหลัง
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            print(f"  Memory used: {memory_used:.2f} MB")
            print(f"  Time taken: {end_time - start_time:.3f} seconds")
            print(f"  Batches loaded: {batches_loaded}")

# 6. Batch Statistics Analyzer
class BatchStatsAnalyzer:
    """Class สำหรับวิเคราะห์สถิติของ batch"""
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        
    def analyze_batch_statistics(self, num_batches=None):
        """วิเคราะห์สถิติของ batches"""
        
        print("\n6. Batch Statistics Analysis")
        print("-" * 35)
        
        batch_sizes = []
        class_distributions = []
        data_statistics = []
        
        for i, batch in enumerate(self.dataloader):
            if num_batches and i >= num_batches:
                break
                
            # ขนาด batch
            batch_size = batch['batch_size']
            batch_sizes.append(batch_size)
            
            # การกระจายของ class
            labels = batch['labels']
            class_dist = Counter(labels.numpy())
            class_distributions.append(class_dist)
            
            # สถิติของข้อมูล
            data = batch['data']
            stats = {
                'mean': data.mean().item(),
                'std': data.std().item(),
                'min': data.min().item(),
                'max': data.max().item()
            }
            data_statistics.append(stats)
        
        # สรุปผล
        print(f"Total batches analyzed: {len(batch_sizes)}")
        print(f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, "
              f"avg={np.mean(batch_sizes):.1f}")
        
        # วิเคราะห์ class balance
        all_class_counts = defaultdict(int)
        for dist in class_distributions:
            for class_id, count in dist.items():
                all_class_counts[class_id] += count
        
        print(f"Overall class distribution: {dict(all_class_counts)}")
        
        # สถิติของข้อมูล
        avg_stats = {
            'mean': np.mean([s['mean'] for s in data_statistics]),
            'std': np.mean([s['std'] for s in data_statistics]),
            'min': np.mean([s['min'] for s in data_statistics]),
            'max': np.mean([s['max'] for s in data_statistics])
        }
        
        print(f"Average data statistics across batches:")
        for key, value in avg_stats.items():
            print(f"  {key}: {value:.4f}")
        
        return {
            'batch_sizes': batch_sizes,
            'class_distributions': class_distributions,
            'data_statistics': data_statistics
        }

# 7. Advanced Batch Processing Techniques
def demonstrate_advanced_techniques():
    """แสดงเทคนิค batch processing ขั้นสูง"""
    
    print("\n7. Advanced Batch Processing Techniques")
    print("-" * 45)
    
    # สร้าง dataset
    dataset = BatchTestDataset(num_samples=500, imbalanced=True, add_sample_weight=True)
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # 1. Balanced Batch Sampler
    print("\n7.1 Balanced Batch Sampler")
    print("-" * 30)
    
    balanced_sampler = BalancedBatchSampler(dataset, batch_size=32, samples_per_class=6)
    balanced_loader = DataLoader(
        dataset,
        batch_sampler=balanced_sampler,
        collate_fn=custom_collate_fn
    )
    
    print(f"Balanced loader: {len(balanced_loader)} batches")
    
    # ทดสอบ balance
    sample_batch = next(iter(balanced_loader))
    sample_labels = sample_batch['labels']
    sample_dist = Counter(sample_labels.numpy())
    print(f"Sample batch class distribution: {dict(sample_dist)}")
    
    # 2. Dynamic Batch Sampler
    print("\n7.2 Dynamic Batch Sampler")
    print("-" * 30)
    
    dynamic_sampler = DynamicBatchSampler(
        dataset,
        min_batch_size=8,
        max_batch_size=64
    )
    
    dynamic_loader = DataLoader(
        dataset,
        batch_sampler=dynamic_sampler,
        collate_fn=custom_collate_fn
    )
    
    print(f"Dynamic loader: {len(dynamic_loader)} batches")
    
    # ดูขนาด batch ต่างๆ
    batch_sizes = []
    complexities = []
    
    for i, batch in enumerate(dynamic_loader):
        if i >= 5:  # ดู 5 batches แรก
            break
        
        batch_size = batch['batch_size']
        batch_sizes.append(batch_size)
        
        # คำนวณ complexity เฉลี่ยของ batch
        data = batch['data']
        avg_complexity = torch.var(data, dim=[1, 2]).mean().item()
        complexities.append(avg_complexity)
        
        print(f"  Batch {i+1}: size={batch_size}, avg_complexity={avg_complexity:.3f}")

# ทดสอบการใช้งาน
print("เริ่มการทดสอบ Batch Processing และ Shuffling")

# 1. สร้าง datasets
print("\n1. Creating Test Datasets")
print("-" * 30)

# Balanced dataset
balanced_dataset = BatchTestDataset(
    num_samples=800,
    feature_dim=30,
    num_classes=4,
    imbalanced=False,
    add_sample_weight=False
)

# Imbalanced dataset
imbalanced_dataset = BatchTestDataset(
    num_samples=800,
    feature_dim=30,
    num_classes=4,
    imbalanced=True,
    add_sample_weight=True
)

print(f"Balanced dataset distribution: {balanced_dataset.get_class_distribution()}")
print(f"Imbalanced dataset distribution: {imbalanced_dataset.get_class_distribution()}")

# 2. Basic DataLoader comparison
print("\n2. Basic DataLoader Comparison")
print("-" * 35)

batch_sizes = [16, 32, 64, 128]

for batch_size in batch_sizes:
    loader = DataLoader(
        balanced_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    print(f"Batch size {batch_size:3d}: {len(loader):2d} batches")

# 3. Shuffling Analysis
shuffling_analyzer = ShufflingAnalyzer(balanced_dataset)
shuffling_results = shuffling_analyzer.analyze_shuffling_effect(
    batch_size=32,
    num_epochs=3
)

# 4. Memory Efficient DataLoader
print("\n4. Memory Efficient DataLoader Test")
print("-" * 40)

mem_efficient_loader = MemoryEfficientDataLoader(
    balanced_dataset,
    batch_size=32,
    shuffle=True
)

mem_efficient_loader.benchmark_memory_usage(num_workers_list=[0, 1])

# 5. Batch Statistics Analysis
print("\n5. Batch Statistics Analysis")
print("-" * 35)

standard_loader = DataLoader(
    balanced_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate_fn
)

stats_analyzer = BatchStatsAnalyzer(standard_loader)
batch_stats = stats_analyzer.analyze_batch_statistics(num_batches=10)

# 6. Custom Collate Functions Test
print("\n6. Custom Collate Functions Test")
print("-" * 35)

# ทดสอบ custom collate
sample_batch = next(iter(standard_loader))

print(f"Batch contents:")
for key, value in sample_batch.items():
    if torch.is_tensor(value):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {value}")

# 7. Advanced Techniques Demo
demonstrate_advanced_techniques()

# 8. Performance Comparison
print("\n8. Performance Comparison")
print("-" * 30)

# เปรียบเทียบประสิทธิภาพของ samplers ต่างๆ
samplers_config = [
    ("Standard Random", DataLoader(balanced_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)),
    ("Sequential", DataLoader(balanced_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)),
    ("Weighted Random", DataLoader(
        imbalanced_dataset,
        batch_size=32,
        sampler=WeightedRandomSampler(
            imbalanced_dataset.sample_weights,
            len(imbalanced_dataset),
            replacement=True
        ),
        collate_fn=custom_collate_fn
    ))
]

for name, loader in samplers_config:
    start_time = time.time()
    
    batches_processed = 0
    for i, batch in enumerate(loader):
        if i >= 20:  # ประมวลผล 20 batches
            break
        
        # จำลองการประมวลผล
        _ = batch['data'].mean()
        batches_processed += 1
    
    end_time = time.time()
    
    print(f"{name:20}: {end_time - start_time:.3f}s for {batches_processed} batches")

# 9. Batch Consistency Check
print("\n9. Batch Consistency Check")
print("-" * 30)

# ตรวจสอบความสม่ำเสมอของ batches
consistency_loader = DataLoader(
    balanced_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=custom_collate_fn
)

# รันหลายครั้งกับ seed เดียวกัน
torch.manual_seed(42)
first_run_batches = []
for i, batch in enumerate(consistency_loader):
    if i >= 3:
        break
    first_run_batches.append(batch['sample_ids'].tolist())

torch.manual_seed(42)
second_run_batches = []
for i, batch in enumerate(consistency_loader):
    if i >= 3:
        break
    second_run_batches.append(batch['sample_ids'].tolist())

# เปรียบเทียบ
consistent = True
for i in range(len(first_run_batches)):
    if first_run_batches[i] != second_run_batches[i]:
        consistent = False
        break

print(f"Batch consistency (no shuffle): {'✓' if consistent else '✗'}")

# ทดสอบกับ shuffle
shuffle_loader = DataLoader(
    balanced_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate_fn
)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
first_shuffle_batch = next(iter(shuffle_loader))['sample_ids'].tolist()

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
second_shuffle_batch = next(iter(shuffle_loader))['sample_ids'].tolist()

shuffle_consistent = first_shuffle_batch == second_shuffle_batch
print(f"Batch consistency (with shuffle): {'✓' if shuffle_consistent else '✗'}")

print("\n" + "=" * 60)
print("สรุป Example 3: Batch Processing และ Shuffling")
print("=" * 60)
print("✓ Custom Dataset พร้อม metadata และ sample weights")
print("✓ Custom collate functions สำหรับ batch processing")
print("✓ Balanced และ Dynamic batch samplers")
print("✓ การวิเคราะห์ผลของ shuffling strategies")
print("✓ Memory-efficient DataLoader configuration")
print("✓ Batch statistics analysis และ monitoring")
print("✓ Performance comparison ของ samplers ต่างๆ")
print("✓ Consistency checking สำหรับ reproducibility")
print("=" * 60)