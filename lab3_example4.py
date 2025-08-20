"""
Lab 6 Example 4: Custom Dataset Creation
เรียนรู้การสร้าง custom datasets สำหรับข้อมูลประเภทต่างๆ
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import time
from abc import ABC, abstractmethod

print("=" * 60)
print("Lab 6 Example 4: Custom Dataset Creation")
print("=" * 60)

# 1. Base Custom Dataset Template
class BaseCustomDataset(Dataset, ABC):
    """Base class สำหรับ custom datasets"""
    
    def __init__(self, data_path=None, transform=None, target_transform=None, 
                 cache_data=False, verbose=True):
        """
        Args:
            data_path: path ไปยังข้อมูล
            transform: การ transform ข้อมูล
            target_transform: การ transform targets
            cache_data: cache ข้อมูลในหน่วยความจำหรือไม่
            verbose: แสดงข้อมูลการโหลดหรือไม่
        """
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.cache_data = cache_data
        self.verbose = verbose
        
        # Cache และ metadata
        self.cached_items = {} if cache_data else None
        self.metadata = {}
        
        # โหลดข้อมูล
        self._load_data()
        
        if self.verbose:
            self._print_dataset_info()
    
    @abstractmethod
    def _load_data(self):
        """Abstract method สำหรับโหลดข้อมูล"""
        pass
    
    @abstractmethod
    def __len__(self):
        """ความยาวของ dataset"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        """ดึงข้อมูล 1 item"""
        pass
    
    def _print_dataset_info(self):
        """แสดงข้อมูลของ dataset"""
        print(f"Dataset loaded: {len(self)} samples")
        if self.metadata:
            print("Metadata:")
            for key, value in self.metadata.items():
                print(f"  {key}: {value}")

# 2. Text Dataset สำหรับ NLP
class TextDataset(BaseCustomDataset):
    """Custom dataset สำหรับข้อมูลข้อความ"""
    
    def __init__(self, texts=None, labels=None, data_path=None, 
                 max_length=128, tokenizer=None, **kwargs):
        """
        Args:
            texts: รายการข้อความ
            labels: รายการ labels
            max_length: ความยาวสูงสุดของ sequence
            tokenizer: tokenizer สำหรับประมวลผลข้อความ
        """
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        
        super().__init__(data_path=data_path, **kwargs)
    
    def _load_data(self):
        """โหลดข้อมูลข้อความ"""
        if self.data_path and os.path.exists(self.data_path):
            # โหลดจากไฟล์
            if self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
                self.texts = df['text'].tolist()
                self.labels = df['label'].tolist() if 'label' in df.columns else None
            elif self.data_path.endswith('.json'):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.texts = [item['text'] for item in data]
                self.labels = [item['label'] for item in data] if 'label' in data[0] else None
        elif self.texts is None:
            # สร้างข้อมูลตัวอย่าง
            self._create_sample_text_data()
        
        # สร้าง vocabulary ถ้าไม่มี tokenizer
        if self.tokenizer is None:
            self.vocab = self._build_vocabulary()
            self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Metadata
        self.metadata = {
            'num_samples': len(self.texts),
            'max_length': self.max_length,
            'vocab_size': len(self.vocab) if hasattr(self, 'vocab') else 'Using external tokenizer',
            'num_classes': len(set(self.labels)) if self.labels else None
        }
    
    def _create_sample_text_data(self):
        """สร้างข้อมูลข้อความตัวอย่าง"""
        # หัวข้อข่าวตัวอย่าง
        sample_texts = [
            "The weather today is sunny and warm",
            "Stock market shows positive growth this quarter",
            "New technology breakthrough in artificial intelligence",
            "Local restaurant serves amazing Thai cuisine",
            "Football team wins championship after intense match",
            "Scientists discover new species in the ocean",
            "Government announces new environmental policies",
            "University students protest for better facilities",
            "Artist creates stunning sculpture for museum",
            "Medical research leads to breakthrough treatment"
        ]
        
        # สร้างข้อความแบบสุ่ม
        categories = ['news', 'sports', 'technology', 'food', 'politics']
        
        self.texts = []
        self.labels = []
        
        for i in range(500):
            # เลือกข้อความแม่แบบ
            base_text = sample_texts[i % len(sample_texts)]
            
            # ปรับแต่งข้อความเล็กน้อย
            words = base_text.split()
            if len(words) > 3:
                # สุ่มเปลี่ยนคำบางคำ
                import random
                if random.random() > 0.7:
                    words[random.randint(0, len(words)-1)] = random.choice(['amazing', 'excellent', 'great', 'wonderful'])
            
            text = ' '.join(words)
            label = categories[i % len(categories)]
            
            self.texts.append(text)
            self.labels.append(label)
        
        # แปลง string labels เป็น numeric
        unique_labels = list(set(self.labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = [label_to_idx[label] for label in self.labels]
        self.label_names = unique_labels
    
    def _build_vocabulary(self):
        """สร้าง vocabulary จากข้อความ"""
        vocab = set()
        for text in self.texts:
            words = text.lower().split()
            vocab.update(words)
        
        # เพิ่ม special tokens
        vocab_list = ['<PAD>', '<UNK>', '<START>', '<END>'] + sorted(list(vocab))
        return vocab_list
    
    def _text_to_sequence(self, text):
        """แปลงข้อความเป็น sequence ของ indices"""
        words = text.lower().split()
        
        if self.tokenizer:
            # ใช้ external tokenizer
            return self.tokenizer(text, max_length=self.max_length, 
                                truncation=True, padding='max_length')
        else:
            # ใช้ vocabulary ของเรา
            sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) 
                       for word in words]
            
            # Truncate หรือ pad
            if len(sequence) > self.max_length - 2:  # -2 สำหรับ START และ END
                sequence = sequence[:self.max_length - 2]
            
            # เพิ่ม START และ END tokens
            sequence = [self.word_to_idx['<START>']] + sequence + [self.word_to_idx['<END>']]
            
            # Padding
            while len(sequence) < self.max_length:
                sequence.append(self.word_to_idx['<PAD>'])
            
            return torch.tensor(sequence, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx] if self.labels else 0
        
        # Cache check
        if self.cached_items and idx in self.cached_items:
            return self.cached_items[idx]
        
        # Process text
        sequence = self._text_to_sequence(text)
        
        # Apply transforms
        if self.transform:
            sequence = self.transform(sequence)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        item = {
            'input_ids': sequence,
            'label': torch.tensor(label, dtype=torch.long),
            'original_text': text,
            'text_length': len(text.split())
        }
        
        # Cache if enabled
        if self.cached_items is not None:
            self.cached_items[idx] = item
        
        return item

# 3. Time Series Dataset
class TimeSeriesDataset(BaseCustomDataset):
    """Custom dataset สำหรับข้อมูล time series"""
    
    def __init__(self, data=None, sequence_length=50, prediction_horizon=1, 
                 features=None, target_column=None, **kwargs):
        """
        Args:
            data: pandas DataFrame หรือ numpy array
            sequence_length: ความยาวของ sequence สำหรับ input
            prediction_horizon: จำนวน time steps ที่ต้องการทำนาย
            features: รายชื่อ features ที่ต้องการใช้
            target_column: column ที่ต้องการทำนาย
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.features = features
        self.target_column = target_column
        self.raw_data = data
        
        super().__init__