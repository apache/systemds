#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import os.path

# check if data already is downloaded
os.makedirs("scripts/staging/shampoo_optimizer/data", exist_ok=True)
data_exists = os.path.exists("scripts/staging/shampoo_optimizer/data/cifar-10-python.tar.gz")
data_exists = not data_exists

# get data 
base = transforms.ToTensor()
train_raw = datasets.CIFAR10(root="scripts/staging/shampoo_optimizer/data", train=True, download=data_exists, transform=base)
raw_loader = DataLoader(train_raw, batch_size=512, shuffle=False, num_workers=0)

# Mean/Std for each channel
n = 0
channel_sum = torch.zeros(3)
channel_sum_sq = torch.zeros(3)

for x, _ in raw_loader:
    # x: [B,3,32,32] in [0,1]
    b = x.size(0)
    x = x.view(b, 3, -1)  # [B,3,1024]
    channel_sum += x.mean(dim=2).sum(dim=0)
    channel_sum_sq += (x ** 2).view(b, 3, -1).mean(dim=2).sum(dim=0)
    n += b

mean = (channel_sum / n).tolist()
std = ((channel_sum_sq / n - torch.tensor(mean) ** 2).sqrt()).tolist()

# 1. Original (only normalisation)
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# 2. Augmentation (Flip + Crop)
transform_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_base = datasets.CIFAR10(root="scripts/staging/shampoo_optimizer/data", train=True, download=False, transform=transform_base)
train_aug  = datasets.CIFAR10(root="scripts/staging/shampoo_optimizer/data", train=True, download=False, transform=transform_aug)
full_dataset = torch.utils.data.ConcatDataset([train_base, train_aug])
loader = DataLoader(full_dataset, batch_size=128, shuffle=True, num_workers=0)

data = []
for x_batch, y_batch in loader:
    x_batch_reshaped = torch.reshape(x_batch, (x_batch.shape[0], -1))
    batch_reshaped = torch.cat((y_batch.unsqueeze(1), x_batch_reshaped), dim=1)
    data.append(batch_reshaped)

data = torch.vstack(data).cpu().numpy()
df = pd.DataFrame(data)
df[0] = df[0].astype("Int16")
df.to_csv("scripts/staging/shampoo_optimizer/cifar10.csv",index=False, header=False)