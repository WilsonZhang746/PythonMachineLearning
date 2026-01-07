# Databricks notebook source
import os
os.getcwd()    #gettinh current working directory
#Out[8]: 'C:\\Users\\Wei Zhang'   the default working directory
work_path="D:\\Research\\ResearchMethods\\Python-Book-Package\\Stevens-DeepLearningwithPyTorch-2020-Point45\\master\\p1ch2"
os.chdir(work_path)      #setting new working directory




from torchvision import models

# COMMAND ----------

dir(models)

# COMMAND ----------

alexnet = models.AlexNet()

# COMMAND ----------

resnet = models.resnet101(pretrained=True)

# COMMAND ----------

resnet

# COMMAND ----------

from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

# COMMAND ----------

from PIL import Image
img = Image.open("../data/p1ch2/bobby.jpg")

# COMMAND ----------

img
img.show()

# COMMAND ----------

img_t = preprocess(img)

# COMMAND ----------

import torch
batch_t = torch.unsqueeze(img_t, 0)   #adds a new dimension of size 1 at a specified position (dim) 

# COMMAND ----------

resnet.eval()

# COMMAND ----------

out = resnet(batch_t)
out

# COMMAND ----------

with open('../data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# COMMAND ----------

_, index = torch.max(out, 1)

# COMMAND ----------

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()

# COMMAND ----------

_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

# COMMAND ----------

