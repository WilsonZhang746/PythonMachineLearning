# Databricks notebook source
import torch


# COMMAND ----------

torch.version.__version__

# COMMAND ----------

a = torch.ones(3,3)

# COMMAND ----------

b = torch.ones(3,3)

# COMMAND ----------

a + b

# COMMAND ----------

a = a.to('cuda')
b = b.to('cuda')
a + b

# COMMAND ----------

