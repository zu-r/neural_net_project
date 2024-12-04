import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import io


splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])

image_bytes = df_train.iloc[0]['image.bytes']

image = Image.open(io.BytesIO(image_bytes))  

plt.imshow(image)
plt.axis('off') 
plt.show()

# test change
