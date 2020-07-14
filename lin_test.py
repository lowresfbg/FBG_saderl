import matplotlib.pyplot as plt
from Dataset.loader import DATASET_5fbg_1

dataset = DATASET_5fbg_1()
plt.plot(*dataset[0])
plt.show()
print(dataset[0])