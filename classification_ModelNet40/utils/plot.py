import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots()
span = [100, 200, 300, 400]
span_pos = [0, 1, 2, 3]
accuracy = [92.71, 93.23, 93.72, 92.99]
text_position = [80.02, 80.15, 80.25, 79.87]
ax.plot(span, accuracy, color='r', label='CCL e=25', linewidth=2, zorder=3)
ax.scatter(span, accuracy, s=50, marker='o', color='k', zorder=4)

for i, txt in enumerate(accuracy):
    ax.annotate(str(txt), (span[i] , accuracy[i]+ 0.05), fontsize=15)

plt.grid()
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
#plt.ylim([79.2, 80.5])
# plt.xlim([0, 100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.legend()

plt.savefig('epoch_modelnet40.png')
plt.show()

