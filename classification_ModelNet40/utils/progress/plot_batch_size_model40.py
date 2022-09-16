import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

fig, ax = plt.subplots()
span = [8, 16, 32, 64, 128, 256]
span_pos = [0, 1, 2, 3, 4, 5]
accuracy = [13.21, 92.18, 93.72, 92.67, 0, 0]
text_position = [80.02, 80.15, 80.25, 79.87, 79.50, 78.87]
ax.plot(span, accuracy, color='r', label='CCL e=25', linewidth=2, zorder=3)
ax.scatter(span_pos, accuracy, s=50, marker='o', color='k', zorder=4)

for i, txt in enumerate(accuracy):
    ax.annotate(str(txt), (span[i], text_position[i]+0.05), fontsize=12)

plt.grid()
plt.xlabel('Temporal span', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
#plt.ylim([79.2, 80.5])
# plt.xlim([0, 100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.legend()

plt.savefig('f_span.png')
plt.show()

