import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

fig, ax = plt.subplots()
#span = [100, 200, 300, 400]
span_pos = [100, 200, 300, 400]
accuracy_mdl = [92.71, 93.23, 93.72, 92.99]
accuracy_scn = [80.92, 84.53, 86.36, 85.7]
text_position = [80.02, 80.15, 80.25, 79.87]
ax.scatter(span_pos, accuracy_mdl, s=50, marker='o', color='k', zorder=4)
ax.scatter(span_pos, accuracy_scn, s=50, marker='o', color='k', zorder=4)
ax.plot(span_pos, accuracy_mdl, color='r', label='ModelNet40', linewidth=2, zorder=3)
ax.plot(span_pos, accuracy_scn, color='g', label='ScanObjectNN', linewidth=2, zorder=3)

for i, txt in enumerate(accuracy_mdl):
    ax.annotate(str(txt), (span_pos[i] , accuracy_mdl[i]+ 0.05), fontsize=15)

for i, txt in enumerate(accuracy_scn):
    ax.annotate(str(txt), (span_pos[i] , accuracy_scn[i]+ 0.05), fontsize=15)

plt.grid()
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
#plt.ylim([79.2, 80.5])
# plt.xlim([0, 100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()

plt.savefig('epoch_modelnet40.png')
plt.show()

