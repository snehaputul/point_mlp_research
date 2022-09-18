import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

fig, ax = plt.subplots()
#span = [100, 200, 300, 400]
span_pos = ['0', '1', '2', '3', '4']
accuracy_mdl = [4.05, 11.18, 93.72, 86.47, 92.59]
accuracy_scn = [13.53, 44.41, 86.36, 85.36, 83.73]
text_position = [80.02, 80.15, 80.25, 79.87, 79.5]
ax.scatter(span_pos, accuracy_mdl, s=50, marker='o', color='k', zorder=4)
ax.scatter(span_pos, accuracy_scn, s=50, marker='o', color='k', zorder=4)
ax.plot(span_pos, accuracy_mdl, color='r', label='ModelNet40', linewidth=2, zorder=3)
ax.plot(span_pos, accuracy_scn, color='g', label='ScanObjectNN', linewidth=2, zorder=3)
labels= ['0.2', '0.02', '0.002', '0.0002', '0.00002']
ax.set_xticklabels(labels)


accuracy_mdl_ = [4.05, 11.18, 93.72, 89, 92.59]
accuracy_scn_= [13.53, 44.41, 86.36, 80, 83.73]
for i, txt in enumerate(accuracy_mdl):
    ax.annotate(str(txt), (span_pos[i] , accuracy_mdl_[i]+ 0.05), fontsize=15)

for i, txt in enumerate(accuracy_scn):
    ax.annotate(str(txt), (span_pos[i] , accuracy_scn_[i]+ 0.05), fontsize=15)

plt.grid()
plt.xlabel('Weight decay', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
#plt.ylim([79.2, 80.5])
# plt.xlim([0, 100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()

plt.savefig('weight_decay_modelnet40.png')
plt.show()

