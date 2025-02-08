import matplotlib.pyplot as plt
import json
import os

losses = list()

with open(os.path.join('./losses_lwir_three_channels.json'), 'r') as j:
    losses = json.load(j)

ys = losses
xs = [x for x in range(len(ys))]

plt.plot(xs, ys)
plt.ylabel('loss')

plt.savefig('./loss_graph_lwir_three_channels.png')