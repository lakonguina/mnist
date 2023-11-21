import numpy as np

from tinygrad import Tensor
from tinygrad.nn.state import safe_load, load_state_dict

from extra.datasets import fetch_mnist

from net import Net

net = Net()

_, _, test_img, test_label = fetch_mnist()

state_dict = safe_load("model.safetensors")
load_state_dict(net, state_dict)

avg_acc = 0

for step in range(1000):
    # random sample a batch
    samp = np.random.randint(0, test_img.shape[0], size=(64))
    batch = Tensor(test_img[samp], requires_grad=False)
    # get the corresponding labels
    labels = test_label[samp]

    # forward pass
    out = net(batch)

    # calculate accuracy
    pred = out.argmax(axis=-1).numpy()
    avg_acc += (pred == labels).mean()

    print(pred)

print(f"Test Accuracy: {avg_acc / 1000}")
