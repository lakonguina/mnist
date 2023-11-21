import numpy as np

from tinygrad import Tensor
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import safe_save, get_state_dict

from extra.datasets import fetch_mnist

from net import Net

net = Net()
opt = SGD([net.l1.weight, net.l2.weight], lr=0.0003)

# Preprare dataset
train_img, train_label, _, _ = fetch_mnist()

with Tensor.train():
  for step in range(1000):
    # random sample a batch
    samp = np.random.randint(0, train_img.shape[0], size=(64))

    batch = Tensor(train_img[samp], requires_grad=False)
    # get the corresponding labels
    labels = Tensor(train_label[samp])
    # forward pass
    out = net(batch)

    # compute loss
    loss = out.sparse_categorical_crossentropy(labels)

    # zero gradients
    opt.zero_grad()

    # backward pass
    loss.backward()

    # update parameters
    opt.step()

    # calculate accuracy
    pred = out.argmax(axis=-1)
    acc = (pred == labels).mean()

    if step % 100 == 0:
      print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")

# first we need the state dict of our model
state_dict = get_state_dict(net)

# then we can just save it to a file
safe_save(state_dict, "model.safetensors")
