# Fresh start

Install anaconda and pytorch in a freshly created ubuntu environment.

## Install

```bash
cd && git clone https://github.com/AndreGraca98/FreshStart.git && cd FreshStart/
source install_anaconda.sh
# Open new terminal window
cd FreshStart/ && source create_torch_env.sh
```

## Try Resnet18 with MNIST example

```bash
python mnist_train_example.py
```
