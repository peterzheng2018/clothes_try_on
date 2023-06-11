# diffuser tool for controlnet

## 🛠️ Installation

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
cd tools
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

## 🚀 Quick Start

### dataset

1.get test image

```sh
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png

wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

2.download fill50k dataset

download from huggingface [url](https://huggingface.co/datasets/fusing/fill50k). The original dataset is hosted in the [ControlNet repo](https://huggingface.co/lllyasviel/ControlNet/blob/main/training/fill50k.zip).

3.use demo.sh script

```sh
bash demo.sh
```
