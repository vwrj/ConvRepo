PyTorch reimplementation of [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/pdf/1803.07728.pdf).

The results below show validation accuracies on CIFAR-10. Each block of the RotNet is extracted and frozen; classification layers are added on top and fine-tuned. 
  
| Model                      | ConvB1 | ConvB2 | ConvB3 |
|----------------------------|--------|--------|--------|
| RotNet with 3 conv. blocks | 78.34  | 85.06  |  60.2  |
