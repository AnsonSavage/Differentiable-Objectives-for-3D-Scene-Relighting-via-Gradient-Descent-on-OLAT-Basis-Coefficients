"""
Feature extraction models for perceptual losses.
"""
import torch
import torch.nn as nn


class VGGIntermediate(nn.Module):
    """VGG feature extractor that captures intermediate layer activations.
    
    Supports VGG16 and VGG19 architectures. Used for style transfer and 
    perceptual losses that compare features at multiple layers.
    
    Args:
        requested: List of layer indices to extract features from
        backbone: Which VGG architecture to use ('vgg16' or 'vgg19')
    """
    def __init__(self, requested=[], backbone='vgg16'):
        super(VGGIntermediate, self).__init__()
        # Use register_buffer so they move to device automatically with the model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        import torchvision.models as models
        self.intermediates = {}
        self.backbone = backbone
        if backbone == 'vgg16':
            self.vgg = models.vgg16(pretrained=True).features.eval()
        elif backbone == 'vgg19':
            self.vgg = models.vgg19(pretrained=True).features.eval()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'vgg16' or 'vgg19'.")
        
        for i, m in enumerate(self.vgg.children()):
            if isinstance(m, nn.ReLU):   # we want to set the relu layers to NOT do the relu in place.
                m.inplace = False          # the model has a hard time going backwards on the in place functions.
            if isinstance(m, nn.MaxPool2d):
                self.vgg[i] = nn.AvgPool2d(2, 2) # In the paper, they used average pools instead of max pools :)
            if i in requested:
                def curry(i):
                    def hook(module, input, output):
                        self.intermediates[i] = output
                    return hook
                m.register_forward_hook(curry(i))

    def forward(self, x):
        self.intermediates = {} # Clear previous activations
        self.vgg(self._normalize(x))
        return self.intermediates

    def _normalize(self, image):
        return (image - self.mean) / self.std
