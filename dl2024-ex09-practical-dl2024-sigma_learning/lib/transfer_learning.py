"""Transfer learning from ResNet model"""

import torch
import torch.nn as nn

from torchvision import models


class ModifiedResNetBase(nn.Module):

    def __init__(self):
        super().__init__()
        # Load ResNet model pretrained on ImageNet data
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        print(self.resnet)

    def disable_gradients(self, model) -> None:
        """
        Freezes the layers of a model
        Args:
            model: The model with the layers to freeze
        Returns:
            None
        """
        # START TODO #############
        # Iterate over model parameters and disable requires_grad
        # This is how we "freeze" these layers (their weights do not change during training)
        for param in model.parameters():
            param.requires_grad = False
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        # START TODO #############
        return self.resnet(x)
        # END TODO #############


class ModifiedResNet1(ModifiedResNetBase):
    def __init__(self):
        super().__init__()
        # START TODO #############
        # Create a new linear layer with 10 outputs
        # Replace the linear of resnet with this new layer. See the model printout for
        # more information
        new_layer = nn.Linear(in_features=self.resnet.fc.in_features, out_features=10)
        self.resnet.fc = new_layer
        # END TODO #############


class ModifiedResNet2(ModifiedResNetBase):
    def __init__(self):
        super().__init__()
        # START TODO #############
        # Create a new linear layer with 10 outputs
        # Replace the linear of resnet with this new layer - this is the only layer we shall train
        # Thus, freeze the other resnet layers by calling disable_gradient. See the model printout for
        # more information
        self.disable_gradients(self.resnet)

        new_layer = nn.Linear(in_features=self.resnet.fc.in_features, out_features=10)
        self.resnet.fc = new_layer
        # END TODO #############


class ModifiedResNet3(ModifiedResNetBase):
    def __init__(self):
        super().__init__()
        # START TODO #############
        # In addition to replacing the output layer of resnet, also allow the second
        # convolutional layer of the second BasicBlock in layer4 of resnet to be trained. See the model printout for
        # more information
        self.disable_gradients(self.resnet)

        new_layer = nn.Linear(in_features=self.resnet.fc.in_features, out_features=10)
        self.resnet.fc = new_layer

        scnd_block_4 = self.resnet.layer4[1]
        conv2 = scnd_block_4.conv2

        for param in conv2.parameters():
            param.requires_grad = True
        # END TODO #############


class ModifiedResNet4(ModifiedResNetBase):
    def __init__(self):
        super().__init__()
        # START TODO #############
        # In addition to replacing the output layer of resnet, also * replace * the second convolutional layer
        # of the second BasicBlock in layer4 of resnet with a same-sized layer with default initialization.
        # See the model printout for more information.
        # Important: In order to pass the test, please first initialize the fully connected layer, and then
        # the convolutional layer!
        self.disable_gradients(self.resnet)

        new_layer = nn.Linear(in_features=self.resnet.fc.in_features, out_features=10)
        self.resnet.fc = new_layer

        scnd_block_4 = self.resnet.layer4[1]

        scnd_block_4.conv2 = nn.Conv2d(
            in_channels=scnd_block_4.conv2.in_channels,
            out_channels=scnd_block_4.conv2.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # END TODO #############


class ModifiedResNet5(ModifiedResNetBase):
    def __init__(self):
        super().__init__()
        # START TODO #############
        # In addition to replacing the output layer of resnet, allow layer4 of resnet to be trained,
        # while freezing all the other layers. See the model printout for
        # more information
        self.disable_gradients(self.resnet)

        new_layer = nn.Linear(in_features=self.resnet.fc.in_features, out_features=10)
        self.resnet.fc = new_layer

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        # END TODO #############
