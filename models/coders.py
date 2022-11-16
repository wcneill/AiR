import torch
from torch import nn

import numpy as np

from tqdm import trange

import matplotlib.pyplot as plt

from data.datasets import show_tensor_images


def init_weights(module):
    if type(module) == nn.Conv2d:
        nn.init.xavier_normal_(module.weight)


def crop(image: torch.Tensor, new_shape: int):
    if image.shape[-1] == new_shape:
        return image

    pad = image.shape[-1] - new_shape

    if pad % 2 == 0:
        pad = pad // 2
        return image[..., pad:-pad, pad:-pad]

    return image[..., :-pad, :-pad]


def train(device, model, epochs, lr,  train_loader, valid_loader=None, criterion=nn.MSELoss(), display_freq=20, decay_rate=0.5, decay_step=20, save_as="model.pt"):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=decay_rate, step_size=decay_step)

    min_valid_loss = torch.inf
    train_loss = [None]
    valid_loss = [None]

    for epoch in trange(epochs):

        running_tl = 0
        running_vl = 0

        for x, y in train_loader:

            model.train()
            optimizer.zero_grad()

            x = x.to(device).float()
            y = y.to(device).float() if y.numel() else x  # numel is zero if y is empty.
            out = model(x)

            # crop due to pixel loss during convolution.
            x = crop(x, out.shape[-1])

            if y.shape[1] != 0:
                y = crop(y, out.shape[-1]).squeeze(1)
                score = criterion(out, y)
            else:
                score = criterion(out, y)

            score.backward()
            optimizer.step()

            running_tl += score.item()

            if epoch % display_freq == 0:
                print(f"Epoch {epoch}:  U-Net Training loss: {train_loss[-1]}")
                images = torch.vstack([out, crop(x, out.shape[-1])])
                show_tensor_images(images)


        if valid_loader:
            with torch.no_grad():
                model.eval()
                for x_valid, y_valid in valid_loader:
                    x_valid = x_valid.to(device).unsqueeze(1)
                    y_valid = y_valid.to(device).unsqueeze(1)
                    out_valid = model(x_valid)

                    y_valid = crop(y_valid, out_valid.shape[-1]).squeeze(1).to(torch.long)
                    score = criterion(out_valid, y_valid)
                    running_vl += score.item()

            if running_vl < min_valid_loss:
                print(f'---------epoch {epoch}---------')
                print(f'New Minimum Mean Validation Loss: {(running_vl/len(valid_loader)):.6f}')
                print('Saving model...')
                torch.save(model.state_dict(), save_as)
                min_valid_loss = running_vl

            valid_loss.append(running_vl / len(valid_loader))

        train_loss.append(running_tl / len(train_loader))

        scheduler.step()

    return train_loss, valid_loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: str = "valid", sample: str = None):
        super().__init__()

        module_list = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=padding),
            nn.ReLU(),
        ]

        if sample == "down":
            module_list.insert(0, nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        elif sample == "up":
            module_list.append(nn.ConvTranspose2d(in_channels=out_channels, out_channels=(out_channels // 2), kernel_size=(2, 2), stride=2))

        self.block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.block.forward(x)


class ImageEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.compression_block = nn.ModuleList([
            ConvBlock(3, 64),
            ConvBlock(64, 128, sample="down"),
            ConvBlock(128, 256, sample="down"),
            ConvBlock(256, 512, sample="down"),
            ConvBlock(512, 1024, sample="down")
        ])

        self.compression_block.apply(init_weights)

    def forward(self, x):
        outputs = []
        for block in self.compression_block:
            x = block.forward(x)
            outputs.append(x.clone())

        return outputs[:-1], outputs[-1]


class ImageDecoder(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        self.init_upsample = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2, 2), stride=2)
        self.final_projection = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=(1, 1))

        self.blocks = nn.ModuleList([
            ConvBlock(1024, 512, sample="up"),
            ConvBlock(512, 256, sample="up"),
            ConvBlock(256, 128, sample="up"),
            ConvBlock(128, 64, sample=None)])

        self.final_projection.apply(init_weights)
        self.init_upsample.apply(init_weights)
        self.blocks.apply(init_weights)

    def forward(self, x, outputs: list):
        outputs.reverse()
        x = self.init_upsample(x)

        for i in torch.arange(len(self.blocks)):
            target_shape = x.shape[-1]
            o = crop(outputs[i], target_shape)
            x = torch.cat([x, o], dim=1)
            x = self.blocks[i].forward(x)

        x = self.final_projection(x)
        print(x.amax())
        return x


class ConvAutoEncoder(nn.Module):

    """From: https://arxiv.org/pdf/1505.04597.pdf"""

    def __init__(self, num_classes):
        super().__init__()
        self.contraction_path = ImageEncoder()
        self.expansion_path = ImageDecoder(num_classes)

    def forward(self, x):
        outputs, x = self.contraction_path(x)
        x = self.expansion_path(x, outputs)
        return x

    def predict(self, x):
        probabilities = torch.softmax(self.forward(x), dim=1)
        return torch.argmax(probabilities, dim=1)