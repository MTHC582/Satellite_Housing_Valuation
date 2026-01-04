"""
DUAL-STREAM NN
Using ResNet50 by MicroSoft & Fine-Tuning it.
A powerful pre-trained model,
which is already smart about shapes and textures.
Since it is already worked on millions of images.
"""

import torch
import torch.nn as nn
from torchvision import models


class ValuationModel(nn.Module):
    def __init__(self, num_numerical_features=15):
        super(ValuationModel, self).__init__()

        # ============================================
        # (CNN) The Eye (Image Processing)
        # ============================================
        # Using ResNet50,
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # ResNet usually outputs 1000 classes (cat, dog, plane...).
        # We replace the last layer to output a "Feature Vector" of size 128 instead.
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_features, 128)

        # ============================================
        # (MLP) The Logic (Numerical Data)
        # ============================================
        # Simple network to process inputs like 'bedrooms', etc.
        self.numerical_net = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # ============================================
        # FUSION: Combining Eye + Logic
        # ============================================
        # Image Vector (128) + Numerical Vector (32) = 160 inputs
        self.head = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # IMP" To prevent ovbrfitting..
            nn.Linear(64, 1),  # Final Output: ONE_NUMBER (The Price)
        )

    def forward(self, image, numerical_data):
        # 1_Process Image
        x_image = self.cnn(image)
        x_image = torch.relu(x_image)

        # 2_Process Numbers
        x_num = self.numerical_net(numerical_data)

        # 3_Fusing stage
        combined = torch.cat((x_image, x_num), dim=1)

        # 4_Final Prediction
        price = self.head(combined)
        return price


# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    print("Testing Model Architecture...")

    # 1_Create a dummy model
    model = ValuationModel(num_numerical_features=15)

    # 2_Create fake data to feed it
    # Batch of 2 images (3 channels, 224x224)
    dummy_images = torch.randn(2, 3, 224, 224)
    # Batch of 2 rows of numerical data (15 columns)
    dummy_features = torch.randn(2, 15)

    # 3_Push data through the model
    output = model(dummy_images, dummy_features)

    print(f"   Model Output Shape: {output.shape}")
    print(f"   (Should be [2, 1] -> 2 prices for 2 houses)")
    print("   SUCCESS: The Model is built correctly!")
