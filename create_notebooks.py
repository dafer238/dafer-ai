import json

# Week 15: Fine-Tuning and Transfer Learning
week15_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Week 15: Fine-Tuning and Transfer Learning\n\n",
                "This notebook explores transfer learning strategies, including feature extraction, full fine-tuning, and parameter-efficient methods.",
            ],
        },
        {"cell_type": "markdown", "metadata": {}, "source": ["## Setup and Imports"]},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "from torchvision import datasets, transforms, models\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f'Using device: {device}')",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

# Week 16: Deployment
week16_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Week 16: Deployment and Production\n\n",
                "This notebook covers practical model deployment, including building inference APIs and containerization.",
            ],
        },
        {"cell_type": "markdown", "metadata": {}, "source": ["## Setup and Imports"]},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torch.nn as nn\n",
                "import time\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f'Using device: {device}')",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

# Write notebooks
with open(
    r"c:\Users\0206100\code\99_Misc\dafer-ai\07_transfer_learning\week15_finetuning\starter.ipynb",
    "w",
) as f:
    json.dump(week15_notebook, f, indent=2)
    print("Created week15 notebook")

with open(
    r"c:\Users\0206100\code\99_Misc\dafer-ai\08_deployment\week16_deployment\starter.ipynb", "w"
) as f:
    json.dump(week16_notebook, f, indent=2)
    print("Created week16 notebook")

print("\nAll notebooks created successfully!")
