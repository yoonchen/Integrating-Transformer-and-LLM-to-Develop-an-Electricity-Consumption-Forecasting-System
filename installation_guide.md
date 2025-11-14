# Installation Guide

This repository contains Jupyter Notebooks and source code for energy consumption prediction and personalized energy-saving recommendation. Follow the steps below to set up the environment and run the notebooks.

1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
2. Set Up Python Environment

It is recommended to create a virtual environment to avoid any conflicts with other Python packages. You can use virtualenv or conda to create the environment.

Using venv (Built-in Python module)
python3 -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows

Using conda (if you prefer Conda environment)
conda create --name energy-prediction python=3.8
conda activate energy-prediction

3. Install Dependencies

Install the required dependencies for the project. You can install all the dependencies listed in the requirements.txt file.

pip install -r requirements.txt

Required Packages:

tensorflow or torch (depending on your models)

numpy

pandas

scikit-learn

matplotlib

seaborn

jupyter

If requirements.txt is not available or some dependencies are missing, you can manually install the most common packages:

pip install jupyter tensorflow numpy pandas scikit-learn matplotlib seaborn

4. Install GPU Support (Optional)

If you're planning to train models on a GPU (e.g., TensorFlow or PyTorch), you will need to install the GPU version of the necessary libraries. Follow these instructions for either TensorFlow or PyTorch GPU setup:

For TensorFlow GPU:
pip install tensorflow-gpu

For PyTorch GPU:

Follow the installation guide here: PyTorch Installation

5. Start Jupyter Notebook

Once all dependencies are installed, you can start the Jupyter Notebook server:

jupyter notebook


This will open the Jupyter interface in your default web browser. From there, you can navigate to the relevant notebook (e.g., llama_training.ipynb, BERTscore.ipynb, V2_CAG.ipynb) and start running the code.

6. Running the Notebooks
llama_training.ipynb:

This notebook contains the training pipeline for the Llama-based model. It covers data preprocessing, model architecture, and training steps. Be sure to have the required dataset for training. You can find the data loading and preprocessing code inside the notebook.

BERTscore.ipynb:

This notebook demonstrates how to compute BERTScore, an evaluation metric for text generation tasks. The notebook is self-contained and requires no external data other than the input/output text for evaluation.

V2_CAG.ipynb:

This notebook implements the CAG (Content-based Attention Generator) model. It's an experimental model, and the code is still being refined. You can run it, but note that some parts might be incomplete.

7. Troubleshooting

If you encounter issues while setting up the environment or running the notebooks, please refer to the following:

Missing dependencies: Ensure you have installed all packages from requirements.txt and any additional packages listed in the notebooks.

Model weights not available: Some notebooks require pre-trained model weights. If they are missing, you can train your models by running the training notebooks, or refer to the README.md for future updates.

GPU setup issues: If you're using a GPU, make sure that CUDA and cuDNN are correctly installed for TensorFlow or PyTorch (see their official documentation for detailed setup).

8. Contributing

If you encounter bugs or have suggestions for improvements, feel free to create an issue or contribute to the repository with a pull request. Contributions are welcome!

Thank you for using this repository. We hope this guide helps you get started with energy consumption prediction and personalized recommendations.
