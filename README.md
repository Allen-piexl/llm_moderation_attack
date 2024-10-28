# Jailbreaking Large Language Models Against Moderation Guardrails via Cipher Characters

This repository contains code for Jailbreaking Large Language Models Against Moderation Guardrails via Cipher Characters. The project is structured to provide a demo (`demo.py`) as well as a comprehensive implementation (`JAM.py`).

## Using Microsoft Azure OpenAI Models

This project utilizes GPT models from Microsoft Azure's OpenAI service. Before proceeding, make sure you have access to Microsoft Azure and have created the necessary GPT models through Azure OpenAI.

### Steps to Create Models on Azure:
1. **Create an Azure Account**: If you do not already have one, [create an Azure account](https://azure.microsoft.com/).
2. **Deploy the OpenAI Service**: Go to the Azure Portal, search for "Azure OpenAI", and deploy the service.
3. **Create Models**: Within the Azure OpenAI service, create the required models. For this project, ensure you have deployed the specific GPT models you plan to use.
4. **Obtain API Keys**: After deployment, obtain your API keys and endpoint from the Azure Portal for integration.

## Environment Setup

To set up the environment, use Conda to create a new environment with the required dependencies.

### Step 1: Create a Conda Environment
```bash
conda create -n JAM python=3.10
conda activate JAM
pip install torch torchvision transformers openai==0.28.0
```

### Step 2: Run JAM.py

## We also provide a Demo for JAM, which is in demo.py
