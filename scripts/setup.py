import os

# Create directories for logs and checkpoints if they don’t exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Check for dependencies
try:
    import torch
except ImportError:
    print("Installing dependencies...")
    os.system("pip install -r requirements.txt")

print("Setup complete!")