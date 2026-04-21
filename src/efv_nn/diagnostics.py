import torch
import os

def debug_print_nan(tensor, label):
    """
    Expensive NaN check with synchronization.
    Checks os.environ["PPC_DEBUG"] dynamically.
    """
    if os.environ.get("PPC_DEBUG", "0") == "1":
        if torch.isnan(tensor).any():
            print(f"!!! [DIAGNOSTIC] NaN DETECTED in: {label} !!!")
            return True
    return False
