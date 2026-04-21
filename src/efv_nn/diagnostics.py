import torch
import os

# Global Toggle for Expensive Safety Checks
# Usage: os.environ["PPC_DEBUG"] = "1" to enable
DEBUG_ENABLED = os.environ.get("PPC_DEBUG", "0") == "1"

def debug_print_nan(tensor, label):
    """
    Expensive NaN check with synchronization.
    Only executes if DEBUG_ENABLED is True.
    """
    if DEBUG_ENABLED:
        if torch.isnan(tensor).any():
            print(f"!!! [DIAGNOSTIC] NaN DETECTED in: {label} !!!")
            return True
    return False

def set_debug(enabled: bool):
    global DEBUG_ENABLED
    DEBUG_ENABLED = enabled
    print(f"PPC Diagnostics: {'ENABLED' if enabled else 'DISABLED'}")
