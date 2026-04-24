# Lazy top-level exports — only import heavy optional deps when explicitly requested
# so that `from efv_nn.ppc_gnn import ...` works without sklearn or matplotlib.

def __getattr__(name):
    if name == "EFVClassifier":
        from .classifier import EFVClassifier
        return EFVClassifier
    if name in ("evaluate_models", "run_ablation"):
        from .experiments import evaluate_models, run_ablation
        return {"evaluate_models": evaluate_models, "run_ablation": run_ablation}[name]
    if name == "plot_efv_results":
        from .visualization import plot_efv_results
        return plot_efv_results
    raise AttributeError(f"module 'efv_nn' has no attribute {name!r}")

__all__ = ["EFVClassifier", "evaluate_models", "run_ablation", "plot_efv_results"]
