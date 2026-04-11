import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_efv_results(clf_vis, results, ablation_results, save_path='efv_comprehensive_results.png'):
    """
    Plots the comprehensive dashboard for the EFV algorithm.
    Can be adapted slightly for future algorithms that share similar history/metrics.
    """
    h = getattr(clf_vis, 'history_', {})
    if not h:
        print("Warning: Visualization requires the model to have a populated `history_` dict.")
        return

    fig = plt.figure(figsize=(24, 30))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Main title
    fig.text(0.5, 0.99, 'Energy-Frequency-Vibration (EFV) Learning Algorithm',
             fontsize=22, fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.975, '"If you want to find the secrets of the universe, think in terms of energy, frequency and vibration" — Nikola Tesla',
             fontsize=12, ha='center', va='top', style='italic', color='#555')
    
    gs = GridSpec(6, 4, figure=fig, hspace=0.45, wspace=0.35,
                  top=0.96, bottom=0.025, left=0.06, right=0.97)
    
    C = {'e': '#e74c3c', 'f': '#8e44ad', 'v': '#2980b9', 'g': '#27ae60', 'gray': '#95a5a6'}
    
    # === ROW 0: Concept + Algorithm summary ===
    ax = fig.add_subplot(gs[0, :2])
    ax.axis('off')
    concept = """
    ┌──────────────────────────────────────────────────────────────────┐
    │                    ALGORITHM ARCHITECTURE                        │
    │                                                                  │
    │  Input x ──► FREQUENCY TRANSFORM ──► ENERGY SCORING ──► Output  │
    │              (Multi-scale Fourier)    (Boltzmann dist.)          │
    │                      ▲                       ▲                   │
    │                      │                       │                   │
    │                  VIBRATION              VIBRATION                │
    │              (damped noise)        (cosine-annealed LR)         │
    │                                                                  │
    │  FREQUENCY: φ(x) = [cos(2πσ₁Wx), sin(2πσ₁Wx), ..., x]         │
    │  ENERGY:    E(x,k) = -(θk·φ(x) + bk)                           │
    │  VIBRATION: η(t) = η₀·(1-A+A·cos(2πPt/T)·(1-t/2T))           │
    │             noise(t) = σ₀·d^t · N(0,1)                         │
    └──────────────────────────────────────────────────────────────────┘
    """
    ax.text(0.02, 0.95, concept, transform=ax.transAxes, fontsize=9.5,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#ccc'))
    
    # Mapping table
    ax = fig.add_subplot(gs[0, 2:])
    ax.axis('off')
    mapping = """
    ┌────────────────────────────────────────────────────────────────┐
    │              PHYSICS → ML MAPPING                              │
    │                                                                │
    │  ENERGY (Physics)           →  ENERGY (ML)                     │
    │  • Systems minimize energy  →  Training minimizes loss         │
    │  • Stable states = minima   →  Classes = energy attractors     │
    │  • Boltzmann distribution   →  Softmax probability             │
    │  • Temperature T            →  Implicit in LR schedule         │
    │                                                                │
    │  FREQUENCY (Physics)        →  FREQUENCY (ML)                  │
    │  • Fourier decomposition    →  Random Fourier Features         │
    │  • Multi-scale analysis     →  Multiple σ scales               │
    │  • Spectral representation  →  Overcomes spectral bias         │
    │                                                                │
    │  VIBRATION (Physics)        →  VIBRATION (ML)                  │
    │  • Damped oscillation       →  Cosine LR annealing + decay     │
    │  • Exploration → settling   →  Large → small perturbations     │
    │  • Resonance                →  LR warm restarts                │
    └────────────────────────────────────────────────────────────────┘
    """
    ax.text(0.02, 0.95, mapping, transform=ax.transAxes, fontsize=9.5,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#ccc'))
    
    # === ROW 1: Training Dynamics ===
    iters = range(len(h['loss']))
    
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(iters, h['loss'], color=C['e'], lw=1.5, alpha=0.85)
    ax.fill_between(iters, h['loss'], alpha=0.1, color=C['e'])
    ax.set_title('⚡ ENERGY (Loss)', fontweight='bold', fontsize=11, color=C['e'])
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy')
    ax.grid(True, alpha=0.2)
    
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(iters, h['accuracy'], color=C['g'], lw=1.5, alpha=0.85)
    ax.fill_between(iters, h['accuracy'], alpha=0.1, color=C['g'])
    ax.set_title('Accuracy', fontweight='bold', fontsize=11, color=C['g'])
    ax.set_xlabel('Epoch'); ax.set_ylabel('Train Accuracy')
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.2)
    acc_final = h['accuracy'][-1]
    ax.annotate(f'{acc_final:.1%}', xy=(len(h['accuracy'])-1, acc_final),
               fontsize=11, ha='right', color=C['g'], fontweight='bold')
    
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(iters, h['lr'], color=C['v'], lw=1.5, alpha=0.85)
    ax.fill_between(iters, h['lr'], alpha=0.08, color=C['v'])
    ax.set_title('🔊 VIBRATION (Learning Rate)', fontweight='bold', fontsize=11, color=C['v'])
    ax.set_xlabel('Epoch'); ax.set_ylabel('Effective LR')
    ax.grid(True, alpha=0.2)
    ax.annotate('Warm restarts\n(oscillation)', xy=(20, h['lr'][20]), fontsize=8,
               color=C['v'], xytext=(25, 10), textcoords='offset points',
               arrowprops=dict(arrowstyle='->', color=C['v']))
    
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(iters, h['noise'], color=C['v'], lw=1.5, alpha=0.85)
    ax.fill_between(iters, h['noise'], alpha=0.08, color=C['v'])
    ax.set_title('🔊 VIBRATION (Noise)', fontweight='bold', fontsize=11, color=C['v'])
    ax.set_xlabel('Epoch'); ax.set_ylabel('Noise Scale')
    ax.grid(True, alpha=0.2)
    ax.annotate('Explore', xy=(5, h['noise'][5]), fontsize=8, color=C['v'],
               xytext=(15, 5), textcoords='offset points',
               arrowprops=dict(arrowstyle='->', color=C['v']))
    ax.annotate('Settle', xy=(130, h['noise'][min(130, len(h['noise'])-1)]), fontsize=8, color=C['v'],
               xytext=(-30, 10), textcoords='offset points',
               arrowprops=dict(arrowstyle='->', color=C['v']))
    
    # === ROW 2: Frequency Analysis ===
    ax = fig.add_subplot(gs[2, 0:2])
    fw = getattr(clf_vis, 'W_', np.array([]))
    n_per = getattr(clf_vis, 'n_freq_per_scale_', 1)
    if fw.size > 0:
        norms = np.sqrt((fw**2).sum(axis=1))
        scale_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
        for s_idx, scale in enumerate(getattr(clf_vis, 'frequency_scales', [])):
            start = s_idx * n_per
            end = start + n_per
            ax.bar(range(start, end), norms[start:end] * scale, 
                   color=scale_colors[s_idx % len(scale_colors)], alpha=0.7, width=1.0, label=f'Scale σ={scale}')
    ax.set_title('📡 FREQUENCY — Feature Components', fontweight='bold', fontsize=12, color=C['f'])
    ax.set_xlabel('Index')
    ax.set_ylabel('Effective Magnitude')
    ax.legend(fontsize=8, ncol=5)
    ax.grid(True, alpha=0.2, axis='y')
    
    # Weight matrix heatmap
    ax = fig.add_subplot(gs[2, 2:4])
    theta = getattr(clf_vis, 'theta_', np.array([[0]]))
    im = ax.imshow(theta, aspect='auto', cmap='RdBu_r', 
                   vmin=-np.abs(theta).max(), vmax=np.abs(theta).max())
    ax.set_title('Weight Matrix θ (Class × Features)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Class')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Weight')
    
    # === ROWS 3-4: Accuracy Comparison Charts ===
    ds_names = list(results.keys())
    clf_names = list(results[ds_names[0]].keys())
    palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
               '#1abc9c', '#e67e22', '#34495e', '#bdc3c7']
    
    for idx, ds in enumerate(ds_names):
        row = 3 + idx // 2
        col = (idx % 2) * 2
        ax = fig.add_subplot(gs[row, col:col+2])
        
        means = [results[ds][c]['accuracy'][0] for c in clf_names]
        stds = [results[ds][c]['accuracy'][1] for c in clf_names]
        
        bars = ax.barh(range(len(clf_names)), means, xerr=stds, height=0.6,
                       color=palette[:len(clf_names)], alpha=0.85, capsize=3, edgecolor='white', linewidth=0.5)
        bars[0].set_edgecolor('#c0392b')
        bars[0].set_linewidth(2.5)
        
        ax.set_yticks(range(len(clf_names)))
        ax.set_yticklabels(clf_names, fontsize=9)
        ax.set_xlabel('Accuracy (CV)', fontsize=10)
        ax.set_title(f'🎯 Accuracy: {ds}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.12)
        ax.grid(True, axis='x', alpha=0.2)
        ax.axvline(x=means[0], color=C['e'], ls='--', alpha=0.3)
        
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(min(m + s + 0.008, 1.09), i, f'{m:.3f}', va='center', fontsize=8,
                   fontweight='bold' if i == 0 else 'normal',
                   color=C['e'] if i == 0 else '#555')
    
    # === ROW 5: Parameter and Time Comparison (Averages across datasets) ===
    # For params and time, we can show a representative view or average
    ax_params = fig.add_subplot(gs[5, 0])
    ax_time = fig.add_subplot(gs[5, 1])
    
    # Extract params (should be consistent across datasets, but we average just in case)
    avg_params = []
    for c in clf_names:
        avg_params.append(np.mean([results[ds][c]['n_params'] for ds in ds_names]))
    
    bars_p = ax_params.barh(range(len(clf_names)), avg_params, height=0.6,
                            color=palette[:len(clf_names)], alpha=0.8, edgecolor='white')
    ax_params.set_yticks(range(len(clf_names)))
    ax_params.set_yticklabels(clf_names, fontsize=8)
    ax_params.set_xscale('log') # Params often vary by orders of magnitude
    ax_params.set_title('🔢 Parameters (Log Scale)', fontweight='bold', fontsize=11)
    ax_params.grid(True, axis='x', alpha=0.2)
    
    for i, p in enumerate(avg_params):
        ax_params.text(p, i, f' {int(p):,}', va='center', fontsize=7, color='#555')

    # Extract time (average across datasets)
    avg_times = []
    for c in clf_names:
        avg_times.append(np.mean([results[ds][c]['fit_time'][0] for ds in ds_names]))
        
    bars_t = ax_time.barh(range(len(clf_names)), avg_times, height=0.6,
                          color=palette[:len(clf_names)], alpha=0.8, edgecolor='white')
    ax_time.set_yticks(range(len(clf_names)))
    ax_time.set_yticklabels([], fontsize=8)
    ax_time.set_title('⏱️ Avg Train Time (s)', fontweight='bold', fontsize=11)
    ax_time.grid(True, axis='x', alpha=0.2)
    
    for i, t in enumerate(avg_times):
        ax_time.text(t, i, f' {t:.3f}s', va='center', fontsize=7, color='#555')

    # === ABLATION & SUMMARY (Shifted slightly) ===
    # Using Subplots inside the grid for ablation if it exists
    if ablation_results:
        ax_abl = fig.add_subplot(gs[5, 2])
        abl_names = list(ablation_results.keys())
        abl_means = [v[0] for v in ablation_results.values()]
        abl_stds = [v[1] for v in ablation_results.values()]
        abl_colors = [C['e']] + [C['gray']] * (len(abl_names) - 1)
        
        bars = ax_abl.bar(range(len(abl_names)), abl_means, yerr=abl_stds,
                          color=abl_colors, alpha=0.85, capsize=4, width=0.5,
                          edgecolor='white', linewidth=1)
        ax_abl.set_xticks(range(len(abl_names)))
        ax_abl.set_xticklabels(abl_names, rotation=35, ha='right', fontsize=8)
        ax_abl.set_ylabel('Accuracy', fontsize=9)
        ax_abl.set_title('🔬 Ablation Study', fontsize=11, fontweight='bold')
        ax_abl.grid(True, axis='y', alpha=0.2)
        ax_abl.set_ylim(0, 1.2)
        for i, (m, s) in enumerate(zip(abl_means, abl_stds)):
            ax_abl.text(i, m + s + 0.02, f'{m:.3f}', ha='center', fontsize=7, fontweight='bold')
    
    # Summary box
    ax_sum = fig.add_subplot(gs[5, 3])
    ax_sum.axis('off')
    
    lines = ["PERFORMANCE SUMMARY", "═" * 45]
    for ds in ds_names:
        first_clf = clf_names[0]
        ours = results[ds][first_clf]['accuracy'][0]
        all_s = {k: v['accuracy'][0] for k, v in results[ds].items()}
        rank = sorted(all_s.values(), reverse=True).index(ours) + 1
        best_n = max(all_s, key=all_s.get)
        lines.append(f"{ds:18s} Rank {rank}/{len(clf_names)}  (Best: {all_s[best_n]:.3f})")
    
    # Add Efficiency highlight
    first_clf = clf_names[0]
    efv_params = np.mean([results[ds][first_clf]['n_params'] for ds in ds_names])
    efv_time = np.mean([results[ds][first_clf]['fit_time'][0] for ds in ds_names])
    lines.append("─" * 45)
    lines.append(f"EFV Efficiency:")
    lines.append(f"• Total Params: {int(efv_params):,}")
    lines.append(f"• Avg Train Time: {efv_time:.4f}s")
    
    ax_sum.text(0.02, 0.98, "\n".join(lines), transform=ax_sum.transAxes,
                fontsize=8.5, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#ddd'))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"\n  Saved diagram: {save_path}")
