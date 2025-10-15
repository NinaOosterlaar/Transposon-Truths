import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- 1) Put YOUR fitted params here (from result.params) ----
PARAMS = {
    "inflate_const": 0.213155,
    "inflate_Nucleosome_Distance": -0.839247,
    "inflate_Centromere_Distance": 0.376034,
    "const": 1.996404,
    "Nucleosome_Distance": -0.024525,
    "Centromere_Distance": 1.003764,
    "alpha": 26.942953,  # NB2 variance: Var = mu + alpha * mu^2
}

# ---- 2) Provide the scaler stats you used when fitting (VERY IMPORTANT) ----
# Replace these with the actual mean/std from your training design matrix
SCALER = {
    "mean": {
        "Nucleosome_Distance": 48.217118,    # <-- FILL IN (bp)
        "Centromere_Distance": 103.465971   # <-- FILL IN (kb)
    },
    "std": {
        "Nucleosome_Distance": 41.504013,    # <-- FILL IN (bp)
        "Centromere_Distance": 333.449497,    # <-- FILL IN (kb)
    },
}

# ---------- core math ----------
import numpy as np
import matplotlib.pyplot as plt

def viz_pi_with_marginals(params, scaler,
                          nuc_range_bp=None, centro_range_kb=None,
                          ngrid=200, heat_n=80):
    """
    params: dict/Series with inflate_const, inflate_* and count part (not used here), alpha
    scaler: {'mean': {'Nucleosome_Distance': m_n, 'Centromere_Distance': m_c},
             'std':  {'Nucleosome_Distance': s_n, 'Centromere_Distance': s_c}}
            nuc in bp, centromere in kb (exactly as used at fit-time)
    """

    def z(x, m, s): return (x - m) / (s + 1e-12)
    def sigmoid(t): return 1/(1+np.exp(-t))

    def pi_of(nuc_bp, centro_kb):
        xn = z(nuc_bp, scaler['mean']['Nucleosome_Distance'], scaler['std']['Nucleosome_Distance'])
        xc = z(centro_kb, scaler['mean']['Centromere_Distance'], scaler['std']['Centromere_Distance'])
        eta = (params['inflate_const']
               + params['inflate_Nucleosome_Distance'] * xn
               + params['inflate_Centromere_Distance'] * xc)
        return sigmoid(eta)

    m_n, s_n = scaler['mean']['Nucleosome_Distance'], scaler['std']['Nucleosome_Distance']
    m_c, s_c = scaler['mean']['Centromere_Distance'], scaler['std']['Centromere_Distance']

    # If ranges not given, use mean ± 2 SD (much more informative than full span)
    if nuc_range_bp is None:
        nuc_range_bp = (max(0, m_n - 2*s_n), m_n + 2*s_n)
    if centro_range_kb is None:
        centro_range_kb = (max(0, m_c - 2*s_c), m_c + 2*s_c)

    # --- 1) π vs centromere, with nucleosome fixed at low/mean/high ---
    cen_grid = np.linspace(*centro_range_kb, ngrid)
    nuc_levels = [m_n - s_n, m_n, m_n + s_n]  # ~ 16th, 50th, 84th percentiles
    plt.figure()
    for lvl, label in zip(nuc_levels, ['nuc: −1 SD', 'nuc: mean', 'nuc: +1 SD']):
        plt.plot(cen_grid, pi_of(np.full_like(cen_grid, lvl), cen_grid), label=label)
    plt.xlabel('Centromere distance (kb)')
    plt.ylabel('Structural-zero probability π')
    plt.title('π vs centromere distance')
    plt.legend(frameon=False)
    plt.tight_layout()

    # --- 2) π vs nucleosome, with centromere fixed at low/mean/high ---
    nuc_grid = np.linspace(*nuc_range_bp, ngrid)
    cen_levels = [m_c - s_c, m_c, m_c + s_c]
    plt.figure()
    for lvl, label in zip(cen_levels, ['centro: −1 SD', 'centro: mean', 'centro: +1 SD']):
        plt.plot(nuc_grid, pi_of(nuc_grid, np.full_like(nuc_grid, lvl)), label=label)
    plt.xlabel('Nucleosome distance (bp)')
    plt.ylabel('Structural-zero probability π')
    plt.title('π vs nucleosome distance')
    plt.legend(frameon=False)
    plt.tight_layout()

    # --- 3) Heatmap around mean ± 2 SD with contours for detail ---
    nx = ny = heat_n
    nuc_grid_h = np.linspace(*nuc_range_bp, nx)
    cen_grid_h = np.linspace(*centro_range_kb, ny)
    XX, YY = np.meshgrid(nuc_grid_h, cen_grid_h)
    ZZ = pi_of(XX, YY)

    plt.figure()
    extent = [nuc_grid_h.min(), nuc_grid_h.max(), cen_grid_h.min(), cen_grid_h.max()]
    im = plt.imshow(ZZ, extent=extent, origin='lower', aspect='auto', cmap='viridis')
    cs = plt.contour(XX, YY, ZZ, colors='white', linewidths=0.8, levels=6, alpha=0.8)
    plt.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    plt.xlabel('Nucleosome distance (bp)')
    plt.ylabel('Centromere distance (kb)')
    plt.title('Predicted structural-zero probability π (zoomed)')
    cbar = plt.colorbar(im); cbar.set_label('π')
    # mark mean point
    plt.scatter([m_n], [m_c], s=25, c='red', marker='x', label='mean')
    plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()

    plt.show()

viz_pi_with_marginals(
    PARAMS, SCALER,
    nuc_range_bp=(0, 200),   
    centro_range_kb=(0, 300)       
)