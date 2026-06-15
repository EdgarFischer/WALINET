import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim
from cmcrameri import cm
import h5py

def _metrics(img, ref, clip_pct: float | None = None):
    """Berechnet **relativen RMSE** (rRMSE ∈ [0, 1] bei Min‑Max‑Norm), PSNR (dB) und
    SSIM zwischen zwei 2‑D‑Arrays. Optionale Outlier‑Entfernung über *clip_pct*.

    Args:
        img:   Testbild (2‑D NumPy‑Array)
        ref:   Referenzbild (2‑D NumPy‑Array)
        clip_pct: Prozent der Pixel mit größtem |Fehler|, die ignoriert werden (z. B. 2).
                  None ⇒ kein Trimmen.

    Returns:
        tuple(rRMSE, psnr, ssim_val)
    """
    # gültige Pixel
    mask = np.isfinite(img) & np.isfinite(ref)
    if not np.any(mask):
        return np.nan, np.nan, np.nan

    a, b = img[mask], ref[mask]
    err  = a - b

    # Trimming: oberste *clip_pct* Fehler entfernen
    if clip_pct is not None and 0 < clip_pct < 100:
        thr  = np.percentile(np.abs(err), 100 - clip_pct)
        keep = np.abs(err) <= thr
        a, b = a[keep], b[keep]
        err  = err[keep]

    # RMSE
    mse  = np.mean(err ** 2)
    rmse = np.sqrt(mse)

    # ----- relativer RMSE (Min‑Max‑Normalisierung) -----
    span = b.max() - b.min()
    rrmse = rmse / span if span > 0 else np.nan  # ∈ [0,1] falls Fehler ≤ Span

    # PSNR (nutzt absoluten RMSE, wie üblich)
    peak = b.max() if span > 0 else 1.0  # fallback
    psnr = 20 * np.log10(peak / rmse) if rmse > 0 else np.inf

    # SSIM (maskiert Outlier analog zum Error‑Mask‑Trimming)
    full_mask = mask.copy()
    if clip_pct is not None and 0 < clip_pct < 100:
        full_mask &= (np.abs(img - ref) <= thr)
    ssim_val = ssim(np.nan_to_num(img), np.nan_to_num(ref), data_range=span if span > 0 else 1.0, mask=full_mask)

    return rrmse, psnr, ssim_val


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_metab_maps(
        metabolite: str,
        methods: list[str],
        quality_clip: bool = False,
        outlier_clip: bool = False,
        clip_pct: float | None = None,
        z: int = 10,
        data_dir: str = "MetabMaps",
        scale: str = 'global',      # 'auto' | 'global' | 'percentile' | 'column'
        percentile: tuple[float, float] = (2, 98)
    ) -> None:
    """Visualisiert AMP-Karten und SD-Quotienten mit flexiblem Scaling.

    scale:
      - 'auto'       -> imshow-Autoskalierung pro Bild
      - 'global'     -> ein gemeinsames Min/Max über alle AMP-Maps
      - 'percentile' -> **per-column** (Methode) vmin/vmax nach Perzentilen
      - 'column'     -> per-column Min/Max + eigene Colorbar
    """
    suffix = "OutlierClip" if outlier_clip else ("QualityClip" if quality_clip else "Orig")

    amps, sds = {}, {}
    for m in methods:
        tag    = f"{m}_{suffix}"
        folder = os.path.join(data_dir, m)
        amp_f  = os.path.join(folder, f"{metabolite}_amp_{tag}.npy")
        sd_f   = os.path.join(folder, f"{metabolite}_sd_{tag}.npy")

        if os.path.isfile(amp_f): amps[m] = np.load(amp_f)
        else: print(f"❌ AMP fehlt: {amp_f}")
        if os.path.isfile(sd_f):  sds[m]  = np.load(sd_f)
        else: print(f"⚠️ SD fehlt:  {sd_f}")

    if not amps:
        raise FileNotFoundError("Keine AMP-Dateien gefunden.")

    amp_slice = {k: v[z, ...] for k, v in amps.items()}  # (H, W, T)

    # ---- Skalierungs-Setup ----
    global_vmin = global_vmax = None
    vmin_col, vmax_col = {}, {}

    all_imgs = np.stack(list(amp_slice.values()), axis=0)  # (M, H, W, T)
    if scale == 'global':
        global_vmin = np.nanmin(all_imgs)
        global_vmax = np.nanmax(all_imgs)
    elif scale == 'percentile':
        # NEU: per-column Percentiles
        p_lo, p_hi = percentile
        for k in methods:
            a = amp_slice[k]
            vmin_col[k] = np.nanpercentile(a, p_lo)
            vmax_col[k] = np.nanpercentile(a, p_hi)
    elif scale == 'column':
        for k in methods:
            a = amp_slice[k]
            vmin_col[k] = np.nanmin(a)
            vmax_col[k] = np.nanmax(a)
    # else 'auto': alles None -> imshow-Auto

    # ---- SD-Quotient (letzte/vrletzte Methode) ----
    ratio_key = None
    if len(methods) >= 2 and methods[-1] in sds and methods[-2] in sds:
        ratio_key = f"{methods[-1]}/{methods[-2]}"
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = sds[methods[-1]][z] / sds[methods[-2]][z]
            ratio[np.isinf(ratio)] = np.nan

    keys = methods + ([ratio_key] if ratio_key else [])
    plot_cols = len(keys)

    # Layout: bei 'column' **oder** 'percentile' je Spalte eigene Colorbar
    if scale in ('column', 'percentile'):
        cb_cols = plot_cols
    else:
        cb_cols = 1 + (1 if ratio_key else 0)

    total_cols = plot_cols + cb_cols
    width_ratios = [1] * plot_cols + [0.2] * cb_cols

    T = amps[methods[0]].shape[-1]
    fig = plt.figure(figsize=(4 * plot_cols + 2, 24))
    gs  = gridspec.GridSpec(T, total_cols, width_ratios=width_ratios, wspace=0.3)

    ims_first = {}
    ref_key = methods[-1]

    cb_ax_map = {}
    if scale in ('column', 'percentile'):
        for i, key in enumerate(keys):
            cb_ax_map[key] = fig.add_subplot(gs[:, plot_cols + i])

    for t in range(T):
        ref_img = amp_slice[ref_key][..., t]
        for c, key in enumerate(keys):
            ax = fig.add_subplot(gs[t, c])
            if key != ratio_key:
                img = amp_slice[key][..., t]
                if scale in ('column', 'percentile'):
                    vmin = vmin_col.get(key, None)
                    vmax = vmax_col.get(key, None)
                else:
                    vmin = global_vmin
                    vmax = global_vmax

                im  = ax.imshow(img, cmap="plasma", vmin=vmin, vmax=vmax)
                if key != ref_key:
                    rrmse, psnr, ss = _metrics(img, ref_img, clip_pct)
                    title_metrics = f"rRMSE:{rrmse:.3f} PSNR:{psnr:.1f} SSIM:{ss:.2f}"
                else:
                    title_metrics = "Referenz"
                ax.set_title(f"{metabolite} {key}, T={t+1}\n{title_metrics}", fontsize=8)
            else:
                im = ax.imshow(ratio[..., t], cmap="plasma")
                ax.set_title(f"SD {ratio_key}, T={t+1}", fontsize=8)

            ax.axis("off")
            if t == 0:
                ims_first[key] = im

    # ---- Colorbars ----
    if scale in ('column', 'percentile'):
        for key in keys:
            cax = cb_ax_map[key]
            cb  = fig.colorbar(ims_first[key], cax=cax)
            label = f"{metabolite} AMP" if key != ratio_key else f"{metabolite} SD-Ratio"
            cb.set_label(label)
            cax.yaxis.set_ticks_position("right")
    else:
        cax_amp = fig.add_subplot(gs[:, plot_cols])
        cb_amp = fig.colorbar(ims_first[methods[0]], cax=cax_amp)
        cb_amp.set_label(f"{metabolite} AMP")
        cax_amp.yaxis.set_ticks_position("right")

        if ratio_key:
            cax_ratio = fig.add_subplot(gs[:, plot_cols+1])
            cb_ratio = fig.colorbar(ims_first[ratio_key], cax=cax_ratio)
            cb_ratio.set_label(f"{metabolite} SD-Ratio")
            cax_ratio.yaxis.set_ticks_position("right")

    plt.tight_layout(rect=[0,0,0.95,1])
    #plt.savefig('MetabMaps.jpg', dpi=200, bbox_inches='tight')
    plt.show()



def plot_metab_ratio(
    metab1: str,
    metab2: str,
    methods: list[str],
    quality_clip: bool = False,
    outlier_clip: bool = False,
    z: int = 10,
    data_dir: str = "MetabMaps",
    scale: str = 'percentile',      # 'auto' | 'global' | 'percentile' | 'column' | 'manual'
    percentile: tuple[float, float] = (2, 98),
    vmin_manual: float = None,      # NEU: manuelles Minimum
    vmax_manual: float = None       # NEU: manuelles Maximum
) -> None:
    """Plottet das Verhältnis AMP(metab1)/AMP(metab2) für mehrere Methoden.
       'percentile' wirkt jetzt **global** (ein gemeinsamer Wertebereich & eine Colorbar).
       'column' wirkt pro Spalte (jede Spalte eigene Colorbar).
       'manual' erlaubt eine benutzerdefinierte Skala über vmin_manual/vmax_manual.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    suffix = "OutlierClip" if outlier_clip else ("QualityClip" if quality_clip else "Orig")

    ratios = {}
    for m in methods:
        folder = os.path.join(data_dir, m)
        f1 = os.path.join(folder, f"{metab1}_amp_{m}_{suffix}.npy")
        f2 = os.path.join(folder, f"{metab2}_amp_{m}_{suffix}.npy")

        if not os.path.isfile(f1):
            print(f"❌ AMP fehlt: {f1}")
            continue
        if not os.path.isfile(f2):
            print(f"❌ AMP fehlt: {f2}")
            continue

        amp1 = np.load(f1)[z, ...]
        amp2 = np.load(f2)[z, ...]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = amp1 / amp2
            ratio[np.isinf(ratio)] = np.nan
        ratios[m] = ratio

    if not ratios:
        raise FileNotFoundError("Keine Ratio-Maps gefunden.")

    methods_plotted = list(ratios.keys())

    # ---- Skalierung ----
    vmin = vmax = None
    vmin_col, vmax_col = {}, {}

    if scale == 'manual':
        vmin, vmax = vmin_manual, vmax_manual

    elif scale == 'global':
        all_imgs = np.stack([ratios[m] for m in methods_plotted], axis=0)
        vmin, vmax = np.nanmin(all_imgs), np.nanmax(all_imgs)

    elif scale == 'percentile':
        p_lo, p_hi = percentile
        all_imgs = np.stack([ratios[m] for m in methods_plotted], axis=0)
        vmin, vmax = np.nanpercentile(all_imgs, p_lo), np.nanpercentile(all_imgs, p_hi)

    elif scale == 'column':
        for m in methods_plotted:
            r = ratios[m]
            vmin_col[m] = np.nanmin(r)
            vmax_col[m] = np.nanmax(r)
    # else: 'auto' -> None

    # ---- Layout ----
    n_methods = len(methods_plotted)
    n_T = next(iter(ratios.values())).shape[-1]
    cb_cols = n_methods if scale == 'column' else 1
    total_cols = n_methods + cb_cols
    width_ratios = [1]*n_methods + [0.2]*cb_cols

    fig = plt.figure(figsize=(4 * n_methods + 2, 4 * n_T))
    gs = gridspec.GridSpec(n_T, total_cols, width_ratios=width_ratios, wspace=0.3)

    ims_first = {}
    cb_ax_map = {}
    if scale == 'column':
        for i, m in enumerate(methods_plotted):
            cb_ax_map[m] = fig.add_subplot(gs[:, n_methods + i])

    # ---- Plots ----
    for c, m in enumerate(methods_plotted):
        ratio = ratios[m]
        for t in range(n_T):
            ax = fig.add_subplot(gs[t, c])
            if scale == 'column':
                vmin_m, vmax_m = vmin_col[m], vmax_col[m]
            else:
                vmin_m, vmax_m = vmin, vmax

            im = ax.imshow(ratio[..., t], cmap=cm.batlow, vmin=vmin_m, vmax=vmax_m)
            ax.set_title(f"{metab1}/{metab2}\n{m}, T={t+1}", fontsize=8)
            ax.axis("off")
            if t == 0:
                ims_first[m] = im

    # ---- Colorbars ----
    if scale == 'column':
        for m in methods_plotted:
            cax = cb_ax_map[m]
            cb = fig.colorbar(ims_first[m], cax=cax)
            cb.set_label(f"{metab1}/{metab2} Ratio ({m})")
            cax.yaxis.set_ticks_position("right")
    else:
        cax = fig.add_subplot(gs[:, -1])
        cb = fig.colorbar(next(iter(ims_first.values())), cax=cax)
        cb.set_label(f"{metab1}/{metab2} Ratio")
        cax.yaxis.set_ticks_position("right")

    plt.tight_layout(rect=[0,0,0.95,1])
    #plt.savefig(f"MetabRatio_{metab1}_{metab2}.jpg", dpi=200, bbox_inches='tight')
    plt.show()

def plot_lcmodel_comparison(
    mat_paths,
    col_titles,
    slice_z=17,
    low_pct=2,
    high_pct=98,
    figsize_per_col=4,
    save_path="lcmodel_comparison.pdf",
    dpi=300,
    mask=None,
    ref_image=None,
    ref_title="Ref"
):
    """
    Plot comparison of LCModel maps across multiple .mat files.
    Saves figure ALWAYS as PDF.

    Parameters
    ----------
    mat_paths : list of str
    col_titles : list of str
    slice_z : int
    low_pct, high_pct : float
    figsize_per_col : float
    save_path : str
        Output filename (will be forced to .pdf)
    dpi : int
    mask : np.ndarray or None
        Optionale 3D-Maske mit Form (Z, Y, X).
    ref_image : np.ndarray or None
        Optionales 3D-Referenzbild (Z, Y, X)
    ref_title : str
        Titel für Referenzbild
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import h5py

    assert len(mat_paths) == len(col_titles), "mat_paths und col_titles müssen gleich lang sein"
    if len(mat_paths) == 0:
        raise ValueError("mat_paths darf nicht leer sein")

    # -------- Loader --------
    def load_lcmodel_maps(mat_path):
        with h5py.File(mat_path, 'r') as f:
            metabos = f['AllMaps']['Metabos']
            titles = []
            for ref in metabos['Title'][0]:
                dset = f[ref]
                titles.append(dset[:].tobytes().decode('utf-16-le'))
            maps = metabos['Normal'][:]
            return titles, np.asarray(maps)

    # -------- Load all files --------
    all_titles, all_maps = [], []
    for p in mat_paths:
        t, m = load_lcmodel_maps(p)
        all_titles.append(t)
        all_maps.append(m)

    # -------- Find common metabolites --------
    base_titles = all_titles[0]
    common_titles = [t for t in base_titles if all(t in tt for tt in all_titles[1:])]

    if not common_titles:
        raise ValueError("Keine gemeinsamen Metabolit-Titel gefunden")

    def idx_for(titles, subset):
        return [titles.index(t) for t in subset]

    maps_per_col = [
        m[idx_for(t, common_titles), ...]
        for t, m in zip(all_titles, all_maps)
    ]

    # -------- Prepare mask --------
    example_slice = maps_per_col[0][0, slice_z, :, :]

    if mask is None:
        mask = np.ones((maps_per_col[0].shape[1], example_slice.shape[0], example_slice.shape[1]), dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.ndim != 3:
            raise ValueError("mask muss 3D sein und Form (Z, Y, X) haben")
        if mask.shape[0] <= slice_z:
            raise ValueError(f"mask hat nur {mask.shape[0]} Slices, slice_z={slice_z} ist ungültig")
        if mask.shape[1:] != example_slice.shape:
            raise ValueError(
                f"mask hat räumliche Form {mask.shape[1:]}, erwartet wird {example_slice.shape}"
            )

    mask_slice = mask[slice_z, :, :] > 0

    # -------- Prepare ref image --------
    has_ref = ref_image is not None
    if has_ref:
        ref_image = np.asarray(ref_image)
        if ref_image.ndim != 3:
            raise ValueError("ref_image muss 3D sein und Form (Z, Y, X) haben")
        if ref_image.shape[0] <= slice_z:
            raise ValueError(f"ref_image hat nur {ref_image.shape[0]} Slices, slice_z={slice_z} ist ungültig")
        if ref_image.shape[1:] != example_slice.shape:
            raise ValueError(
                f"ref_image hat räumliche Form {ref_image.shape[1:]}, erwartet wird {example_slice.shape}"
            )

    # -------- Plot --------
    n = len(common_titles)
    C = len(maps_per_col)
    total_cols = C + (1 if has_ref else 0)

    fig = plt.figure(figsize=(figsize_per_col * total_cols + 1, n * 3))

    gs = gridspec.GridSpec(
        nrows=n, ncols=total_cols + 1,
        width_ratios=[1] * total_cols + [0.05],
        wspace=0.05, hspace=0.25
    )

    for i, title in enumerate(common_titles):
        row_maps = [maps_per_col[c][i, slice_z, :, :] for c in range(C)]

        masked_row_maps = []
        for m in row_maps:
            masked = np.where(mask_slice, m, np.nan)
            masked_row_maps.append(masked)

        valid = [m for m in masked_row_maps if not np.isnan(m).all()]
        if valid:
            vals = np.concatenate([m[~np.isnan(m)] for m in valid])
        else:
            vals = np.array([0.0, 1.0])

        vmin_row, vmax_row = np.percentile(vals, (low_pct, high_pct))

        ims = []
        col_offset = 0

        # --- Ref column ---
        if has_ref:
            ax = fig.add_subplot(gs[i, 0])
            ref_slice = np.where(mask_slice, ref_image[slice_z, :, :], np.nan)
            ref_arr = np.nan_to_num(ref_slice, nan=0.0)

            ax.imshow(
                ref_arr,
                cmap='gray',
                origin='lower'
            )
            ax.set_title(ref_title, fontsize=9)
            ax.axis('off')

            col_offset = 1

        # --- LCModel columns ---
        for c, (m, lbl) in enumerate(zip(masked_row_maps, col_titles)):
            ax = fig.add_subplot(gs[i, c + col_offset])
            arr = np.nan_to_num(m, nan=0.0)

            im = ax.imshow(
                arr,
                cmap='plasma',
                origin='lower',
                vmin=vmin_row,
                vmax=vmax_row
            )

            ax.set_title(f"{title} — {lbl}", fontsize=9)
            ax.axis('off')
            ims.append(im)

        # --- Colorbar ---
        cax = fig.add_subplot(gs[i, total_cols])
        cb = fig.colorbar(ims[0], cax=cax)
        cax.set_title("LCM", fontsize=8)
        cax.tick_params(labelsize=7)

    plt.tight_layout()

    # -------- ALWAYS SAVE AS PDF --------
    if not save_path.endswith(".pdf"):
        save_path += ".pdf"

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Figure gespeichert als PDF: {save_path}")

    plt.show()