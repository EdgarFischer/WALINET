import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def interactive_spectra_viewer_multi(
    image_volume,
    spectrum_funcs,
    labels=None,
    overlay_labels=None,
    x_axis=None,
    x_label="f",
    f_min=None,
    f_max=None,
    ppm_min=None,
    ppm_max=None,
    z_init=0,
    x_init=0,
    y_init=0,
    cmap="gray",
    figsize=None,
    legend=True,
    ncols_spec=2,
    sharey=False,
    same_ylim=False,
    overlay_first_two=True,
    metabolite_lines=None,
    extra_spectrum_funcs=None,
    extra_labels=None,
    extra_linestyle="-.",
    extra_linewidth=1.4,
    compact=True,
    show_image_axes=False,
):
    """
    Interaktiver Viewer:
    - links: 3D-Bildvolumen image_volume[x, y, z]
    - rechts:
        * falls overlay_first_two=True und mindestens 3 Spektren vorhanden:
            Plot 1 = spectrum 0 + spectrum 1 überlagert
            Plot 2 = spectrum 2 separat
            weitere Spektren ab spectrum 3 jeweils separat
        * sonst:
            alle Spektren separat

    Neue Option:
    - extra_spectrum_funcs:
        optionale zusätzliche Kurven pro angezeigtem Plot.
        Länge muss der Anzahl der angezeigten Plots entsprechen.

        Beispiel bei 2 angezeigten Plots:
            extra_spectrum_funcs=[
                unet_func_for_plot_1,
                unet_func_for_plot_2,
            ]

        Jeder Eintrag darf auch eine Liste von Funktionen sein:
            extra_spectrum_funcs=[
                [unet_func, other_func],
                [unet_func_residual],
            ]

    - extra_labels:
        Labels für diese zusätzlichen Kurven. Gleiche Struktur wie extra_spectrum_funcs.
    """

    image_volume = np.asarray(image_volume)

    if image_volume.ndim != 3:
        raise ValueError("image_volume muss shape (X, Y, Z) haben.")

    if not isinstance(spectrum_funcs, (list, tuple)) or len(spectrum_funcs) == 0:
        raise ValueError("spectrum_funcs muss eine nicht-leere Liste/Tuple von Funktionen sein.")

    X, Y, Z = image_volume.shape
    n_specs = len(spectrum_funcs)

    if not (0 <= z_init < Z):
        raise ValueError(f"z_init muss zwischen 0 und {Z-1} liegen.")
    if not (0 <= x_init < X):
        raise ValueError(f"x_init muss zwischen 0 und {X-1} liegen.")
    if not (0 <= y_init < Y):
        raise ValueError(f"y_init muss zwischen 0 und {Y-1} liegen.")

    # Initialspektren prüfen
    spec_init_list = []
    for func in spectrum_funcs:
        spec = np.asarray(func(x_init, y_init, z_init)).squeeze()
        if spec.ndim != 1:
            raise ValueError("Alle spectrum_funcs müssen 1D-Arrays zurückgeben.")
        spec_init_list.append(spec)

    F = len(spec_init_list[0])

    for i, spec in enumerate(spec_init_list):
        if len(spec) != F:
            raise ValueError(
                f"Alle Spektren müssen die gleiche Länge haben. "
                f"Spektrum 0 hat Länge {F}, Spektrum {i} hat Länge {len(spec)}."
            )

    # x-Achse vorbereiten
    if x_axis is None:
        x_axis = np.arange(F)
    else:
        x_axis = np.asarray(x_axis).squeeze()
        if x_axis.ndim != 1:
            raise ValueError("x_axis muss ein 1D-Array sein.")
        if len(x_axis) != F:
            raise ValueError(
                f"x_axis muss die gleiche Länge wie das Spektrum haben. "
                f"Erwartet: {F}, bekommen: {len(x_axis)}."
            )

    # ppm-Bereich in Indexbereich umrechnen
    if ppm_min is not None or ppm_max is not None:
        if ppm_min is None:
            ppm_min = np.min(x_axis)
        if ppm_max is None:
            ppm_max = np.max(x_axis)

        ppm_low = min(ppm_min, ppm_max)
        ppm_high = max(ppm_min, ppm_max)

        valid_idx = np.where((x_axis >= ppm_low) & (x_axis <= ppm_high))[0]

        if len(valid_idx) == 0:
            raise ValueError(
                f"Kein Punkt der x_axis liegt im Bereich "
                f"{ppm_low} bis {ppm_high} ppm."
            )

        f_min = int(valid_idx.min())
        f_max = int(valid_idx.max())

    if f_min is None:
        f_min = 0
    if f_max is None:
        f_max = F - 1

    if not (0 <= f_min <= f_max < F):
        raise ValueError(f"f_min/f_max müssen im Bereich 0 bis {F-1} liegen.")

    x_plot = x_axis[f_min:f_max + 1]

    use_overlay = overlay_first_two and (n_specs >= 3)

    # Plot-Struktur
    plot_defs = []

    if use_overlay:
        plot_defs.append({
            "type": "overlay",
            "spec_indices": [0, 1],
        })
        plot_defs.append({
            "type": "single",
            "spec_indices": [2],
        })
        for i in range(3, n_specs):
            plot_defs.append({
                "type": "single",
                "spec_indices": [i],
            })
    else:
        for i in range(n_specs):
            plot_defs.append({
                "type": "single",
                "spec_indices": [i],
            })

    n_plots = len(plot_defs)

    # Labels / Plot-Titel
    if labels is None:
        if use_overlay:
            labels = ["Overlay", "Spectrum 3"] + [f"Spectrum {i+1}" for i in range(3, n_specs)]
        else:
            labels = [f"Spectrum {i+1}" for i in range(n_specs)]

    if len(labels) != n_plots:
        raise ValueError(
            f"labels muss Länge {n_plots} haben, hat aber Länge {len(labels)}."
        )

    if overlay_labels is None and use_overlay:
        overlay_labels = ["Spectrum 1", "Spectrum 2"]

    if use_overlay and len(overlay_labels) != 2:
        raise ValueError("overlay_labels muss genau 2 Einträge haben.")

    # Extra-Funktionen normalisieren: pro Plot eine Liste von Funktionen
    def _normalize_extra_funcs(extra_spectrum_funcs):
        if extra_spectrum_funcs is None:
            return [[] for _ in range(n_plots)]

        if len(extra_spectrum_funcs) != n_plots:
            raise ValueError(
                f"extra_spectrum_funcs muss Länge {n_plots} haben, "
                f"hat aber Länge {len(extra_spectrum_funcs)}."
            )

        out = []
        for item in extra_spectrum_funcs:
            if item is None:
                out.append([])
            elif callable(item):
                out.append([item])
            elif isinstance(item, (list, tuple)):
                if not all(callable(f) for f in item):
                    raise ValueError("Alle Einträge in extra_spectrum_funcs müssen Funktionen sein.")
                out.append(list(item))
            else:
                raise ValueError(
                    "Jeder Eintrag in extra_spectrum_funcs muss None, "
                    "eine Funktion oder eine Liste von Funktionen sein."
                )

        return out

    extra_funcs_per_plot = _normalize_extra_funcs(extra_spectrum_funcs)

    # Extra-Labels normalisieren
    def _normalize_extra_labels(extra_labels):
        if extra_labels is None:
            return [
                [f"extra {j+1}" for j in range(len(funcs))]
                for funcs in extra_funcs_per_plot
            ]

        if len(extra_labels) != n_plots:
            raise ValueError(
                f"extra_labels muss Länge {n_plots} haben, "
                f"hat aber Länge {len(extra_labels)}."
            )

        out = []

        for plot_idx, item in enumerate(extra_labels):
            n_extra = len(extra_funcs_per_plot[plot_idx])

            if n_extra == 0:
                out.append([])
                continue

            if isinstance(item, str):
                if n_extra != 1:
                    raise ValueError(
                        "Ein String als extra_label ist nur erlaubt, "
                        "wenn genau eine extra Kurve im Plot liegt."
                    )
                out.append([item])

            elif isinstance(item, (list, tuple)):
                if len(item) != n_extra:
                    raise ValueError(
                        f"extra_labels[{plot_idx}] muss Länge {n_extra} haben, "
                        f"hat aber Länge {len(item)}."
                    )
                out.append(list(item))

            else:
                raise ValueError(
                    "extra_labels muss pro Plot ein String oder eine Liste von Strings sein."
                )

        return out

    extra_labels_per_plot = _normalize_extra_labels(extra_labels)

    # Extra-Initialdaten prüfen
    extra_init_data = []

    for plot_idx, funcs in enumerate(extra_funcs_per_plot):
        curves = []

        for func in funcs:
            spec = np.asarray(func(x_init, y_init, z_init)).squeeze()

            if spec.ndim != 1:
                raise ValueError("Alle extra_spectrum_funcs müssen 1D-Arrays zurückgeben.")
            if len(spec) != F:
                raise ValueError(
                    f"Extra-Spektrum in Plot {plot_idx} hat Länge {len(spec)}, "
                    f"erwartet wurde {F}."
                )

            curves.append(spec[f_min:f_max + 1])

        extra_init_data.append(curves)

    # Layout
    ncols_spec = max(1, int(ncols_spec))
    nrows_spec = math.ceil(n_plots / ncols_spec)

    if figsize is None:
        if compact:
            figsize = (
                4.8 + 4.1 * ncols_spec,
                max(3.6, 2.55 * nrows_spec + 1.0),
            )
        else:
            figsize = (
                6 + 5.2 * ncols_spec,
                max(4.5, 3.2 * nrows_spec + 1.5),
            )

    fig = plt.figure(figsize=figsize)

    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.75, 5.2] if compact else [1.0, 4.8],
        wspace=0.12 if compact else 0.28,
    )

    # Linkes Bild
    ax_img = fig.add_subplot(outer[0, 0])

    img0 = image_volume[:, :, z_init]
    im = ax_img.imshow(img0.T, origin="lower", cmap=cmap)

    ax_img.set_title(f"z={z_init}")
    marker, = ax_img.plot(x_init, y_init, "ro", markersize=4)

    if show_image_axes:
        ax_img.set_xlabel("x")
        ax_img.set_ylabel("y")
    else:
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_xlabel("")
        ax_img.set_ylabel("")

    # Rechte Spektren
    spec_grid = outer[0, 1].subgridspec(
        nrows_spec,
        ncols_spec,
        wspace=0.24 if compact else 0.32,
        hspace=0.32 if compact else 0.45,
    )

    axes_spec = []
    plot_line_groups = []
    shared_ax = None

    for plot_idx, plot_def in enumerate(plot_defs):
        row = plot_idx // ncols_spec
        col = plot_idx % ncols_spec

        if sharey and shared_ax is not None:
            ax = fig.add_subplot(spec_grid[row, col], sharey=shared_ax)
        else:
            ax = fig.add_subplot(spec_grid[row, col])
            if shared_ax is None and sharey:
                shared_ax = ax

        current_lines = []

        if plot_def["type"] == "overlay":
            idx0, idx1 = plot_def["spec_indices"]

            spec0 = spec_init_list[idx0][f_min:f_max + 1]
            spec1 = spec_init_list[idx1][f_min:f_max + 1]

            line0, = ax.plot(
                x_plot,
                spec0,
                color="black",
                linewidth=1.4,
                alpha=0.9,
                linestyle="-",
                label=overlay_labels[0],
                zorder=3,
            )

            line1, = ax.plot(
                x_plot,
                spec1,
                color="C3",
                linewidth=1.4,
                alpha=0.9,
                linestyle="--",
                label=overlay_labels[1],
                zorder=3,
            )

            current_lines.extend([line0, line1])

        else:
            idx = plot_def["spec_indices"][0]
            spec = spec_init_list[idx][f_min:f_max + 1]

            line, = ax.plot(
                x_plot,
                spec,
                linewidth=1.4,
                alpha=0.9,
                label=labels[plot_idx],
                zorder=3,
            )

            current_lines.append(line)

        # Extra-Kurven in denselben Plot
        for extra_idx, extra_curve in enumerate(extra_init_data[plot_idx]):
            line_extra, = ax.plot(
                x_plot,
                extra_curve,
                linewidth=extra_linewidth,
                alpha=0.95,
                linestyle=extra_linestyle,
                label=extra_labels_per_plot[plot_idx][extra_idx],
                zorder=4,
            )
            current_lines.append(line_extra)

        ax.set_title(labels[plot_idx], fontsize=10 if compact else None)
        ax.set_xlabel(x_label)
        ax.set_ylabel("signal")
        ax.margins(x=0.01, y=0.06)

        if x_label == "ppm":
            ax.invert_xaxis()

        if legend:
            ax.legend(loc="best", fontsize=8 if compact else None)

        axes_spec.append(ax)
        plot_line_groups.append(current_lines)

    # Leere Achsen ausblenden
    total_slots = nrows_spec * ncols_spec

    for i in range(n_plots, total_slots):
        row = i // ncols_spec
        col = i % ncols_spec
        ax_empty = fig.add_subplot(spec_grid[row, col])
        ax_empty.axis("off")

    current = {"x": x_init, "y": y_init, "z": z_init}

    # Weniger whitespace:
    # Falls metabolite_lines vorhanden sind, rechts Platz lassen.
    # Sonst fast bis an den Rand gehen.
    right_margin = 0.84 if metabolite_lines else 0.97

    if compact:
        plt.subplots_adjust(
            left=0.035,
            right=right_margin,
            bottom=0.12,
            top=0.94,
        )
    else:
        plt.subplots_adjust(
            left=0.06,
            right=right_margin,
            bottom=0.18,
            top=0.94,
        )

    metabolite_artists = []

    def draw_metabolite_lines():
        """
        Zeichnet Metabolitenlinien in alle Spektrenplots.
        Die Textliste wird nur einmal rechts neben dem letzten Spektrenplot gezeichnet.
        """
        for artist in metabolite_artists:
            artist.remove()
        metabolite_artists.clear()

        if metabolite_lines is None or len(metabolite_lines) == 0:
            return

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        x_low = np.min(x_plot)
        x_high = np.max(x_plot)

        visible_items = [
            (name, xpos)
            for name, xpos in metabolite_lines.items()
            if x_low <= xpos <= x_high
        ]

        if len(visible_items) == 0:
            return

        visible_items = sorted(
            visible_items,
            key=lambda kv: kv[1],
            reverse=True,
        )

        for i, (name, xpos) in enumerate(visible_items):
            color = color_cycle[i % len(color_cycle)]

            for ax in axes_spec:
                line = ax.axvline(
                    xpos,
                    linestyle="--",
                    linewidth=1.0 if compact else 1.2,
                    alpha=0.75,
                    color=color,
                    zorder=1,
                )
                metabolite_artists.append(line)

        label_ax = axes_spec[-1]

        for i, (name, xpos) in enumerate(visible_items):
            color = color_cycle[i % len(color_cycle)]

            text = label_ax.text(
                1.04,
                0.98 - i * 0.075,
                f"{xpos:4.2f}  {name}",
                transform=label_ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color=color,
                family="monospace",
                clip_on=False,
                zorder=5,
            )
            metabolite_artists.append(text)

    def get_current_plot_data(x, y, z):
        """
        Liefert für jeden angezeigten Plot die aktuell geplotteten 1D-Daten.
        Reihenfolge entspricht plot_line_groups.
        """
        full_specs = []

        for func in spectrum_funcs:
            spec = np.asarray(func(x, y, z)).squeeze()

            if spec.ndim != 1:
                raise ValueError("Spektrumsfunktionen müssen 1D-Arrays zurückgeben.")
            if len(spec) != F:
                raise ValueError("Spektrumlänge darf sich nicht zwischen Voxeln ändern.")

            full_specs.append(spec[f_min:f_max + 1])

        plot_data = []

        for plot_idx, plot_def in enumerate(plot_defs):
            curves = [full_specs[idx] for idx in plot_def["spec_indices"]]

            # Extra-Kurven für diesen Plot anhängen
            for func in extra_funcs_per_plot[plot_idx]:
                spec = np.asarray(func(x, y, z)).squeeze()

                if spec.ndim != 1:
                    raise ValueError("Extra-Spektrumsfunktionen müssen 1D-Arrays zurückgeben.")
                if len(spec) != F:
                    raise ValueError("Extra-Spektrumlänge darf sich nicht zwischen Voxeln ändern.")

                curves.append(spec[f_min:f_max + 1])

            plot_data.append(curves)

        return plot_data

    def apply_y_limits(plot_data):
        if same_ylim:
            all_curves = [curve for curves in plot_data for curve in curves]

            y_min = min(np.min(curve) for curve in all_curves)
            y_max = max(np.max(curve) for curve in all_curves)

            if y_min == y_max:
                pad = 1.0 if y_min == 0 else 0.05 * abs(y_min)
                y_min -= pad
                y_max += pad
            else:
                pad = 0.05 * (y_max - y_min)
                y_min -= pad
                y_max += pad

            for ax in axes_spec:
                ax.set_ylim(y_min, y_max)

        else:
            for ax in axes_spec:
                ax.relim()
                ax.autoscale_view()

    # Initiale Skalierung
    init_plot_data = []

    for plot_idx, plot_def in enumerate(plot_defs):
        curves = []

        for idx in plot_def["spec_indices"]:
            curves.append(spec_init_list[idx][f_min:f_max + 1])

        for extra_curve in extra_init_data[plot_idx]:
            curves.append(extra_curve)

        init_plot_data.append(curves)

    apply_y_limits(init_plot_data)
    draw_metabolite_lines()

    # Slider
    if compact:
        ax_slider = plt.axes([0.16, 0.04, 0.62, 0.025])
    else:
        ax_slider = plt.axes([0.20, 0.04, 0.55, 0.03])

    z_slider = Slider(
        ax=ax_slider,
        label="z",
        valmin=0,
        valmax=Z - 1,
        valinit=z_init,
        valstep=1,
    )

    def redraw_spectra():
        x = current["x"]
        y = current["y"]
        z = current["z"]

        plot_data = get_current_plot_data(x, y, z)

        for plot_idx, curves in enumerate(plot_data):
            for line, curve in zip(plot_line_groups[plot_idx], curves):
                line.set_data(x_plot, curve)

        apply_y_limits(plot_data)
        draw_metabolite_lines()

    def onclick(event):
        if event.inaxes != ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if x < 0 or x >= X or y < 0 or y >= Y:
            return

        current["x"] = x
        current["y"] = y

        marker.set_data([x], [y])

        redraw_spectra()
        fig.canvas.draw_idle()

    def update_z(val):
        z = int(z_slider.val)
        current["z"] = z

        new_img = image_volume[:, :, z]
        im.set_data(new_img.T)
        ax_img.set_title(f"z={z}")

        redraw_spectra()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    z_slider.on_changed(update_z)

    plt.show()

    return fig, ax_img, axes_spec, z_slider

def make_ppm_axis(
    vec_size,
    dwelltime,
    larmor_freq,
    ppm_ref=4.7,
    ref_freq_hz=0.0,
    dwelltime_unit="ns",
):
    """
    Erzeugt eine ppm-Achse für ein MRS-Spektrum.

    Parameters
    ----------
    vec_size : int
        Anzahl Spektralpunkte, z.B. 288.

    dwelltime : float
        Dwelltime aus den Metadaten.

    larmor_freq : float
        Larmorfrequenz in Hz, z.B. 123231706.

    ppm_ref : float
        ppm-Wert der Referenz. Für 1H-Wasser meistens 4.7 ppm.

    ref_freq_hz : float
        Position der Referenz auf der FFT-Frequenzachse in Hz.
        Wenn unbekannt: 0.0.

    dwelltime_unit : str
        "s", "ms", "us" oder "ns".

    invert : bool
        Wenn True, wird die Achse so ausgegeben, dass hohe ppm links
        und niedrige ppm rechts stehen.

    Returns
    -------
    ppm : np.ndarray
        ppm-Achse mit Länge vec_size.
    """

    unit_scale = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
    }

    if dwelltime_unit not in unit_scale:
        raise ValueError("dwelltime_unit muss 's', 'ms', 'us' oder 'ns' sein.")

    dt_s = dwelltime * unit_scale[dwelltime_unit]
    bandwidth_hz = 1.0 / dt_s
    larmor_mhz = larmor_freq / 1e6

    freq_hz = np.linspace(
        -bandwidth_hz / 2,
        bandwidth_hz / 2,
        vec_size,
        endpoint=False,
    )

    ppm = ppm_ref - (freq_hz - ref_freq_hz) / larmor_mhz


    return ppm


def make_spec_func(arr4d):
    def f(x, y, z):
        return arr4d[x, y, z, :]
    return f

import numpy as np
import matplotlib.pyplot as plt


def plot_train_inference_grid(
    res,
    indices,
    res_extra=None,
    label="model A",
    extra_label="model B",
    x_axis=None,
    x_label="ppm",
    f_min=None,
    f_max=None,
    ppm_min=None,
    ppm_max=None,
    component="real",
    figsize=None,
    linewidth=1.0,
    alpha=0.9,
    grid=True,
):
    indices = list(indices)

    spectra = res["spectra"]
    target_nuisance = res["target_nuisance"]
    pred_nuisance = res["pred_nuisance"]

    if target_nuisance is None:
        raise ValueError("res['target_nuisance'] is None.")

    metabos_gt = spectra - target_nuisance
    metabos_pred = spectra - pred_nuisance

    if res_extra is not None:
        pred_nuisance_extra = res_extra["pred_nuisance"]
        metabos_pred_extra = res_extra["spectra"] - pred_nuisance_extra
    else:
        pred_nuisance_extra = None
        metabos_pred_extra = None

    N, F = spectra.shape

    # exactly same logic as interactive viewer
    if x_axis is None:
        x_axis = np.arange(F)
    else:
        x_axis = np.asarray(x_axis).squeeze()
        if x_axis.ndim != 1:
            raise ValueError("x_axis must be 1D.")
        if len(x_axis) != F:
            raise ValueError(
                f"x_axis must have same length as spectra. "
                f"Expected {F}, got {len(x_axis)}."
            )

    # ppm range -> index range, same as viewer
    if ppm_min is not None or ppm_max is not None:
        if ppm_min is None:
            ppm_min = np.min(x_axis)
        if ppm_max is None:
            ppm_max = np.max(x_axis)

        ppm_low = min(ppm_min, ppm_max)
        ppm_high = max(ppm_min, ppm_max)

        valid_idx = np.where((x_axis >= ppm_low) & (x_axis <= ppm_high))[0]

        if len(valid_idx) == 0:
            raise ValueError(
                f"No point of x_axis lies in range "
                f"{ppm_low} to {ppm_high} ppm."
            )

        f_min = int(valid_idx.min())
        f_max = int(valid_idx.max())

    if f_min is None:
        f_min = 0
    if f_max is None:
        f_max = F - 1

    if not (0 <= f_min <= f_max < F):
        raise ValueError(f"f_min/f_max must be in range 0 to {F-1}.")

    x_plot = x_axis[f_min:f_max + 1]

    def crop(arr):
        return arr[..., f_min:f_max + 1]

    def pick(arr):
        arr = crop(arr)

        if component == "real":
            return np.real(arr)
        elif component == "imag":
            return np.imag(arr)
        elif component == "abs":
            return np.abs(arr)
        elif component == "phase":
            return np.angle(arr)
        else:
            raise ValueError("component must be 'real', 'imag', 'abs', or 'phase'.")

    n_rows = len(indices)

    if figsize is None:
        figsize = (10.8, max(1.35 * n_rows + 0.8, 2.8))

    # Important: sharex=False, otherwise invert_xaxis can toggle shared axes
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=figsize,
        squeeze=False,
        sharex=False,
    )

    for row, i in enumerate(indices):
        if i < 0 or i >= N:
            raise ValueError(f"Index {i} outside valid range 0..{N-1}.")

        ax0, ax1, ax2 = axes[row]

        ax0.plot(
            x_plot,
            pick(spectra[i]),
            color="black",
            linewidth=linewidth,
            alpha=alpha,
            label="input",
        )

        ax1.plot(
            x_plot,
            pick(target_nuisance[i]),
            color="black",
            linewidth=linewidth,
            alpha=alpha,
            label="nuisance GT",
        )
        ax1.plot(
            x_plot,
            pick(pred_nuisance[i]),
            color="C3",
            linewidth=linewidth,
            alpha=alpha,
            label=f"{label} nuisance",
        )

        if pred_nuisance_extra is not None:
            ax1.plot(
                x_plot,
                pick(pred_nuisance_extra[i]),
                color="C0",
                linewidth=linewidth,
                alpha=alpha,
                linestyle="--",
                label=f"{extra_label} nuisance",
            )

        ax2.plot(
            x_plot,
            pick(metabos_gt[i]),
            color="black",
            linewidth=linewidth,
            alpha=alpha,
            label="metabos GT",
        )
        ax2.plot(
            x_plot,
            pick(metabos_pred[i]),
            color="C3",
            linewidth=linewidth,
            alpha=alpha,
            label=f"{label} metabos",
        )

        if metabos_pred_extra is not None:
            ax2.plot(
                x_plot,
                pick(metabos_pred_extra[i]),
                color="C0",
                linewidth=linewidth,
                alpha=alpha,
                linestyle="--",
                label=f"{extra_label} metabos",
            )

        if row == 0:
            ax0.set_title("Input", fontsize=9)
            ax1.set_title("Nuisance", fontsize=9)
            ax2.set_title("Metabos", fontsize=9)

        ax0.set_ylabel(f"#{i}", fontsize=8)

        for ax in (ax0, ax1, ax2):
            ax.tick_params(axis="both", labelsize=7)
            ax.margins(x=0.01, y=0.05)

            if grid:
                ax.grid(alpha=0.15)

            # same as interactive viewer
            if x_label == "ppm":
                ax.invert_xaxis()

    for ax in axes[-1]:
        ax.set_xlabel(x_label, fontsize=8)

    for ax in axes[0]:
        ax.legend(fontsize=6.5, loc="best", framealpha=0.85)

    fig.tight_layout(pad=0.45, w_pad=0.6, h_pad=0.35)

    return fig, axes

def compute_normalized_nuisance_mse(res, normalization="projection_energy", eps=1e-8):
    spectra = res["spectra"]
    target = res["target_nuisance"]
    pred = res["pred_nuisance"]
    lipid_proj = res["lipid_proj"]

    valid = np.isfinite(spectra).all(axis=1) & np.isfinite(target).all(axis=1) & np.isfinite(pred).all(axis=1)

    spectra = spectra[valid]
    target = target[valid]
    pred = pred[valid]

    if normalization == "projection_energy":
        lipid_proj = lipid_proj[valid]
        norm = np.sqrt(np.sum(np.abs(spectra - lipid_proj) ** 2, axis=1, keepdims=True) + eps)

    elif normalization == "max_abs":
        norm = np.max(np.abs(spectra), axis=1, keepdims=True)

    else:
        raise ValueError("normalization must be 'projection_energy' or 'max_abs'.")

    norm = np.maximum(norm, eps)

    return float(np.mean(np.abs(pred / norm - target / norm) ** 2))