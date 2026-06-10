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

    Erweiterungen:
    - x_axis kann z.B. eine ppm-Achse sein.
    - x_label setzt das Label der x-Achse.
    - falls x_label == "ppm", wird die x-Achse invertiert.
    - ppm_min / ppm_max erlauben Bereichsauswahl direkt in ppm.
      Beispiel: ppm_min=2.0, ppm_max=4.7.
      Wenn ppm_min/ppm_max gesetzt sind, überschreiben sie f_min/f_max.
    - metabolite_lines kann ein Dictionary sein:
        {"NAA": 2.01, "Cr": 3.03, ...}
      Dann werden farbige vertikale Linien in allen Spektrenplots gezeichnet.
      Die Beschriftung erscheint nur einmal rechts neben dem letzten Plot.
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

    # x-Achse vorbereiten: Index oder z.B. ppm
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
        if x_axis is None:
            raise ValueError("ppm_min/ppm_max benötigen eine x_axis, z.B. ppm.")

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

    # Definition der Plot-Struktur rechts
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

    # Layout
    ncols_spec = max(1, int(ncols_spec))
    nrows_spec = math.ceil(n_plots / ncols_spec)

    if figsize is None:
        figsize = (6 + 5.2 * ncols_spec, max(4.5, 3.2 * nrows_spec + 1.5))

    fig = plt.figure(figsize=figsize)

    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 4.8],
        wspace=0.28,
    )

    # Linkes Bild
    ax_img = fig.add_subplot(outer[0, 0])

    img0 = image_volume[:, :, z_init]
    im = ax_img.imshow(img0.T, origin="lower", cmap=cmap)

    ax_img.set_title(f"Slice z={z_init}")
    ax_img.set_xlabel("x")
    ax_img.set_ylabel("y")

    marker, = ax_img.plot(x_init, y_init, "ro")

    # Rechte Spektren
    spec_grid = outer[0, 1].subgridspec(
        nrows_spec,
        ncols_spec,
        wspace=0.32,
        hspace=0.45,
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
                linewidth=1.5,
                alpha=0.9,
                linestyle="-",
                label=overlay_labels[0],
                zorder=3,
            )

            line1, = ax.plot(
                x_plot,
                spec1,
                color="C3",
                linewidth=1.5,
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
                linewidth=1.5,
                alpha=0.9,
                label=labels[plot_idx],
                zorder=3,
            )

            current_lines.append(line)

        ax.set_title(labels[plot_idx])
        ax.set_xlabel(x_label)
        ax.set_ylabel("signal")

        if x_label == "ppm":
            ax.invert_xaxis()

        if legend:
            ax.legend(loc="best")

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

    # Platz rechts für Metabolitenliste
    plt.subplots_adjust(bottom=0.18, right=0.84)

    metabolite_artists = []

    def draw_metabolite_lines():
        """
        Zeichnet Metabolitenlinien in alle Spektrenplots.
        Die Textliste wird nur einmal rechts neben dem letzten Spektrenplot gezeichnet.
        Es werden nur Metaboliten angezeigt, die im aktuellen x_plot-Bereich liegen.
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

        # Linien in allen Spektrenplots
        for i, (name, xpos) in enumerate(visible_items):
            color = color_cycle[i % len(color_cycle)]

            for ax in axes_spec:
                line = ax.axvline(
                    xpos,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                    color=color,
                    zorder=1,
                )
                metabolite_artists.append(line)

        # Beschriftung nur einmal beim letzten/rechten Plot
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
        Rückgabe:
            list of list of np.ndarray
            - bei Overlay: [spec0_crop, spec1_crop]
            - bei Single:  [spec_crop]
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

        for plot_def in plot_defs:
            curves = [full_specs[idx] for idx in plot_def["spec_indices"]]
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

    for plot_def in plot_defs:
        curves = []
        for idx in plot_def["spec_indices"]:
            curves.append(spec_init_list[idx][f_min:f_max + 1])
        init_plot_data.append(curves)

    apply_y_limits(init_plot_data)
    draw_metabolite_lines()

    # Slider
    ax_slider = plt.axes([0.20, 0.08, 0.55, 0.03])

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
        ax_img.set_title(f"Slice z={z}")

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