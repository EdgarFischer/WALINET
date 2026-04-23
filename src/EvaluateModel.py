import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math


def interactive_spectra_viewer_multi(
    image_volume,
    spectrum_funcs,
    labels=None,
    overlay_labels=None,
    f_min=None,
    f_max=None,
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

    Parameter
    ----------
    image_volume : np.ndarray
        Array mit shape (X, Y, Z), das links angezeigt wird.

    spectrum_funcs : list of callables
        Liste von Funktionen f(x, y, z) -> 1D array der Länge F

    labels : list of str oder None
        Titel der Einzelplots.
        Bei overlay_first_two=True:
            labels[0] = Titel des Overlay-Plots
            labels[1] = Titel des 3. Spektrums (separat)
            labels[2:] = Titel weiterer Einzelplots

    overlay_labels : list/tuple mit 2 strings oder None
        Legendenlabels für die beiden Kurven im Overlay-Plot.

    f_min, f_max : int oder None
        Frequenzindexbereich für die Anzeige.

    z_init, x_init, y_init : int
        Startposition.

    cmap : str
        Colormap für linkes Bild.

    figsize : tuple oder None
        Figuregröße. Falls None, automatisch gewählt.

    legend : bool
        Ob Legenden angezeigt werden sollen.

    ncols_spec : int
        Anzahl Spalten für die Spektren-Subplots.

    sharey : bool
        Ob alle Spektrenplots die gleiche y-Achse teilen sollen.

    same_ylim : bool
        Falls True, bekommen alle 1D-Plots dieselben y-Limits.

    overlay_first_two : bool
        Falls True und mindestens 3 Spektren vorhanden:
        erste zwei Spektren in einem Overlay-Plot.
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

    if f_min is None:
        f_min = 0
    if f_max is None:
        f_max = F - 1

    if not (0 <= f_min <= f_max < F):
        raise ValueError(f"f_min/f_max müssen im Bereich 0 bis {F-1} liegen.")

    f_axis = np.arange(f_min, f_max + 1)

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
        figsize = (5 + 4 * ncols_spec, max(4, 2.8 * nrows_spec + 1.5))

    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 4],
        wspace=0.3
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
    spec_grid = outer[0, 1].subgridspec(nrows_spec, ncols_spec, wspace=0.3, hspace=0.4)

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
                f_axis,
                spec0,
                color="black",
                linewidth=1.5,
                alpha=0.9,
                linestyle="-",
                label=overlay_labels[0],
            )
            line1, = ax.plot(
                f_axis,
                spec1,
                color="C3",
                linewidth=1.5,
                alpha=0.9,
                linestyle="--",
                label=overlay_labels[1],
            )

            current_lines.extend([line0, line1])

        else:
            idx = plot_def["spec_indices"][0]
            spec = spec_init_list[idx][f_min:f_max + 1]

            line, = ax.plot(
                f_axis,
                spec,
                linewidth=1.5,
                alpha=0.9,
                label=labels[plot_idx],
            )
            current_lines.append(line)

        ax.set_title(labels[plot_idx])
        ax.set_xlabel("f")
        ax.set_ylabel("signal")

        if legend:
            ax.legend()

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

    plt.subplots_adjust(bottom=0.18)

    # Slider
    ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
    z_slider = Slider(
        ax=ax_slider,
        label="z",
        valmin=0,
        valmax=Z - 1,
        valinit=z_init,
        valstep=1
    )

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

    def redraw_spectra():
        x = current["x"]
        y = current["y"]
        z = current["z"]

        plot_data = get_current_plot_data(x, y, z)

        for plot_idx, curves in enumerate(plot_data):
            for line, curve in zip(plot_line_groups[plot_idx], curves):
                line.set_data(f_axis, curve)

        if same_ylim:
            apply_y_limits(plot_data)
        else:
            for ax in axes_spec:
                ax.relim()
                ax.autoscale_view()

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


def make_spec_func(arr4d):
    def f(x, y, z):
        return arr4d[x, y, z, :]
    return f