import numpy as np

def calculate_component_ratios(
    Water,
    Lipids,
    Metabos,
    brain_mask,
    ppm,
    spectral_axis=-2,
    lipid_ppm_max=4.3,
):
    def to_spectrum(fid):
        return np.fft.fftshift(
            np.fft.fft(
                fid,
                axis=spectral_axis,
            ),
            axes=spectral_axis,
        )

    water_spec = to_spectrum(Water)
    lipid_spec = to_spectrum(Lipids)
    metab_spec = to_spectrum(Metabos)

    lipid_mask = ppm < lipid_ppm_max

    water_max = np.max(
        np.abs(water_spec),
        axis=spectral_axis,
    )

    lipid_max = np.max(
        np.abs(
            np.compress(
                lipid_mask,
                lipid_spec,
                axis=spectral_axis,
            )
        ),
        axis=spectral_axis,
    )

    metab_max = np.max(
        np.abs(metab_spec),
        axis=spectral_axis,
    )

    valid = (
        brain_mask.astype(bool)
        & np.isfinite(metab_max)
        & (metab_max > 0)
    )

    water_ratio = np.full(
        metab_max.shape,
        np.nan,
        dtype=np.float32,
    )

    lipid_ratio = np.full_like(
        water_ratio,
        np.nan,
    )

    water_ratio[valid] = (
        water_max[valid]
        / metab_max[valid]
    )

    lipid_ratio[valid] = (
        lipid_max[valid]
        / metab_max[valid]
    )

    return water_ratio, lipid_ratio

def pool_valid_voxels(
    values,
    brain_mask,
):
    valid = (
        brain_mask.astype(bool)
        & np.isfinite(values)
        & (values > 0)
    )

    return values[valid]