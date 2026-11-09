import numpy as np
import pandas as pd
from scipy.integrate import simpson
import matplotlib.pyplot as plt

import opencsp.common.lib.tool.file_tools as ft

import opencsp.common.lib.opencsp_path.opencsp_root_path as opencsp_path

PLANK = 6.62607015e-34  # Planck's constant. Units: J * s  (Joule-seconds)
SPEED_OF_LIGHT = 299792458  # Speed of light.    Units: m / s  (meters per second)


def solar_flux_ground_ASTM(am15_dir, am15_file_name):
    ASTM_15_data = pd.read_excel(ft.join(am15_dir, am15_file_name))
    am15_data = ASTM_15_data.iloc[2:].to_numpy()

    wavelengths = am15_data[:, 0]
    direct_gnd_irrad = am15_data[:, 1]

    total_pfd_ground = simpson(y=direct_gnd_irrad, x=wavelengths)  # Produces a value in W/m^2
    return total_pfd_ground


def solar_photon_flux_ground_ASTM(
    am15_dir, am15_file_name, wavelength_scaling_coeff=1.0, wavelength_range=None, transmission_spectrum=None
):
    """
    Calculate solar photon flux at ground level using ASTM G173 data.

    Parameters:
    - am15_dir: Directory containing the ASTM G173 file
    - am15_file_name: Name of the ASTM G173 Excel file
    - wavelength_scaling_coeff: Flat scaling coefficient (used if transmission_spectrum is None)
    - wavelength_range: Tuple of (min_wavelength, max_wavelength) in nm to integrate over
    - transmission_spectrum: Array of transmission coefficients corresponding to wavelengths

    Returns:
    - Photon flux spectral density in photons/(s*m^2)
    """
    ASTM_15_data = pd.read_excel(ft.join(am15_dir, am15_file_name))
    am15_data = ASTM_15_data.iloc[2:].to_numpy()

    wavelengths = am15_data[:, 0]
    # ASTM G173 Solar Spectral Irradiance.
    # Units: W / (m^2 * nm)  --> Which is fundamentally: Joule / (s * m^2 * nm)
    direct_gnd_irrad = am15_data[:, 1]

    # Apply wavelength range filter if specified
    if wavelength_range is not None:
        min_wl, max_wl = wavelength_range
        mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
        wavelengths = wavelengths[mask]
        direct_gnd_irrad = direct_gnd_irrad[mask]

    # =============================================================================
    # UNIT CONVERSIONS & INTERMEDIATE VARIABLES
    # =============================================================================
    # Convert the wavelength values to meters to match the SI units of 'c' (m/s).
    # New Variable Units: m (meters)
    wavelengths_m = wavelengths * 1e-9

    # Use transmission spectrum if provided, otherwise use flat coefficient
    if transmission_spectrum is not None:
        # Ensure transmission spectrum matches wavelength array length
        if len(transmission_spectrum) != len(wavelengths):
            raise ValueError("Transmission spectrum length must match wavelength array length")
        tau_wave = transmission_spectrum
    else:
        tau_wave = np.ones_like(direct_gnd_irrad) * wavelength_scaling_coeff

    # =============================================================================
    # SPECTRAL PHOTON FLUX DENSITY (The Integrand)
    # =============================================================================
    # This step calculates the photon arrival rate per wavelength bin.
    # Dimensional breakdown of this specific calculation line:
    # Numerator:   [Joule / (s * m^2 * nm)] * [Dimensionless] * [m]  =  [Joule * m / (s * m^2 * nm)]
    # Denominator: [Joule * s] * [m / s]                            =  [Joule * m]
    # Result:      [Joule * m / (s * m^2 * nm)] / [Joule * m]       =  1 / (s * m^2 * nm)
    #
    # New Variable Units: photons / (s * m^2 * nm)
    scaled_irr_wave = (direct_gnd_irrad * tau_wave * wavelengths_m) / (PLANK * SPEED_OF_LIGHT)

    # =============================================================================
    # FINAL INTEGRATION
    # =============================================================================
    # Integrate across the wavelength spectrum using Simpson's rule.
    # This numerically multiplies our 'photon_flux_spectral' array by the step
    # size segments along the 'wavelengths_nm' axis.
    #
    # Dimensional breakdown of integration:
    # [photons / (s * m^2 * nm)] * [nm]  =  photons / (s * m^2)
    #
    # Final Variable Units: photons * s^-1 * m^-2  (Photons per second per meter squared)
    photon_flux_spectral = simpson(y=scaled_irr_wave, x=wavelengths)
    return photon_flux_spectral


def calculate_photon_flux_density(
    pfd_ground, weather_factor, concentration_factor, camera_lens_area, beam_area, camera_sensor_area_um2
):
    field_irr = pfd_ground * weather_factor  # photons / (s * m^2)
    beam_irr = field_irr * concentration_factor  # photons / (s * m^2)
    camera_lens_irr = beam_irr * (camera_lens_area / beam_area)  # photons / (s * m^2)
    camera_lens_irr_converted = camera_lens_irr * 1 / (10**12)  # photons / (s * m^2) * m^2/um^2
    # camera_lens_irr = beam_irr * camera_lens_area  # photons / (s)
    # camera_sensor_flux_density = camera_lens_irr / camera_sensor_area_um2  # photons / (s * um^2)
    return camera_lens_irr_converted  # photons / (s * um^2) @ the start of the camera lens


def concentration_factor_calc(mirror_facet_area, number_of_facets, spot_diameter):
    area_field = mirror_facet_area * number_of_facets  # meters * meters = meters squared
    spot_area = np.pi * (spot_diameter / 2) ** 2  # meters squared
    concentration = area_field / spot_area
    return concentration


def calculate_necessary_filter_factor(
    FWC, S_dark, I_dark, I_photon, QE, camera_sensor_area_um, ISO, f_num, shutter_speed
):
    # electrons/second - electrons/second
    term_1 = ((FWC - S_dark) / shutter_speed) - I_dark
    # photons/(second*um^2) * electrons/photon * um^2
    term_2 = I_photon * QE * camera_sensor_area_um
    # Unitless gain value for camera settings compared to ISO 100 and f-1.0 aperture baseline
    term_3 = (ISO / 100) * (1 / f_num**2)
    term_4 = 1 / (term_2 * term_3)

    filter_factor = term_1 * term_4
    return filter_factor


def calculate_optical_density(filter_factor):
    OD = -np.log(filter_factor) / np.log(10)
    return OD


def plot_camera_results(results, heliostat_multiples, heliostat_facet_area, facets_per_heliostat, show_plots=False):
    """
    Generate plots showing filter factors and optical densities for all camera configurations.

    Parameters:
    - results: Dictionary of results from run_camera_analysis
    - heliostat_multiples: Array of heliostat counts
    - heliostat_facet_area: Area of each heliostat facet in m²
    - facets_per_heliostat: Number of facets per heliostat
    """
    # Create figure with two subplots (one for each metric)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Define a set of colors and markers for different configurations
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']

    # Create a list to store all lines for the legend
    lines = []
    labels = []

    # Calculate total heliostat area for title
    total_heliostat_area = heliostat_facet_area * facets_per_heliostat

    # Plot filter factors
    for i, (config_key, data) in enumerate(results.items()):
        config = data['config']
        label = (
            f"{config['name']} | "
            f"f/{config['aperture']} | "
            f"ISO {config['ISO']} | "
            f"{config['shutter_speed']*1000:.0f}ms"
        )

        # Use consistent color and marker for this configuration
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        (line1,) = ax1.plot(
            heliostat_multiples,
            data['filter_factors'],
            color=color,
            marker=marker,
            linestyle='-',
            markersize=5,
            linewidth=2,
            label=label,
        )

        # Store line and label for legend
        lines.append(line1)
        labels.append(label)

    ax1.set_ylabel('Filter Factor')
    ax1.set_title(
        f'Filter Factor vs Number of Heliostats with {facets_per_heliostat} facets\n'
        f'{total_heliostat_area:.2f} m² total per heliostat'
    )
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot optical densities with same color/marker combinations
    for i, (config_key, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax2.plot(
            heliostat_multiples,
            data['optical_densities'],
            color=color,
            marker=marker,
            linestyle='-',
            markersize=5,
            linewidth=2,
        )

    ax2.set_xlabel('Number of Heliostats')
    ax2.set_ylabel('Optical Density')
    ax2.set_title(
        f'Filter Factor vs Number of Heliostats with {facets_per_heliostat} facets\n'
        f'{total_heliostat_area:.2f} m² total per heliostat'
    )
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a single legend for both plots
    fig.legend(lines, labels, loc='right', fancybox=True, shadow=True, ncol=1)

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for legend
    if show_plots:
        plt.show()
    return fig, (ax1, ax2)


def run_camera_analysis(
    camera_configs,
    heliostat_multiples,
    facets_per_heliostat,
    facet_area,
    heliostat_beam_size,
    weather_attn_factor,
    camera_lens_diameter,
    pfd_ground,
):
    """
    Run analysis for multiple camera configurations and return structured results.

    Parameters:
    - camera_configs: List of camera configuration dictionaries
    - heliostat_multiples: Array of heliostat counts
    - facets_per_heliostat: Array containing the number of facets per heliostat
    - facet_area: area of a heliostat facet in meters squared
    - heliostat_beam_size: Array of beam sizes (one per heliostat count)
    - weather_attn_factor: Weather attenuation factor
    - camera_lens_diameter: Diameter of camera lens in meters
    - pfd_ground: Ground photon flux density

    Returns:
    - Dictionary containing all results organized by camera configuration
    """
    results = {}

    for config in camera_configs:
        # Generate a unique key for this configuration
        config_key = (config['name'], config['aperture'], config['ISO'], config['shutter_speed'])

        filter_factors = []
        optical_densities = []
        concentrations = []
        photon_fluxes = []

        for helios, beam, fac_area, num_fac in zip(
            heliostat_multiples, heliostat_beam_size, facet_area, facets_per_heliostat
        ):
            concentration_factor = concentration_factor_calc(
                mirror_facet_area=fac_area, number_of_facets=num_fac * helios, spot_diameter=beam
            )
            concentrations.append(concentration_factor)

            I_photon_lens = calculate_photon_flux_density(
                pfd_ground=pfd_ground,
                weather_factor=weather_attn_factor,
                concentration_factor=concentration_factor,
                camera_lens_area=np.pi * (camera_lens_diameter / 2) ** 2,
                beam_area=beam,
                camera_sensor_area_um2=config["sensor_area"],
            )
            I_photon = I_photon_lens * config["lens_trans"]
            photon_fluxes.append(I_photon)

            filter_factor = calculate_necessary_filter_factor(
                FWC=config["FWC"],
                S_dark=config["S_dark"],
                I_dark=config["I_dark"],
                I_photon=I_photon,
                QE=config["QE"],
                camera_sensor_area_um=config["sensor_area"],
                ISO=config["ISO"],
                f_num=config["aperture"],
                shutter_speed=config["shutter_speed"],
            )
            filter_factors.append(filter_factor)

            optical_density = calculate_optical_density(filter_factor)
            optical_densities.append(optical_density)

        # Store results for this configuration
        results[config_key] = {
            'config': config,
            'filter_factors': filter_factors,
            'optical_densities': optical_densities,
            'concentrations': concentrations,
            'photon_fluxes': photon_fluxes,
        }

    return results


if __name__ == "__main__":

    # Define base camera configurations
    base_configs = {
        "Nikon D3300": {
            "FWC": 34446,
            "S_dark": 2,
            "I_dark": 518400,
            "sensor_area": 15.13,
            "QE": 0.56,
            "lens_trans": 0.95,
        },
        "Nikon D3200": {
            "FWC": 28746,
            "S_dark": 3,
            "I_dark": 518400,
            "sensor_area": 14.28,
            "QE": 0.43,
            "lens_trans": 0.95,
        },
    }

    # Generate camera configurations with different settings
    camera_configs = []

    simulation_cases = 50

    heliostat_multiples = np.linspace(1, simulation_cases, simulation_cases)
    facets_per_heliostat = np.full(simulation_cases, 1)
    facet_area = np.full(simulation_cases, 1.2192**2)  # m^2
    heliostat_beam_size = np.full(simulation_cases, 1)

    wavelength_scaling_array = 1
    weather_attn_factor = 1
    camera_lens_diameter = 0.052  # meters for reflective ND 4.0 Optic Installed

    pfd_ground = solar_photon_flux_ground_ASTM(
        am15_dir=opencsp_path.opencsp_top_level_dir(),
        am15_file_name="ASTM_G173-23_Direct_Normal_and_Hemispherical_Irradiance_Standard_Air_Mass_1.5.xlsx",
        wavelength_scaling_coeff=wavelength_scaling_array,
        wavelength_range=(280, 4000),
    )
    # For each camera model
    for model, base in base_configs.items():
        # Test different apertures
        for aperture in [5.6]:
            # Test different ISOs
            for iso in [100, 200, 1000]:
                # Test different shutter speeds
                for shutter_speed in [1 / 30]:
                    config = base.copy()
                    config.update({"name": model, "aperture": aperture, "ISO": iso, "shutter_speed": shutter_speed})
                    camera_configs.append(config)

    # Run analysis
    results = run_camera_analysis(
        camera_configs=camera_configs,
        heliostat_multiples=heliostat_multiples,
        facets_per_heliostat=facets_per_heliostat,
        facet_area=facet_area,
        heliostat_beam_size=heliostat_beam_size,
        weather_attn_factor=weather_attn_factor,
        camera_lens_diameter=camera_lens_diameter,
        pfd_ground=pfd_ground,
    )

    # Generate plots
    figure, axes = plot_camera_results(
        results,
        heliostat_multiples,
        heliostat_facet_area=facet_area[0],
        facets_per_heliostat=facets_per_heliostat[0],
        show_plots=True,
    )
    print("done")
