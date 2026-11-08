"""Label values and input channel order shared by data loading, models and metrics.

NDWS fire masks use the label values for both the day-t (`prev_fire`) and the
day-t+1 (`next_fire`) masks (Huot et al. 2022). The batch format that uses these
names is described in `docs/batch-format.md`.
"""

FIRE = 1
NO_FIRE = 0
# Cloud cover or an uncertain MODIS detection: the true state is unknown, so these
# pixels are excluded from every loss and metric rather than counted as "no fire".
IGNORE_INDEX = -1

# Order of the channels in `inputs`. Names keep the source quantity and its unit,
# although the values in a batch are cleaned and normalized.
NDWS_CHANNELS: tuple[str, ...] = (
    "elevation_m",
    "wind_dir_sin",
    "wind_dir_cos",
    "wind_speed_m_s",
    "temp_min_k",
    "temp_max_k",
    "humidity_kg_kg",
    "precip_mm",
    "drought_index",
    "ndvi",
    "population_per_km2",
    "energy_release",
    "prev_fire",
)
CH: dict[str, int] = {name: i for i, name in enumerate(NDWS_CHANNELS)}
