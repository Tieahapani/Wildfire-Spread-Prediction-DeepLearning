"""Label values shared by data loading, losses and metrics.

NDWS fire masks use these values for both the day-t (`PrevFireMask`) and the
day-t+1 (`FireMask`) masks (Huot et al. 2022).
"""

FIRE = 1
NO_FIRE = 0
# Cloud cover or an uncertain MODIS detection: the true state is unknown, so these
# pixels are excluded from every loss and metric rather than counted as "no fire".
IGNORE_INDEX = -1
