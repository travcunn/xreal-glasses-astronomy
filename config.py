"""All user-tunable settings for the sky prototype."""

# Observer location (free-look heading, but location fixes which sky is overhead).
LATITUDE_DEG = 37.7749       # default: San Francisco
LONGITUDE_DEG = -122.4194
ELEVATION_M = 0.0

# Star rendering
MAG_LIMIT = 6.5              # faintest star magnitude to draw (naked-eye limit)
LABEL_MAG_LIMIT = 1.8        # only label stars at least this bright

# Camera / display
FOV_DEG = 57.0               # XREAL One Pro field of view (tunable during calibration)
GLASSES_RESOLUTION = (1920, 1080)
DISPLAY_INDEX = None         # None = auto-detect the 1920x1080 glasses display

# IMU
IMU_HOST = "169.254.2.1"
IMU_PORT = 52998
ACCEL_GAIN = 0.02            # complementary filter: how hard gravity corrects drift

# Magnetometer heading anchor
MAGNETIC_DECLINATION_DEG = 13.0   # fallback if WMM lookup fails (SF ~ +13 E)
HEADING_GAIN = 0.02               # how fast the compass pulls yaw toward true north
IP_GEOLOCATION = True             # auto-detect observer location at startup
MAG_CALIBRATION_PATH = "mag_calibration.json"

# Toggles
SHOW_MILKY_WAY = False
SHOW_HORIZON = True
