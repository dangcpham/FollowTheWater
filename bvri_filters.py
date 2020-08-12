import read_files
DEFAULT_FILTER_PATH = 'filters/'

### Import Filters and Atmosphere Transmission ###

u_filter = read_files.filters(DEFAULT_FILTER_PATH,'u_filter.txt', auto_interpolate=True)
u_filter_profile = [u_filter.getRawFilterWavelength(), u_filter.getRawFilterTransmission()]
u_interpolated = u_filter.getInterpolatedFilter()

b_filter = read_files.filters(DEFAULT_FILTER_PATH,'b_filter.txt', auto_interpolate=True)
b_filter_profile = [b_filter.getRawFilterWavelength(), b_filter.getRawFilterTransmission()]
b_interpolated = b_filter.getInterpolatedFilter()

v_filter = read_files.filters(DEFAULT_FILTER_PATH,'v_filter.txt', auto_interpolate=True)
v_filter_profile = [v_filter.getRawFilterWavelength(), v_filter.getRawFilterTransmission()]
v_interpolated = v_filter.getInterpolatedFilter()

r_filter = read_files.filters(DEFAULT_FILTER_PATH,'r_filter.txt', auto_interpolate=True)
r_filter_profile = [r_filter.getRawFilterWavelength(), r_filter.getRawFilterTransmission()]
r_interpolated = r_filter.getInterpolatedFilter()

i_filter = read_files.filters(DEFAULT_FILTER_PATH,'i_filter.txt', auto_interpolate=True)
i_filter_profile = [i_filter.getRawFilterWavelength(), i_filter.getRawFilterTransmission()]
i_interpolated = i_filter.getInterpolatedFilter()

earth_sun_transmission = read_files.atmosphere_transmission('data/earth_sun_transmission.csv')
earth_sun_transmission_data = earth_sun_transmission.getTransmission()
earth_sun_transmission_interp = earth_sun_transmission.getInterpolation()

earth_sun_toa = read_files.atmosphere_transmission('data/earth_flux_toa.csv')
earth_sun_toa_flux = earth_sun_toa.getTransmission()
earth_sun_toa_interp = earth_sun_toa.getInterpolation()

all_filter_profile = [u_filter_profile, b_filter_profile, v_filter_profile, r_filter_profile, i_filter_profile]
all_filter_interp = [u_interpolated, b_interpolated, v_interpolated, r_interpolated, i_interpolated]