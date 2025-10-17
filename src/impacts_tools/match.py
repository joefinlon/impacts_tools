"""
Classes for matched IMPACTS ER-2 radar instances

Code by Joe Finlon, Univ. Washington, 2022
Follows methodology in Finlon et al. 2022 [doi: 10.1175/JAS-D-21-0311.1]
"""
import xarray as xr
import numpy as np
from pyproj import Proj
from scipy.stats import iqr
from scipy.spatial import cKDTree
from scipy.ndimage import generic_filter
from abc import ABC, abstractmethod
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom')

class Match(ABC):
    """
    A class to represent matched radar instances during the IMPACTS field campaign.
    
    Match is an Abstract Base Class - meaning we always require a more specific class 
    to be instantiated - ie you have to call Tamms() or Psd(), you can't just call Instrument()
    Parameters
    ----------
    data : xarray.Dataset()
        Matched radar data and attributes
    """
    @abstractmethod     # this stops you from being able to make a new generic instrument
    def __init__(self):
        """
        This is an abstract method since only inherited classes will be used to instantiate Match objects.
        """
        self.name = None
        self.data = None
        
    def qc_radar(self, radar_dataset, alt_bounds_p3, include_ldr=False, qc=False):
        """
        xarray.Dataset (1-D) of QC'd radar data, with bad values removed
        
        Parameters
        ----------
        radar_dataset: impacts_tools.er2.XXX(Radar).Dataset object
        alt_bounds_p3: tuple
            Minimum and maximum P-3 altitude (m) for flight segment.
        include_ldr: bool
            Optionally include LDR pixels in masking criteria
        qc: bool
            Additionally applies a filter on suspected P-3 skinpaints
        """
        if (self.name == 'Matched CRS') or (self.name == 'Matched EXRAD'):
            # mask based on alt relative to P-3 (+/- 250 m) and ground
            ds_qc = radar_dataset.copy()
            mask = np.ones(ds_qc.height.shape, dtype=bool)
            if (self.name == 'Matched CRS') and include_ldr:
                mask[
                    (ds_qc['height'].values >= alt_bounds_p3[0] - 250.) &
                    (ds_qc['height'].values <= alt_bounds_p3[1] + 250.) &
                    (~np.isnan(ds_qc['dbz'].values)) &
                    (~np.isnan(ds_qc['vel'].values)) &
                    (~np.isnan(ds_qc['ldr'].values)) &
                    (ds_qc['height'].values >= 500.)] = False
                mask = xr.DataArray(
                    data = mask,
                    dims = ['range', 'time'],
                    coords = dict(
                        range = ds_qc.range,
                        time = ds_qc.time
                    )
                )
            else:
                mask[
                    (ds_qc['height'].values >= alt_bounds_p3[0] - 250.) &
                    (ds_qc['height'].values <= alt_bounds_p3[1] + 250.) &
                    (~np.isnan(ds_qc['dbz'].values)) &
                    (~np.isnan(ds_qc['vel'].values)) &
                    (ds_qc['height'].values >= 500.)] = False
                mask = xr.DataArray(
                    data = mask,
                    dims = ['range', 'time'],
                    coords = dict(
                        range = ds_qc.range,
                        time = ds_qc.time
                    )
                )
            ds_qc = ds_qc.where(~mask) # set nan outside of these alts
            
            if qc: # additional qc based on dbz variability and spectrum width
                # dbz stdev using rolling window
                # window size reflects similar horiz (300 m) and vert (270 m)
                dbz_std = radar_dataset.dbz.rolling(
                    range=9, time=3, center=True
                ).std(skipna=True)

                if (self.name != 'Matched CRS') and (ds_qc.data.attrs['Date'][:4] != '2023'):
                    # compute top percentile of spectrum width
                    spw_p01 = np.nanpercentile(ds_qc.width.values, 1)

                    # additional masking based on these thresholds
                    ds_qc = ds_qc.where(
                        (dbz_std <= 5.) & (ds_qc.width > spw_p01)
                    )
                else: # no spectrum width in 2023 CRS data, only qc by dbz
                    ds_qc = ds_qc.where(dbz_std <= 5.)
        elif self.name == 'Matched HIWRAP':
            # mask based on alt relative to P-3 (+/- 250 m) and ground
            ds_qc = radar_dataset.copy()
            mask = np.ones(ds_qc.height.shape, dtype=bool)
            if ds_qc.attrs['Date'] == '20230125': # special case (no Ka vel)
                mask[
                    (ds_qc['height'].values >= alt_bounds_p3[0] - 250.) &
                    (ds_qc['height'].values <= alt_bounds_p3[1] + 250.) &
                    (~np.isnan(ds_qc['dbz_ku'].values)) &
                    (~np.isnan(ds_qc['dbz_ka'].values)) &
                    (~np.isnan(ds_qc['vel_ku'].values)) &
                    (ds_qc['height'].values >= 500.)] = False
            else:
                mask[
                    (ds_qc['height'].values >= alt_bounds_p3[0] - 250.) &
                    (ds_qc['height'].values <= alt_bounds_p3[1] + 250.) &
                    (~np.isnan(ds_qc['dbz_ku'].values)) &
                    (~np.isnan(ds_qc['dbz_ka'].values)) &
                    (~np.isnan(ds_qc['vel_ku'].values)) &
                    (~np.isnan(ds_qc['vel_ka'].values)) &
                    (ds_qc['height'].values >= 500.)] = False
            mask = xr.DataArray(
                data = mask,
                dims = ['range', 'time'],
                coords = dict(
                    range = ds_qc.range,
                    time = ds_qc.time
                )
            )
            ds_qc = ds_qc.where(~mask) # set nan outside of these alts
            
            if qc: # additional qc based on dbz variability and spectrum width
                # dbz stdev using rolling window
                # window size reflects similar horiz (300 m) and vert (270 m)
                dbz_ku_std = radar_dataset.dbz_ku.rolling(
                    range=9, time=3, center=True
                ).std(skipna=True)
                dbz_ka_std = radar_dataset.dbz_ka.rolling(
                    range=9, time=3, center=True
                ).std(skipna=True)

                # compute top percentile of spectrum width
                spw_ku_p01 = np.nanpercentile(ds_qc.width_ku.values, 1)
                if ds_qc.attrs['Date'] == '20230125':
                    spw_ka_p01 = 0. # dummy val to skip qc based on spec width
                else:
                    spw_ka_p01 = np.nanpercentile(ds_qc.width_ka.values, 1)

                # additional masking based on these thresholds (run twice)
                ds_qc = ds_qc.where(
                    (dbz_ku_std <= 5.) & (ds_qc.width_ku > spw_ku_p01)
                )
                ds_qc = ds_qc.where(
                    (dbz_ka_std <= 5.) & (ds_qc.width_ka > spw_ka_p01)
                )
        
        # build 1D dataset with select radar vars
        if self.name == 'Matched HIWRAP':
            mask = (
                np.isnan(ds_qc.dbz_ku.values) + np.isnan(ds_qc.dbz_ka.values)
            )
        else:
            mask = np.isnan(ds_qc.dbz.values)
        gate_idx = np.arange(mask.shape[0] * mask.shape[1])
        mask_flat = xr.DataArray(
            data = mask.flatten(),
            dims = 'gate_idx'
        )
        time_flat = xr.DataArray(
            data = np.tile(
                np.atleast_2d(radar_dataset['time'].values),
                (radar_dataset.dims['range'], 1)).flatten(),
            dims = 'gate_idx',
            attrs = radar_dataset['time'].attrs
        )
        hght = xr.DataArray(
            data =  radar_dataset['height'].values.flatten(),
            dims = 'gate_idx',
            attrs = radar_dataset['lat'].attrs
        )
        dist_raw = (
            xr.DataArray(np.ones(radar_dataset.dims['range']), dims=('range')) *
            radar_dataset['distance']).values.flatten()
        dist = xr.DataArray(
            data =  dist_raw - np.min(dist_raw),
            dims = 'gate_idx',
            attrs = radar_dataset['distance'].attrs
        )
        lat = xr.DataArray(
            data =  (
                xr.DataArray(np.ones(radar_dataset.dims['range']), dims=('range')) *
                radar_dataset['lat']).values.flatten(),
            dims = 'gate_idx',
            attrs = radar_dataset['lat'].attrs
        )
        lon = xr.DataArray(
            data =  (
                xr.DataArray(np.ones(radar_dataset.dims['range']), dims=('range')) *
                radar_dataset['lon']).values.flatten(),
            dims = 'gate_idx',
            attrs = radar_dataset['lon'].attrs
        )
        if (self.name == 'Matched CRS') or (self.name == 'Matched EXRAD'):
            dbz = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(radar_dataset.dims['range']), dims=('range')) *
                    radar_dataset['dbz']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_dataset['dbz'].attrs
            )
            vel = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(radar_dataset.dims['range']), dims=('range')) *
                    radar_dataset['vel']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_dataset['vel'].attrs
            )
            if (self.name == 'Matched CRS') and include_ldr:
                ldr = xr.DataArray(
                    data =  (
                        xr.DataArray(np.ones(radar_dataset.dims['range']), dims=('range')) *
                        radar_dataset['ldr']).values.flatten(),
                    dims = 'gate_idx',
                    coords = dict(
                        time = time_flat, height = hght, distance = dist,
                        lat = lat, lon = lon),
                    attrs = radar_dataset['ldr'].attrs
                )
                data_vars = {'dbz': dbz, 'vel': vel, 'ldr': ldr}
            else:
                data_vars = {'dbz': dbz, 'vel': vel}
        elif self.name == 'Matched HIWRAP':
            dbz_ka = xr.DataArray(
                data =  radar_dataset['dbz_ka'].values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_dataset['dbz_ka'].attrs
            )
            dbz_ku = xr.DataArray(
                data =  radar_dataset['dbz_ku'].values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_dataset['dbz_ku'].attrs
            )
            vel_ka = xr.DataArray(
                data =  radar_dataset['vel_ka'].values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_dataset['vel_ka'].attrs
            )
            vel_ku = xr.DataArray(
                data =  radar_dataset['vel_ku'].values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_dataset['vel_ku'].attrs
            )
            data_vars = {
                'dbz_ka': dbz_ka, 'dbz_ku': dbz_ku, 'vel_ka': vel_ka, 'vel_ku': vel_ku
            }
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'time': time_flat,
                'height': hght,
                'distance': dist,
                'lat': lat,
                'lon': lon
            }
        )

        # trim dataset based on flattened boolean flag
        ds = ds.where(~mask_flat, drop=True)
        return ds
    
    def qc_lidar(self, lidar_dataset, alt_bounds_p3, qc=False):
        """
        xarray.Dataset (1-D) of QC'd lidar data, with bad values removed
        
        Parameters
        ----------
        lidar_dataset: impacts_tools.er2.Cpl(Lidar).Dataset object
        alt_bounds_p3: tuple
            Minimum and maximum P-3 altitude (m) for flight segment.
        qc: bool
            Additionally applies a 0.5 sigma filter on L1B data
        """
        mask = np.ones(lidar_dataset[list(lidar_dataset.data_vars)[0]].shape, dtype=bool)
        
        # mask pixels > 50 m below (above) min (max) P-3 altitude
        if self.name == 'Matched CPL ATB': # L1B backscatter data
            var_qc = {}
            if qc: # adapted despeckle routine from er2 module
                for var in ['atb_1064', 'atb_532', 'atb_355']:
                    temp_datacopy =  lidar_dataset[var].copy()
                    temp_datafiltered = gaussian_filter(
                        temp_datacopy, 0.5
                    ) # run the data array through a gaussian filter
                    lidar_dataset[var] = lidar_dataset[var].where(
                        np.isfinite(temp_datafiltered)
                    )
            mask[
                (lidar_dataset['height'].values >= alt_bounds_p3[0] - 250.) &
                (lidar_dataset['height'].values <= alt_bounds_p3[1] + 250.) &
                (lidar_dataset['atb_1064'].values >= 1.e-3) &
                (lidar_dataset['atb_532'].values >= 1.e-3) &
                (lidar_dataset['atb_355'].values >= 1.e-3) &
                (lidar_dataset['height'].values >= 500.)] = False
        elif self.name == 'Matched CPL Profiles': # L2 profile data
            mask[
                (lidar_dataset['height'].values >= alt_bounds_p3[0] - 250.) &
                (lidar_dataset['height'].values <= alt_bounds_p3[1] + 250.) &
                (~np.isnan(lidar_dataset['pbsc_1064'].values)) &
                (~np.isnan(lidar_dataset['pbsc_532'].values)) &
                (~np.isnan(lidar_dataset['pbsc_355'].values)) &
                (~np.isnan(lidar_dataset['ext_1064'].values)) &
                (~np.isnan(lidar_dataset['ext_532'].values)) &
                (~np.isnan(lidar_dataset['ext_355'].values)) &
                (~np.isnan(lidar_dataset['dpol_1064'].values)) &
                (lidar_dataset['height'].values >= 500.)] = False
        
        # build 1D dataset with select lidar vars
        gate_idx = np.arange(mask.shape[0] * mask.shape[1])
        mask_flat = xr.DataArray(
            data = mask.flatten(),
            dims = 'gate_idx'
        )
        time_flat = xr.DataArray(
            data = np.tile(
                np.atleast_2d(lidar_dataset['time'].values),
                (lidar_dataset.dims['gate'], 1)).flatten(),
            dims = 'gate_idx',
            attrs = lidar_dataset['time'].attrs
        )
        hght = xr.DataArray(
            data =  lidar_dataset['height'].values.flatten(),
            dims = 'gate_idx',
            attrs = lidar_dataset['lat'].attrs
        )
        dist_raw = (
            xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
            lidar_dataset['distance']).values.flatten()
        dist = xr.DataArray(
            data =  dist_raw - np.min(dist_raw),
            dims = 'gate_idx',
            attrs = lidar_dataset['distance'].attrs
        )
        lat = xr.DataArray(
            data =  (
                xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
               lidar_dataset['lat']).values.flatten(),
            dims = 'gate_idx',
            attrs = lidar_dataset['lat'].attrs
        )
        lon = xr.DataArray(
            data =  (
                xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
                lidar_dataset['lon']).values.flatten(),
            dims = 'gate_idx',
            attrs = lidar_dataset['lon'].attrs
        )
        if self.name == 'Matched CPL ATB':
            atb1064 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
                   lidar_dataset['atb_1064']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = lidar_dataset['atb_1064'].attrs
            )
            atb532 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
                    lidar_dataset['atb_532']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs =lidar_dataset['atb_532'].attrs
            )
            atb355 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
                    lidar_dataset['atb_355']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = lidar_dataset['atb_355'].attrs
            )
            data_vars = {'atb_1064': atb1064, 'atb_532': atb532, 'atb_355': atb355}
        elif self.name == 'Matched CPL Profiles':
            pbsc1064 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
                   lidar_dataset['pbsc_1064']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = lidar_dataset['pbsc_1064'].attrs
            )
            pbsc532 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
                    lidar_dataset['pbsc_532']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs =lidar_dataset['pbsc_532'].attrs
            )
            pbsc355 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
                    lidar_dataset['pbsc_355']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = lidar_dataset['pbsc_355'].attrs
            )
            cr1 = xr.DataArray(
                data =  pbsc1064.values / pbsc532.values,
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = dict(
                	description = 'Color ratio between 1064 and 532 nm',
                	units = '#'
                )
            )
            cr2 = xr.DataArray(
                data =  pbsc1064.values / pbsc355.values,
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = dict(
                	description = 'Color ratio between 1064 and 355 nm',
                	units = '#'
                )
            )
            cr3 = xr.DataArray(
                data =  pbsc532.values / pbsc355.values,
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = dict(
                	description = 'Color ratio between 532 and 355 nm',
                	units = '#'
                )
            )
            dpol1064 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
                   lidar_dataset['dpol_1064']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = lidar_dataset['dpol_1064'].attrs
            )
            ext1064 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
                   lidar_dataset['ext_1064']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = lidar_dataset['ext_1064'].attrs
            )
            ext532 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
                    lidar_dataset['ext_532']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs =lidar_dataset['ext_532'].attrs
            )
            ext355 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_dataset.dims['gate']), dims=('gate')) *
                    lidar_dataset['ext_355']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = lidar_dataset['ext_355'].attrs
            )
            data_vars = {
                'pbsc_1064': pbsc1064, 'pbsc_532': pbsc532, 'pbsc_355': pbsc355,
                'cr_1064_532': cr1, 'cr_1064_355': cr2, 'cr_532_355': cr3,
                'dpol_1064': dpol1064, 'ext_1064': ext1064, 'ext_532': ext532,
                'ext_355': ext355
            }
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'time': time_flat,
                'height': hght,
                'distance': dist,
                'lat': lat,
                'lon': lon
            }
        )

        # trim dataset based on flattened boolean flag
        ds = ds.where(~mask_flat, drop=True)
        
        return ds
    
    def match_lidar(
            self, lidar_qc, p3_object, query_k, dist_thresh, time_thresh,
            ref_coords, n_workers):
        # define map proj to calculate cartesian distances
        if ref_coords is None:
            ref_coords = (p3_object['lat'].values[0], p3_object['lon'].values[0])
        p = Proj(
            proj='laea', zone=10, ellps='WGS84',
            lat_0=ref_coords[0], lon_0=ref_coords[1]
        )
        
        # use proj to get cartiesian distances between the datasets
        lid_x, lid_y = p(lidar_qc['lon'].values, lidar_qc['lat'].values)
        p3_x, p3_y = p(p3_object['lon'].values, p3_object['lat'].values)
        
        if len(lid_x) == 0: # no available lidar data, skip rest of routine
            return None
        # perform the kdtree search
        kdt = cKDTree(
            list(zip(lid_x, lid_y, lidar_qc['height'].values)), leafsize=16
        )
        prdistance, prind1d = kdt.query(
            list(zip(p3_x, p3_y, p3_object['alt_gps'].values)),
            k=query_k, eps=0, p=2,
            distance_upper_bound=dist_thresh, workers=n_workers
        )
        
        # remove matched data outside of defined bounds
        bad_inds = np.where(prind1d == lidar_qc[list(lidar_qc.data_vars)[0]].shape)
        if ((query_k == 1) and (len(bad_inds[0]) > 0)) or (
                (query_k > 1) and (len(bad_inds[0]) > 0)) or (
                (query_k > 1) and (len(bad_inds[1]) > 0)):
            # mask inds and distances that are outside the search area
            prind1d[bad_inds] = np.ma.masked
            prdistance[bad_inds] = np.ma.masked
            
        # mask data outside distance bounds
        if self.name == 'Matched CPL ATB':
            match_dict = {}
            for wavelength in [1064, 532, 355]:
                match_dict[f'atb_{wavelength}'] = dict(
                    data = np.ma.masked_where(
                        prind1d == 0, lidar_qc[f'atb_{wavelength}'].values[prind1d]
                    ),
                    description = (
                        'Mean attenuated total backscatter profile at '
                        f'{wavelength} nm among matched lidar gates'
                    ),
                    units = 'km**-1 sr**-1'
                )
        elif self.name == 'Matched CPL Profiles':
            match_dict = {}
            match_dict['dpol_1064'] = dict(
                data = np.ma.masked_where(
                        prind1d == 0, lidar_qc['dpol_1064'].values[prind1d]
                    ),
                    description = (
                        'Mean depolarization ratio profile at 1064 nm among '
                        'matched lidar gates'
                    ),
                    units = '#'
            )
            for wavelength_pair in ['1064_532', '1064_355', '532_355']:
            	match_dict[f'cr_{wavelength_pair}'] = dict(
                    data = np.ma.masked_where(
                        prind1d == 0, lidar_qc[f'cr_{wavelength_pair}'].values[prind1d]
                    ),
                    description = (
                        f'Mean color ratio between {wavelength_pair.replace("_", " and ")} '
                        'nm among matched lidar gates'
                    ),
                    units = '#'
                )
            for wavelength in [1064, 532, 355]:
                match_dict[f'ext_{wavelength}'] = dict(
                    data = np.ma.masked_where(
                        prind1d == 0, lidar_qc[f'ext_{wavelength}'].values[prind1d]
                    ),
                    description = (
                        'Mean extinction coefficient profile at '
                        f'{wavelength} nm among matched lidar gates'
                    ),
                    units = 'km**-1'
                )
                match_dict[f'pbsc_{wavelength}'] = dict(
                    data = np.ma.masked_where(
                        prind1d == 0, lidar_qc[f'pbsc_{wavelength}'].values[prind1d]
                    ),
                    description = (
                        'Mean particulate backscatter coefficient profile at '
                        f'{wavelength} nm among matched lidar gates'
                    ),
                    units = 'km**-1 sr**-1'
                )
            
        # perform the matching routine
        if query_k == 1: # nearest neighbor
            return
        else: # more than closest gate considered, use Barnes-weighted mean
            W_d_k = np.ma.array(
                np.exp(-1 * prdistance**2. / dist_thresh**2.)
            ) # distance weights
            
            # use Barnes weighting based on gate distance to P-3
            if self.name == 'Matched CPL ATB': # L1B ATB data
                for wavelength in [1064, 532, 355]:
                    W_d_k2 = np.ma.masked_where(
                        np.ma.getmask(match_dict[f'atb_{wavelength}']['data']),
                        W_d_k.copy()
                    ) # mask weights where ATB is masked
                    w1 = np.ma.sum(
                        W_d_k2 * match_dict[f'atb_{wavelength}']['data'], axis=1
                    ) # weighted sum of ATB per N-s period
                    w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights per N-s period
                    match_dict[f'atb_{wavelength}']['data'] = w1 / w2 # flatten matched, weighted ATB
                    '''
                    dbz_stdev = np.ma.zeros(dbz_matched.shape[0])
                    for i in range(dbz_matched_temp.shape[0]):
                        square_diff = (
                            dbz_matched_temp[i, :] - dbz_matched[i]
                        )**2. # diff**2 between gates and weighted mean
                        ssd = np.nansum(square_diff) # sum of squared diff between gates and weighted mean
                        if np.isnan(ssd):
                            dbz_stdev[i] = np.nan
                        else:
                            num_goodvals = len(dbz_matched_temp[i, :]) - np.sum(np.isnan(square_diff))
                            dbz_stdev[i] = np.sqrt(ssd / num_goodvals)
                    dbz_stdev = np.ma.masked_invalid(dbz_stdev)
                    dbz_matched = np.ma.masked_where(
                        dbz_stdev > 5., dbz_matched).filled(np.nan) # mask suspected skin paint artifact
                    '''
            elif self.name == 'Matched CPL Profiles': # L2 profile data
                for var in [
                        'pbsc_1064', 'pbsc_532', 'pbsc_355', 'cr_1064_532',
                        'cr_1064_355', 'cr_532_355', 'dpol_1064',
                        'ext_1064', 'ext_532', 'ext_355']:
                    W_d_k2 = np.ma.masked_where(
                        np.ma.getmask(match_dict[var]['data']),
                        W_d_k.copy()
                    ) # mask weights where data is masked
                    w1 = np.ma.sum(
                        W_d_k2 * match_dict[var]['data'], axis=1
                    ) # weighted sum of values per N-s period
                    w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights per N-s period
                    match_dict[var]['data'] = w1 / w2 # flatten matched, weighted data
            # time
            time_2d = np.tile(
                np.atleast_2d(p3_object['time'].values).T, (1, query_k)
            )
            tdiff = (
                lidar_qc['time'].values[prind1d] - time_2d
            ) / np.timedelta64(1, 's') # [s]
            W_d_k2 = np.ma.masked_where(np.ma.getmask(tdiff), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * tdiff, axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            tdiff_matched = w1 / w2
            time_lidar_matched = np.array([], dtype='datetime64[ns]')
            for i in range(len(tdiff_matched)):
                time_lidar_matched = np.append(
                    time_lidar_matched,
                    p3_object['time'].values[i] + np.timedelta64(int(tdiff_matched[i]), 's')
                )
            
            # distance difference
            W_d_k2 = np.ma.masked_where(np.ma.getmask(prdistance), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * prdistance, axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            dist_matched = w1 / w2
            
            # altitude
            W_d_k2 = np.ma.masked_where(
                np.ma.getmask(lidar_qc['height'].values[prind1d]),
                W_d_k.copy()
            )
            w1 = np.ma.sum(W_d_k2 * lidar_qc['height'].values[prind1d], axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            alt_lidar_matched = w1 / w2
            
        # along track distance
        if 'distance' in lidar_qc.coords:
            nomdist = np.zeros(len(p3_object['time']))
            for i in range(len(p3_object['time'])):
                if ~np.isnan(time_lidar_matched[i]):
                    nomdist[i] = lidar_qc['distance'].values[
                        np.argmin(np.abs(lidar_qc['time'].values - time_lidar_matched[i]))
                    ]
        
        # mask data
        # time difference between points
        mask_timedelta = np.zeros(len(time_lidar_matched), dtype=bool)
        time_lastgood = time_lidar_matched[0]
        for idx in range(1, len(time_lidar_matched)):
            if np.isnat(time_lidar_matched[idx]): # mask if NaT
                mask_timedelta[idx] = True
            elif time_lidar_matched[idx] - time_lastgood < np.timedelta64(200, 'ms'):
                mask_timedelta[idx] = True
            else:
                time_lastgood = time_lidar_matched[idx]
        # mask_timedelta = np.append(
        #     np.array([False]),
        #     np.array(np.diff(time_lidar_matched) < np.timedelta64(500, 'ms'), dtype=bool)
        # ) # matched lidar time doesn't change when it should always increase
        mask_altdiff = np.abs(
            alt_lidar_matched.data - p3_object['alt_gps'].values
        ) > 250. # mean gate alt > 250 m from P-3 alt
        mask_tdiff = np.abs(tdiff_matched) > time_thresh # time offset too big
        if self.name == 'Matched CPL ATB':
            mask_final = (
                np.isnan(match_dict['atb_1064']['data']) +
                np.isnan(match_dict['atb_532']['data']) +
                np.isnan(match_dict['atb_355']['data']) +
                mask_tdiff.data + mask_altdiff + mask_timedelta
            )
        elif self.name == 'Matched CPL Profiles':
            mask_final = (
                np.isnan(match_dict['pbsc_1064']['data']) +
                np.isnan(match_dict['pbsc_532']['data']) +
                np.isnan(match_dict['pbsc_355']['data']) +
                np.isnan(match_dict['dpol_1064']['data']) +
                np.isnan(match_dict['ext_1064']['data']) +
                np.isnan(match_dict['ext_532']['data']) +
                np.isnan(match_dict['ext_355']['data']) +
                mask_tdiff.data + mask_altdiff + mask_timedelta
            )
            
        # establish the data arrays
        time = p3_object['time'].values

        time_lidar = xr.DataArray(
            data = np.ma.masked_where(mask_final, time_lidar_matched),
            dims = 'time',
            attrs = dict(
                description = 'Mean time of the matched lidar gates',
            )
        )
        dist_lidar = xr.DataArray(
            data = np.ma.masked_where(mask_final, nomdist),
            dims = 'time',
            attrs = dict(
                description = ('Along-track distance corresponding to each '
                               'matched observation'),
                units = 'm'
            )
        )
        alt_lidar = xr.DataArray(
            data = np.ma.masked_where(mask_final, alt_lidar_matched),
            dims = 'time',
            attrs = dict(
                description = 'Mean altitdue of the matched lidar gates ',
                units = 'm'
            )
        )
        data_vars = {}
        data_vars['dist_offset'] = xr.DataArray(
            data = np.ma.masked_where(mask_final, dist_matched),
            dims = 'time',
            coords = dict(time = time, time_lidar = time_lidar),
            attrs = dict(
                description = 'Mean distance between matched lidar gates and P-3',
                units = 'm'
            )
        )
        data_vars['time_offset'] = xr.DataArray(
            data = np.ma.masked_where(mask_final, tdiff_matched),
            dims = 'time',
            coords = dict(time = time, time_lidar = time_lidar),
            attrs = dict(
                description = 'Mean time offset between matched lidar gates and P-3',
                units = 's'
            )
        )
        if self.name == 'Matched CPL ATB':
            for wavelength in [1064, 532, 355]:
                data_vars[f'atb_{wavelength}'] = xr.DataArray(
                    data = np.ma.masked_where(mask_final, match_dict[f'atb_{wavelength}']['data']),
                    dims = 'time',
                    coords = dict(time = time, time_lidar = time_lidar),
                    attrs = dict(
                        description = match_dict[f'atb_{wavelength}']['description'],
                        units = match_dict[f'atb_{wavelength}']['units']
                    )
                )
        elif self.name == 'Matched CPL Profiles':
            for var in [
                    'pbsc_1064', 'pbsc_532', 'pbsc_355', 'cr_1064_532',
                    'cr_1064_355', 'cr_532_355', 'dpol_1064',
                    'ext_1064', 'ext_532', 'ext_355']:
                data_vars[var] = xr.DataArray(
                    data = np.ma.masked_where(mask_final, match_dict[var]['data']),
                    dims = 'time',
                    coords = dict(time = time, time_lidar = time_lidar),
                    attrs = dict(
                        description = match_dict[var]['description'],
                        units = match_dict[var]['units']
                    )
                )
        
        # create dataset
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'time': time,
                'time_lidar': time_lidar,
                'distance': dist_lidar,
                'altitude': alt_lidar
            },
            attrs = {
                'distance_max': f'{dist_thresh:.0f} m',
                'time_diff_max': f'{time_thresh:.0f} s',
                'n_neighbors': f'{query_k}'
            }
        )
        
        return ds
    
    def match_cloudtop_lidar(self, lidar_object):
        """
        Add cloud top properties to the matched CPL object by interpolating the
        cloud top properties to the nearest matched time.
        """
        # get var names related to cloud top products
        vars_top = []
        for var in lidar_object.data_vars:
            if 'top' in var:
                vars_top.append(var)
        
        # interpolate cloud top products to matched object
        data_cloudtop = lidar_object[vars_top].interp(
            time=self.data.time_lidar, method='linear'
        )
        
        # add products to matched object
        ds_merged = xr.merge(
            [self.data, data_cloudtop], compat='override', combine_attrs='drop_conflicts'
        )
        
        return ds_merged
    
    def match_radar(
            self, radar_qc, p3_object, query_k, dist_thresh, time_thresh,
            ref_coords, n_workers):
        # first check if QC'd data has LDR variable (CRS; include_ldr=True)
        contains_ldr = False
        if 'ldr' in list(radar_qc.data_vars):
            contains_ldr = True
        # define map proj to calculate cartesian distances
        if ref_coords is None:
            ref_coords = (p3_object['lat'].values[0], p3_object['lon'].values[0])
        p = Proj(
            proj='laea', zone=10, ellps='WGS84',
            lat_0=ref_coords[0], lon_0=ref_coords[1]
        )
        
        # use proj to get cartiesian distances between the datasets
        rad_x, rad_y = p(radar_qc['lon'].values, radar_qc['lat'].values)
        p3_x, p3_y = p(p3_object['lon'].values, p3_object['lat'].values)
        
        # perform the kdtree search
        kdt = cKDTree(
            list(zip(rad_x, rad_y, radar_qc['height'].values)), leafsize=16
        )
        prdistance, prind1d = kdt.query(
            list(zip(p3_x, p3_y, p3_object['alt_gps'].values)),
            k=query_k, eps=0, p=2,
            distance_upper_bound=dist_thresh, workers=n_workers
        )
        
        # remove matched data outside of defined bounds
        if self.name == 'Matched HIWRAP':
            bad_inds = np.where(prind1d == radar_qc['dbz_ka'].shape)
        else:
            bad_inds = np.where(prind1d == radar_qc['dbz'].shape)
        if ((query_k == 1) and (len(bad_inds[0]) > 0)) or (
                (query_k > 1) and (len(bad_inds[0]) > 0)) or (
                (query_k > 1) and (len(bad_inds[1]) > 0)):
            # mask inds and distances that are outside the search area
            prind1d[bad_inds] = np.ma.masked
            prdistance[bad_inds] = np.ma.masked
            
        # mask dbz outside distance bounds
        if (self.name == 'Matched CRS') or (self.name == 'Matched EXRAD'):
            dbz_matched = np.ma.masked_where(
                prind1d == 0, radar_qc['dbz'].values[prind1d]
            )
            vel_matched = np.ma.masked_where(
                prind1d == 0, radar_qc['vel'].values[prind1d]
            )
            if contains_ldr:
                ldr_matched = np.ma.masked_where(
                    prind1d == 0, radar_qc['ldr'].values[prind1d]
                )
            key_dbz, key_vel = ('dbz', 'vel')
        elif self.name == 'Matched HIWRAP':
            dbz_matched = np.ma.masked_where(
                prind1d == 0, radar_qc['dbz_ka'].values[prind1d]
            )
            dbz2_matched = np.ma.masked_where(
                prind1d == 0, radar_qc['dbz_ku'].values[prind1d]
            )
            vel_matched = np.ma.masked_where(
                prind1d == 0, radar_qc['vel_ka'].values[prind1d]
            )
            vel2_matched = np.ma.masked_where(
                prind1d == 0, radar_qc['vel_ku'].values[prind1d]
            )
            key_dbz, key_vel = ('dbz_ka', 'vel_ka')
        
        # Perform the matching routine
        if query_k == 1: # nearest neighbor
            print('Nearest neighbor not supported at this time')
            return
        else: # more than closest gate considered, use Barnes-weighted mean
            # dbz
            W_d_k = np.ma.array(
                np.exp(-1 * prdistance**2. / dist_thresh**2.)
            ) # distance weights
            W_d_k2 = np.ma.masked_where(
                np.ma.getmask(dbz_matched), W_d_k.copy()
            ) # mask weights where dbz is masked
            w1 = np.ma.sum(
                W_d_k2 * 10.**(dbz_matched / 10.), axis=1
            ) # weighted sum of linear Z per N-s period
            w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights per N-s period
            dbz_matched_temp = dbz_matched.copy()
            dbz_matched = 10. * np.ma.log10(w1 / w2) # flatten matched, weighted dbz
            dbz_stdev = np.ma.zeros(dbz_matched.shape[0])
            for i in range(dbz_matched_temp.shape[0]):
                square_diff = (
                    dbz_matched_temp[i, :] - dbz_matched[i]
                )**2. # diff**2 between gates and weighted mean
                ssd = np.nansum(square_diff) # sum of squared diff between gates and weighted mean
                if np.isnan(ssd):
                    dbz_stdev[i] = np.nan
                else:
                    num_goodvals = len(dbz_matched_temp[i, :]) - np.sum(np.isnan(square_diff))
                    dbz_stdev[i] = np.sqrt(ssd / num_goodvals)
            dbz_stdev = np.ma.masked_invalid(dbz_stdev)
            dbz_matched = np.ma.masked_where(
                dbz_stdev > 5., dbz_matched).filled(np.nan) # mask suspected skin paint artifact
            
            # vel
            W_d_k2 = np.ma.masked_where(
                np.ma.getmask(vel_matched), W_d_k.copy()
            ) # mask weights where vel is masked
            w1 = np.ma.sum(W_d_k2 * vel_matched, axis=1) # weighted sum of vel per N-s period
            w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights per N-s period
            vel_matched = w1 / w2 # flatten matched vel

            # ldr, if applicable
            if contains_ldr:
                W_d_k2 = np.ma.masked_where(
                    np.ma.getmask(ldr_matched), W_d_k.copy()
                ) # mask weights where ldr is masked
                w1 = np.ma.sum(
                    W_d_k2 * 10.**(ldr_matched / 10.), axis=1
                ) # weighted sum of ldr (linear units) per N-s period
                w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights per N-s period
                ldr_matched = 10. * np.ma.log10(w1 / w2) # flatten matched, weighted ldr [dB]
            
            # dbz2 and vel2 (HIWRAP Ku)
            if self.name == 'Matched HIWRAP':
                # dbz2
                W_d_k2 = np.ma.masked_where(
                    np.ma.getmask(dbz2_matched), W_d_k.copy()
                ) # mask weights where dbz is masked
                w1 = np.ma.sum(
                    W_d_k2 * 10.**(dbz2_matched / 10.), axis=1
                ) # weighted sum of linear Z per N-s period
                w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights per N-s period
                dbz2_matched_temp = dbz2_matched.copy()
                dbz2_matched = 10. * np.ma.log10(w1 / w2) # flatten matched, weighted dbz
                dbz2_stdev = np.ma.zeros(dbz2_matched.shape[0])
                for i in range(dbz2_matched_temp.shape[0]):
                    square_diff = (
                        dbz2_matched_temp[i, :] - dbz2_matched[i]
                    ) ** 2. # diff**2 between gates and weighted mean
                    ssd = np.nansum(square_diff) # sum of squared diff between gates and weighted mean
                    if np.isnan(ssd):
                        dbz2_stdev[i] = np.nan
                    else:
                        num_goodvals = len(dbz2_matched_temp[i, :]) - np.sum(np.isnan(square_diff))
                        dbz2_stdev[i] = np.sqrt(ssd / num_goodvals)
                dbz2_stdev = np.ma.masked_invalid(dbz2_stdev)
                dbz2_matched = np.ma.masked_where(
                    dbz2_stdev > 5., dbz2_matched).filled(np.nan) # mask suspected skin paint artifact
                
                # vel2
                W_d_k2 = np.ma.masked_where(
                    np.ma.getmask(vel2_matched), W_d_k.copy()
                ) # mask weights where vel is masked
                w1 = np.ma.sum(W_d_k2 * vel2_matched, axis=1) # weighted sum of vel per N-s period
                w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights per N-s period
                vel2_matched = w1 / w2 # flatten matched vel
                
            # time
            time_2d = np.tile(
                np.atleast_2d(p3_object['time'].values).T, (1, query_k)
            )
            tdiff = (
                radar_qc['time'].values[prind1d] - time_2d
            ) / np.timedelta64(1, 's') # [s]
            W_d_k2 = np.ma.masked_where(np.ma.getmask(tdiff), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * tdiff, axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            tdiff_matched = w1 / w2
            time_radar_matched = np.array([], dtype='datetime64[ns]')
            for i in range(len(tdiff_matched)):
                time_radar_matched = np.append(
                    time_radar_matched,
                    p3_object['time'].values[i] + np.timedelta64(int(tdiff_matched[i]), 's')
                )
            
            # distance difference
            W_d_k2 = np.ma.masked_where(np.ma.getmask(prdistance), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * prdistance, axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            dist_matched = w1 / w2
            
            # altitude
            W_d_k2 = np.ma.masked_where(
                np.ma.getmask(radar_qc['height'].values[prind1d]),
                W_d_k.copy()
            )
            w1 = np.ma.sum(W_d_k2 * radar_qc['height'].values[prind1d], axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            alt_radar_matched = w1 / w2
            
        # along track distance
        nomdist = np.zeros(len(p3_object['time']))
        for i in range(len(p3_object['time'])):
            if ~np.isnan(time_radar_matched[i]):
                nomdist[i] = radar_qc['distance'].values[
                    np.argmin(np.abs(radar_qc['time'].values - time_radar_matched[i]))
                ]
        
        # mask data
        mask_timedelta = np.append(
            np.array([False]),
            np.array(np.diff(time_radar_matched) < np.timedelta64(500, 'ms'), dtype=bool)
        ) # matched radar time doesn't change when it should always increase
        mask_altdiff = np.abs(
            alt_radar_matched.data - p3_object['alt_gps'].values
        ) > 250. # mean gate alt > 250 m from P-3 alt
        mask_tdiff = np.abs(tdiff_matched) > time_thresh # time offset too big
        if self.name == 'Matched HIWRAP':
            mask_final = (
                np.isnan(dbz_matched) + np.isnan(dbz2_matched) +
                mask_tdiff.data + mask_altdiff + mask_timedelta
            )
        else:
            mask_final = (
                np.isnan(dbz_matched) + mask_tdiff.data + mask_altdiff + mask_timedelta
            )
        
        # establish the data arrays
        time = p3_object['time'].values

        time_radar = xr.DataArray(
            data = np.ma.masked_where(mask_final, time_radar_matched),
            dims = 'time',
            attrs = dict(
                description = 'Mean time of the matched radar gates',
            )
        )
        dist_radar = xr.DataArray(
            data = np.ma.masked_where(mask_final, nomdist),
            dims = 'time',
            attrs = dict(
                description = ('Along-track distance corresponding to each '
                               'matched observation'),
                units = 'm'
            )
        )
        ddiff = xr.DataArray(
            data = np.ma.masked_where(mask_final, dist_matched),
            dims = 'time',
            coords = dict(time = time, time_radar = time_radar),
            attrs = dict(
                description = 'Mean distance between matched radar gates and P-3',
                units = 'm'
            )
        )
        tdiff = xr.DataArray(
            data = np.ma.masked_where(mask_final, tdiff_matched),
            dims = 'time',
            coords = dict(time = time, time_radar = time_radar),
            attrs = dict(
                description = 'Mean time offset between matched radar gates and P-3',
                units = 's'
            )
        )
        dbz = xr.DataArray(
            data = np.ma.masked_where(mask_final, dbz_matched),
            dims = 'time',
            coords = dict(time = time, time_radar = time_radar),
            attrs = dict(
                description = 'Mean reflectivity among matched radar gates',
                units = 'dBZ'
            )
        )
        vel = xr.DataArray(
            data = np.ma.masked_where(mask_final, vel_matched),
            dims = 'time',
            coords = dict(time = time, time_radar = time_radar),
            attrs = dict(
                description = 'Mean Doppler velocity among matched radar gates',
                units = 'm s**-1'
            )
        )
        if self.name == 'Matched HIWRAP':
            dbz2 = xr.DataArray(
                data = np.ma.masked_where(mask_final, dbz2_matched),
                dims = 'time',
                coords = dict(time = time, time_radar = time_radar),
                attrs = dict(
                    description = 'Mean reflectivity among matched radar gates',
                    units = 'dBZ'
                )
            )
            vel2 = xr.DataArray(
                data = np.ma.masked_where(mask_final, vel2_matched),
                dims = 'time',
                coords = dict(time = time, time_radar = time_radar),
                attrs = dict(
                    description = 'Mean Doppler velocity among matched radar gates',
                    units = 'm s**-1'
                )
            )
            data_vars = {
                'dist_diff': ddiff,
                'time_diff': tdiff,
                'dbz_ka': dbz,
                'dbz_ku': dbz2,
                'vel_ka': vel,
                'vel_ku': vel2
            }
        else:
            if self.name == 'Matched CRS':
                freq = 'w'
            elif self.name == 'Matched EXRAD':
                freq = 'x'
            if contains_ldr:
                ldr = xr.DataArray(
                    data = np.ma.masked_where(mask_final, ldr_matched),
                    dims = 'time',
                    coords = dict(time = time, time_radar = time_radar),
                    attrs = dict(
                        description = 'Mean linear depolarization ratio among matched radar gates',
                        units = 'dB'
                    )
                )
                data_vars = {
                    'dist_diff': ddiff,
                    'time_diff': tdiff,
                    f'dbz_{freq}': dbz,
                    f'vel_{freq}': vel,
                    f'ldr_{freq}': ldr
                }
            else:
                data_vars = {
                    'dist_diff': ddiff,
                    'time_diff': tdiff,
                    f'dbz_{freq}': dbz,
                    f'vel_{freq}': vel
                }
            
        # create dataset
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'time': time,
                'time_radar': time_radar,
                'distance': dist_radar
            },
            attrs = {
                'distance_max': f'{dist_thresh:.0f} m',
                'time_diff_max': f'{time_thresh:.0f} s',
                'n_neighbors': f'{query_k}'
            }
        )
        
        return ds

class MatchLidar(ABC):
    """
    A class to represent matched lidar instances during the IMPACTS field campaign.
    
    MatchLidar is an Abstract Base Class - meaning we always require a more specific class 
    to be instantiated - ie you have to call Tamms() or Psd(), you can't just call Instrument()
    Parameters
    ----------
    data : xarray.Dataset()
        Matched radar data and attributes
    """
    @abstractmethod     # this stops you from being able to make a new generic instrument
    def __init__(self):
        """
        This is an abstract method since only inherited classes will be used to instantiate Match objects.
        """
        self.name = None
        self.data = None
        
    def qc_radar(self, radar_dataset, qc=False):
        """
        xarray.Dataset (1-D) of QC'd radar data, with bad values removed
        """
        if (self.name == 'Matched CRS') or (self.name == 'Matched EXRAD'):
            # build mask
            mask = np.zeros(radar_dataset['dbz'].shape, dtype=bool)
            mask[
                (np.isnan(radar_dataset['dbz'].values)) |
                (radar_dataset['height'].values < 500.)] = True
            
            if qc: # additional qc based on Z variability and SPW
                # compute IQR of Z using 9x3 (vertical x horizontal) moving window
                # window should reflect similar distance in horizontal and vertical
                iqr_dbz = generic_filter(
                    radar_dataset['dbz'].values, iqr,
                    size=(9,3), mode='nearest'
                )

                # compute 5th percentile of spectrum width
                p05_spw = np.nanpercentile(radar_dataset['width'].values, 5)
                
                # additional conditions for mask
                mask[(iqr_dbz > 5.) & (radar_dataset['width'].values < p05_spw)] = True
        elif self.name == 'Matched HIWRAP':
            # build mask
            mask = np.zeros(radar_dataset['dbz_ku'].shape, dtype=bool)
            mask[
                (np.isnan(radar_dataset['dbz_ku'].values)) |
                (np.isnan(radar_dataset['dbz_ka'].values)) |
                (radar_dataset['height'].values < 500.)] = True
            
            if qc: # additional qc based on Z variability and SPW
                # compute IQR of Z using 9x3 (vertical x horizontal) moving window
                # window should reflect similar distance in horizontal and vertical
                iqr_dbz_ka = generic_filter(
                    radar_dataset['dbz_ka'].values, iqr,
                    size=(9,3), mode='nearest'
                )
                iqr_dbz_ku = generic_filter(
                    radar_dataset['dbz_ku'].values, iqr,
                    size=(9,3), mode='nearest'
                )

                # compute 5th percentile of spectrum width
                p05_spw_ka = np.nanpercentile(radar_dataset['width_ka'].values, 5)
                p05_spw_ku = np.nanpercentile(radar_dataset['width_ku'].values, 5)
                
                # additional conditions for mask
                mask[(iqr_dbz_ka > 5.) & (radar_dataset['width_ka'].values < p05_spw_ka)] = True
                mask[(iqr_dbz_ku > 5.) & (radar_dataset['width_ku'].values < p05_spw_ku)] = True
        
        # build 1D dataset with select radar vars
        gate_idx = np.arange(mask.shape[0] * mask.shape[1])
        mask_flat = xr.DataArray(
            data = mask.flatten(),
            dims = 'gate_idx'
        )
        time_flat = xr.DataArray(
            data = np.tile(
                np.atleast_2d(radar_dataset['time'].values),
                (radar_dataset.dims['range'], 1)).flatten(),
            dims = 'gate_idx',
            attrs = radar_dataset['time'].attrs
        )
        hght = xr.DataArray(
            data =  radar_dataset['height'].values.flatten(),
            dims = 'gate_idx',
            attrs = radar_dataset['lat'].attrs
        )
        dist_raw = (
            xr.DataArray(np.ones(radar_dataset.dims['range']), dims=('range')) *
            radar_dataset['distance']).values.flatten()
        dist = xr.DataArray(
            data =  dist_raw - np.min(dist_raw),
            dims = 'gate_idx',
            attrs = radar_dataset['distance'].attrs
        )
        lat = xr.DataArray(
            data =  (
                xr.DataArray(np.ones(radar_dataset.dims['range']), dims=('range')) *
                radar_dataset['lat']).values.flatten(),
            dims = 'gate_idx',
            attrs = radar_dataset['lat'].attrs
        )
        lon = xr.DataArray(
            data =  (
                xr.DataArray(np.ones(radar_dataset.dims['range']), dims=('range')) *
                radar_dataset['lon']).values.flatten(),
            dims = 'gate_idx',
            attrs = radar_dataset['lon'].attrs
        )
        if (self.name == 'Matched CRS') or (self.name == 'Matched EXRAD'):
            dbz = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(radar_dataset.dims['range']), dims=('range')) *
                    radar_dataset['dbz']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_dataset['dbz'].attrs
            )
            data_vars = {'dbz': dbz}
        elif self.name == 'Matched HIWRAP':
            dbz_ka = xr.DataArray(
                data =  radar_dataset['dbz_ka'].values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_dataset['dbz_ka'].attrs
            )
            dbz_ku = xr.DataArray(
                data =  radar_dataset['dbz_ku'].values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_dataset['dbz_ku'].attrs
            )
            data_vars = {'dbz_ka': dbz_ka, 'dbz_ku': dbz_ku}
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'time': time_flat,
                'height': hght,
                'distance': dist,
                'lat': lat,
                'lon': lon
            }
        )

        # trim dataset based on flattened boolean flag
        ds = ds.where(~mask_flat, drop=True)
        
        return ds

# ====================================== #
# HIWRAP
# ====================================== #
class Hiwrap(Match):
    """
    A class to represent the CPL data matched to the P-3 position during the IMPACTS field campaign.
    Inherits from Instrument()
    
    Parameters
    ----------
    hiwrap_object: impacts_tools.er2.Hiwrap(Radar) object
        The time-trimmed HIWRAP radar object
    p3_object: impacts_tools.p3.P3() object
        The time-trimmed P-3 Met-Nav object
    query_k: int
        Number of radar gates considered in the average (1 - nearest neighbor)
    dist_thresh: float
        Maximum distance (m) allowed in the kdTree search
    time_thresh: None or float
        Maximum time offset (s) between ER-2 and P-3 allowed for matched instance
    qc: bool
        True - remove gates with high dbz gradient, no dbz, low spw
        False - only remove gates with no dbz
    ref_coords: None or 2-element tuple of float
        Reference lat, lon pair (deg) to project ER-2 and P-3 points to cartesian grid.
        None defaults to first lat, lon point in P-3 object
    """
    
    def __init__(
            self, radar_object, p3_object, query_k=30, dist_thresh=4000.,
            time_thresh=None, qc=False, ref_coords=None, n_workers=1):
        self.name = 'Matched HIWRAP'
        
        # get P-3 alt bounds
        alt_minP3 = np.nanmin(p3_object['alt_gps'].values)
        alt_maxP3 = np.nanmax(p3_object['alt_gps'].values)
        alt_boundsP3 = (alt_minP3, alt_maxP3)
        
        # qc radar data (threshold by Z gradient and SW if specified)
        radar_qc = self.qc_radar(radar_object, alt_boundsP3, qc)
        
        # read the raw data
        self.data = self.match_radar(
            radar_qc, p3_object, query_k, dist_thresh, time_thresh,
            ref_coords, n_workers
        )
        """
        xarray.Dataset of matched HIWRAP reflectivity and attributes
        Dimensions:
            - time: np.array(np.datetime64[s]) - The UTC time start of the N-s upsampled interval
        Coordinates:
            - time (time): np.array(np.datetime64[s]) - The UTC time start of the N-s upsampled interval
            - distance (time): xarray.DataArray(float) - Along track ER-2 distance (m)
            - lat (time): xarray.DataArray(float) - Latitude (degrees)
            - lon (time): xarray.DataArray(float) - Longitude (degrees)
        Variables:
            - alt_gps (time) : xarray.DataArray(float) - P-3 GPS altitude (m above mean sea level)
            - temp (time) : xarray.DataArray(float) - Static (ambient) air temperature (deg C)
            - dwpt (time) : xarray.DataArray(float) - Dew point temperature (deg C)
            - z_ku (time) : xarray.DataArray(float) - Matched Ku-band reflectivity (dBZ)
            - z_ka (time) : xarray.DataArray(float) - Matched Ka-band reflectivity (dBZ)
            - dfr_ku_ka (time) : xarray.DataArray(float) - Matched dual frequency ratio (dB)
        """
        
# ====================================== #
# CRS
# ====================================== #
class Crs(Match):
    """
    A class to represent the CRS data matched to the P-3 position during the IMPACTS field campaign.
    Inherits from Instrument()
    
    Parameters
    ----------
    crs_object: impacts_tools.er2.Crs(Radar) object
        The time-trimmed CRS radar object
    p3_object: impacts_tools.p3.P3() object
        The time-trimmed P-3 Met-Nav object
    query_k: int
        Number of radar gates considered in the average (1 - nearest neighbor)
    dist_thresh: float
        Maximum distance (m) allowed in the kdTree search
    time_thresh: None or float
        Maximum time offset (s) between ER-2 and P-3 allowed for matched instance
    include_ldr: bool
        True - include LDR pixels when finding good pixels, and match to P-3
        False - exclude LDR pixels; only match Ze and Velocity
    qc: bool
        True - remove gates with high dbz gradient, no dbz, low spw
        False - only remove gates with no dbz
    ref_coords: None or 2-element tuple of float
        Reference lat, lon pair (deg) to project ER-2 and P-3 points to cartesian grid.
        None defaults to first lat, lon point in P-3 object
    """
    
    def __init__(
            self, radar_object, p3_object, query_k=30, dist_thresh=4000.,
            time_thresh=None, include_ldr=False, qc=False, ref_coords=None, n_workers=1):
        self.name = 'Matched CRS'
        
        # get P-3 alt bounds
        alt_minP3 = np.nanmin(p3_object['alt_gps'].values)
        alt_maxP3 = np.nanmax(p3_object['alt_gps'].values)
        alt_boundsP3 = (alt_minP3, alt_maxP3)
        
        # qc radar data (threshold by Z gradient and SW if specified)
        radar_qc = self.qc_radar(radar_object, alt_boundsP3, include_ldr, qc)
        
        # read the raw data
        self.data = self.match_radar(
            radar_qc, p3_object, query_k, dist_thresh, time_thresh,
            ref_coords, n_workers
        )
        """
        xarray.Dataset of matched HIWRAP reflectivity and attributes
        Dimensions:
            - time: np.array(np.datetime64[s]) - The UTC time start of the N-s upsampled interval
        Coordinates:
            - time (time): np.array(np.datetime64[s]) - The UTC time start of the N-s upsampled interval
            - distance (time): xarray.DataArray(float) - Along track ER-2 distance (m)
            - lat (time): xarray.DataArray(float) - Latitude (degrees)
            - lon (time): xarray.DataArray(float) - Longitude (degrees)
        Variables:
            - alt_gps (time) : xarray.DataArray(float) - P-3 GPS altitude (m above mean sea level)
            - temp (time) : xarray.DataArray(float) - Static (ambient) air temperature (deg C)
            - dwpt (time) : xarray.DataArray(float) - Dew point temperature (deg C)
            - z_ku (time) : xarray.DataArray(float) - Matched Ku-band reflectivity (dBZ)
            - z_ka (time) : xarray.DataArray(float) - Matched Ka-band reflectivity (dBZ)
            - dfr_ku_ka (time) : xarray.DataArray(float) - Matched dual frequency ratio (dB)
        """
        
# ====================================== #
# CPL
# ====================================== #
class Cpl(Match):
    """
    A class to represent the CPL data matched to the P-3 position during the IMPACTS field campaign.
    Inherits from Match()
    
    Parameters
    ----------
    lidar_object: impacts_tools.er2.Cpl(Lidar).data xarray dataset object
        The time-trimmed CPL lidar object
    p3_object: impacts_tools.p3.P3().data xarray dataset object
        The time-trimmed P-3 Met-Nav object
    query_k: int
        Number of radar gates considered in the average (1 - nearest neighbor)
    dist_thresh: float
        Maximum distance (m) allowed in the kdTree search
    time_thresh: None or float
        Maximum time offset (s) between ER-2 and P-3 allowed for matched instance
    qc: bool
        True - remove gates with high dbz gradient, no dbz, low spw
        False - only remove gates with no dbz
    ref_coords: None or 2-element tuple of float
        Reference lat, lon pair (deg) to project ER-2 and P-3 points to cartesian grid.
        None defaults to first lat, lon point in P-3 object
    """
    
    def __init__(
            self, lidar_object, p3_object, query_k=30, dist_thresh=4000.,
            time_thresh=None, qc=False, ref_coords=None, n_workers=1):
        if 'atb_1064' in lidar_object.data_vars: # L1B ATB data
            self.name = 'Matched CPL ATB'
        elif 'dpol_1064' in lidar_object.data_vars: # L2 profile data
            self.name = 'Matched CPL Profiles'
        
        # get P-3 alt bounds
        alt_minP3 = np.nanmin(p3_object['alt_gps'].values)
        alt_maxP3 = np.nanmax(p3_object['alt_gps'].values)
        alt_boundsP3 = (alt_minP3, alt_maxP3)
            
        # qc lidar data
        lidar_qc = self.qc_lidar(lidar_object, alt_boundsP3, qc)

        # match the data
        self.data = self.match_lidar(
            lidar_qc, p3_object, query_k, dist_thresh, time_thresh,
            ref_coords, n_workers
        )
        
        # add cloud top lidar properties to matched object (optional)
        if (self.data is not None) and (
                ('atb_1064_top' in lidar_object.data_vars) or (
                'dpol_1064_top' in lidar_object.data_vars)): # cloud top data exists
            self.data = self.match_cloudtop_lidar(lidar_object)