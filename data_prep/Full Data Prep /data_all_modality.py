# ================================================================================
# ================================================================================
# ================================================================================
# Step I: Data Collection Process for Multimodal Life Expectancy Prediction. Each of the following sections corresponds to a different data source that we will be using for our multimodal life expectancy prediction project. The code is organized into three main sections, each dedicated to a specific dataset:
# ================================================================================
# ================================================================================
# ================================================================================


# 1 - IHME Life Expectancy
# 2 - Multimodal Satalite Data
# 3 - FAO Livestock


###################################
# 1 - IHME Life Expectancy
###################################

import pandas as pd
import os

folder_path = '/Users/faizahmad/Desktop/00 Shisir/LifeEx Data and CSV files/Life Expectency Data'

# List all the csv files in the folder

csv_files = [x for x in os.listdir(folder_path) if x.endswith('.CSV')] ## this is an example of list comprehension
csv_files

# lets create an empty list to store the dataframes
dataframes = []

# lets loop over the list of CSV files and read each one

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

print(type(dataframes))
print(len(dataframes)) ## A total of 20 dataframes, stored as a list


# Modify the processing to include FIPS code with 5-digit padding

reduced_dataframe = []

for i in range(0, 20):
    
    ## lets extract the total life expectency and only of the age group less than 1 year olds.
    df2 = dataframes[i].loc[(dataframes[i]['race_name'] == 'Total') & (dataframes[i]['age_name'] == '<1 year')]

    ## lets remove empty cells
    df3 = df2.dropna()

    ## As the dataframe consists of life expectencey at the state level as well
    ## lets gather only those with county, since fips for state end at 56, will set the condition to be greater than this to get the data at the county level.
    df4 = df3.loc[(df3['fips'] > 60)]

    ## lets delete these columns.
    df5 = df4.drop(['measure_id', 'location_id', 'measure_name', 'race_id', 'race_name', 'sex_id', 'sex_name', 'age_group_id',
     'age_name', 'metric_id', 'metric_name', 'upper', 'lower'], axis=1)

    ## Convert FIPS to integer then format with 5-digit padding
    df5['fips'] = df5['fips'].astype(int).astype(str).str.zfill(5)

    ## lets rename the columns
    df5 = df5.rename(columns={'val': 'MeanLifeExpectency', 'fips': 'fips'})
    
    ## Reorder columns: location_name, fips, year, MeanLifeExpectency
    df5 = df5[['location_name', 'fips', 'year', 'MeanLifeExpectency']]

    reduced_dataframe.append(df5)


final_df=pd.concat(reduced_dataframe,ignore_index=True)
final_df
final_df.to_csv('/Users/faizahmad/Desktop/LE Crop data/LE_Crop_data.csv', index=False)





###################################
# 2 - Multimodal Satalite Data
###################################

# 2A

#  =====================================================
#  ENHANCED GEE PIPELINE FOR REMOTE SENSING FOCUS
#  Version: Agricultural-Only (No SDOH/Weather)
# =====================================================

from google.colab import drive
drive.mount('/content/drive')

import ee, os, time, json, pandas as pd
from google.colab import drive

try:
    ee.Initialize(project='ee-faiz2009cu')
except:
    ee.Authenticate()
    ee.Initialize(project='ee-faiz2009cu')

# ------------------- CONFIG -------------------
TEST_MODE = False
SCALE = 250
FOLDER = 'Agricultural_RS_LE_2025'
BASE = f'/content/drive/MyDrive/{FOLDER}'
os.makedirs(BASE, exist_ok=True)

if TEST_MODE:
    YEARS = [2020, 2021, 2022]
    STATES = ['06']  # California
else:
    YEARS = range(2000, 2025)
    STATES = None

# ------------------- EXPANDED BAND CATALOG -------------------
BANDS = {
    # Your existing sensors
    'landsat89': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
    's2': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
    's1': ['VV', 'VH'],
    'cropland_usda': ['cropland'],
    'dynamic_world': ['water', 'trees', 'grass', 'flooded_vegetation',
                      'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'],
    'water_jrc': ['waterClass'],
    'dem': ['DEM'],

    # NEW: Additional products
    'modis_ndvi': ['NDVI', 'EVI'],
    'modis_lst': ['LST_Day_1km', 'LST_Night_1km'],
    'soil_texture': ['b0'],  # Surface soil texture
    'nightlights': ['avg_rad'],  # Proxy for economic activity (optional)
}

# ------------------- EXPANDED ASSET MAP -------------------
ASSET_MAP = {
    # Existing
    'landsat89':     {'enabled': False, 'id': 'LANDSAT/LC08/C02/T1_L2', 'temporal': 'median', 'add_ndvi': True},
    's2':            {'enabled': False, 'id': 'COPERNICUS/S2_SR_HARMONIZED', 'temporal': 'median', 'add_ndvi': True},
    's1':            {'enabled': False, 'id': 'COPERNICUS/S1_GRD', 'temporal': 'median'},
    'cropland_usda': {'enabled': True, 'id': 'USDA/NASS/CDL', 'temporal': 'mode', 'categorical': True},
    'dynamic_world': {'enabled': True, 'id': 'GOOGLE/DYNAMICWORLD/V1', 'temporal': 'mode', 'categorical': True},
    'water_jrc':     {'enabled': True, 'id': 'JRC/GSW1_4/YearlyHistory', 'temporal': 'mode', 'categorical': True},
    'dem':           {'enabled': True, 'id': 'COPERNICUS/DEM/GLO30', 'static': True},

    # NEW: MODIS for longer time series
    'modis_ndvi': {
        'enabled': True,
        'id': 'MODIS/061/MOD13A1',
        'temporal': 'median',
        'scale_override': 500  # MODIS native resolution
    },

    'modis_lst': {
        'enabled': True,
        'id': 'MODIS/061/MOD11A2',
        'temporal': 'mean',
        'scale_override': 1000,
        'preprocessing': lambda img: img.multiply(0.02).subtract(273.15)  # Kelvin to Celsius
    },

    # NEW: Soil properties (static)
    'soil_texture': {
        'enabled': True,
        'id': 'OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02',
        'static': True
    },

    # OPTIONAL: VIIRS Nightlights (economic activity proxy)
    'nightlights': {
        'enabled': False,  # Set True if you want urban-rural gradient
        'id': 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG',
        'temporal': 'mean'
    }
}

# =====================================================
#               HELPER FUNCTIONS (ADDED FIXES)
# =====================================================

def mask_landsat(image):
    """
    Cloud masking for Landsat 8/9 using QA_PIXEL band.
    Scales optical bands to 0-1 surface reflectance.
    """
    qa = image.select('QA_PIXEL')
    # Mask Dilated Cloud, Cirrus, Cloud, and Cloud Shadow
    mask = qa.bitwiseAnd(1 << 1).eq(0) \
        .And(qa.bitwiseAnd(1 << 2).eq(0)) \
        .And(qa.bitwiseAnd(1 << 3).eq(0)) \
        .And(qa.bitwiseAnd(1 << 4).eq(0))

    # Apply scaling factors for Landsat Collection 2
    return image.updateMask(mask) \
        .multiply(0.0000275).add(-0.2) \
        .copyProperties(image, image.propertyNames())

def mask_s2(image):
    """
    Cloud masking for Sentinel-2 using QA60 band.
    Scales to 0-1 surface reflectance.
    """
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
             qa.bitwiseAnd(cirrusBitMask).eq(0))

    # Scale to 0-1 (S2 data is typically 0-10000)
    return image.updateMask(mask).divide(10000) \
                .copyProperties(image, image.propertyNames())

def fix_cdl(image):
    """Ensure CDL has the correct band name."""
    return image.select(['cropland'])

# =====================================================

# ------------------- TEXTURE FEATURES (NEW) -------------------
def add_texture_features(img):
    """
    Compute GLCM texture metrics for Sentinel-1.
    Useful for crop structure characterization.
    """
    glcm = img.select(['VV']).glcmTexture(size=3)

    texture_bands = [
        'VV_asm',   # Angular Second Moment (uniformity)
        'VV_contrast',
        'VV_corr',  # Correlation
        'VV_ent'    # Entropy (randomness)
    ]

    return img.addBands(glcm.select(texture_bands))

# ------------------- SPECTRAL INDICES (EXPANDED) -------------------
def add_spectral_indices(img, sensor='landsat'):
    """
    Add comprehensive spectral indices for agricultural monitoring.
    """
    if sensor == 'landsat':
        nir, red, green, swir1 = 'SR_B5', 'SR_B4', 'SR_B3', 'SR_B6'
    else:  # sentinel-2
        nir, red, green, swir1 = 'B8', 'B4', 'B3', 'B11'

    # Vegetation indices
    ndvi = img.normalizedDifference([nir, red]).rename('NDVI')
    evi = img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': img.select(nir), 'RED': img.select(red), 'BLUE': img.select('SR_B2' if sensor=='landsat' else 'B2')}
    ).rename('EVI')

    savi = img.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
        {'NIR': img.select(nir), 'RED': img.select(red)}
    ).rename('SAVI')

    # Moisture/water indices
    ndwi = img.normalizedDifference([green, nir]).rename('NDWI')
    ndmi = img.normalizedDifference([nir, swir1]).rename('NDMI')  # Normalized Difference Moisture Index

    # Soil/bare ground
    bsi = img.expression(
        '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
        {
            'SWIR': img.select(swir1),
            'RED': img.select(red),
            'NIR': img.select(nir),
            'BLUE': img.select('SR_B2' if sensor=='landsat' else 'B2')
        }
    ).rename('BSI')

    return img.addBands([ndvi, evi, savi, ndwi, ndmi, bsi])

# ------------------- UPDATED PIPELINE -------------------
class AgricultureRSPipeline:
    def __init__(self):
        drive.mount('/content/drive')
        self.fc = ee.FeatureCollection('TIGER/2018/Counties')\
                  .filter(ee.Filter.inList('STATEFP',
                      ['02','15','60','66','69','72','78']).Not())
        self.states = STATES or self.fc.aggregate_array('STATEFP').distinct().getInfo()

    def export(self, table, name, year=None):
        """Export to Drive with error handling."""
        suffix = f"_{year}" if year else ""
        task_name = f"{name}{suffix}"

        task = ee.batch.Export.table.toDrive(
            collection=table,
            folder=FOLDER,
            description=task_name,
            fileNamePrefix=task_name,
            fileFormat='CSV'
        )

        try:
            task.start()
            print(f"→ Task submitted: {task_name}")
            return task
        except ee.EEException as e:
            print(f"!! FAILED: {task_name}. Error: {e}")
            return None

    def run(self):
        """Main execution loop."""
        active_task_count = 0
        QUEUE_LIMIT = 2900

        for asset, cfg in ASSET_MAP.items():
            if not cfg['enabled']:
                continue

            print(f"\n=== {asset.upper()} ===")
            bands = BANDS[asset]
            is_static = cfg.get('static', False)
            scale = cfg.get('scale_override', SCALE)

            for year in (YEARS if not is_static else [None]):
                for state in self.states:
                    # Check existing files
                    suffix = f"_{year}" if year else ""
                    file_path = os.path.join(BASE, f"{asset}_{state}{suffix}.csv")
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        print(f"  -> File exists: {asset}_{state}{suffix}.csv. Skipping.")
                        continue

                    # Queue management
                    while active_task_count >= QUEUE_LIMIT:
                        print(f"  ...Queue full. Waiting 5 min...")
                        time.sleep(300)
                        active_tasks = [t for t in ee.batch.Task.list() if t.state in ['RUNNING', 'READY']]
                        active_task_count = len(active_tasks)

                    # Build image
                    counties = self.fc.filter(ee.Filter.eq('STATEFP', state))
                    geom = counties.geometry().bounds()

                    try:
                        if is_static:
                            img = ee.Image(cfg['id']).select(bands)
                        else:
                            start, end = f'{year}-01-01', f'{year}-12-31'
                            col = ee.ImageCollection(cfg['id']).filterDate(start, end).filterBounds(geom)

                            # Sensor-specific preprocessing
                            if asset == 'landsat89':
                                if year >= 2022:
                                    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterDate(start, end).filterBounds(geom)
                                    col = col.merge(l9)
                                # --- ERROR FIXED: mask_landsat was missing ---
                                col = col.map(mask_landsat)
                                col = col.map(lambda i: add_spectral_indices(i, 'landsat'))

                            elif asset == 's2':
                                # --- ERROR FIXED: mask_s2 was missing ---
                                col = col.map(mask_s2)
                                col = col.map(lambda i: add_spectral_indices(i, 's2'))

                            elif asset == 's1':
                                col = col.map(add_texture_features)

                            elif asset == 'modis_lst' and 'preprocessing' in cfg:
                                col = col.map(cfg['preprocessing'])

                            elif asset == 'cropland_usda':
                                # --- ERROR FIXED: fix_cdl was missing ---
                                col = col.map(fix_cdl)

                            # Temporal reduction
                            if cfg['temporal'] == 'median':
                                img = col.median()
                            elif cfg['temporal'] == 'mean':
                                img = col.mean()
                            else:
                                img = col.mode()

                            # Select final bands
                            all_bands = bands + (['NDVI', 'EVI', 'SAVI', 'NDWI', 'NDMI', 'BSI'] if cfg.get('add_ndvi') else [])
                            if asset == 's1':
                                all_bands += ['VV_asm', 'VV_contrast', 'VV_corr', 'VV_ent']

                            img = img.select([b for b in all_bands if b in img.bandNames().getInfo()])

                        # Reducers
                        red = ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True)
                        if cfg.get('categorical'):
                            red = red.combine(ee.Reducer.frequencyHistogram(), '', True)
                        else:
                            red = red.combine(ee.Reducer.percentile([10, 25, 50, 75, 90]), '', True)

                        # Spatial aggregation
                        stats = img.reduceRegions(
                            collection=counties,
                            reducer=red,
                            scale=scale,
                            tileScale=4
                        )

                        # Tag and export
                        def tag(f):
                            props = {'STATEFP': f.get('STATEFP'), 'COUNTYFP': f.get('COUNTYFP'), 'NAME': f.get('NAME')}
                            if year: props['year'] = year
                            return ee.Feature(None, props).copyProperties(f, f.propertyNames())

                        stats = stats.map(tag)
                        new_task = self.export(stats, f"{asset}_{state}", year)
                        if new_task:
                            active_task_count += 1

                    except Exception as e:
                        print(f"!! ERROR: {asset}_{state}_{year}: {e}")

        print(f"\nAll tasks queued! → Drive → {FOLDER}")

# RUN
pipeline = AgricultureRSPipeline()
pipeline.run()


# =====================================================
# 2B
# =====================================================

#  =====================================================
#  ENHANCED GEE PIPELINE FOR REMOTE SENSING FOCUS
#  Version: Agricultural-Only (No SDOH/Weather)
# =====================================================

# ------------------- CONFIG -------------------
TEST_MODE = False
SCALE = 250
FOLDER = 'Agricultural_RS_LE_2025'
BASE = f'/content/drive/MyDrive/{FOLDER}'
os.makedirs(BASE, exist_ok=True)

if TEST_MODE:
    YEARS = [2020, 2021, 2022]
    STATES = ['06']  # California
else:
    YEARS = range(2000, 2025)
    STATES = None

# ------------------- EXPANDED BAND CATALOG -------------------
BANDS = {
    # Your existing sensors
    'landsat89': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
    's2': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
    's1': ['VV', 'VH'],
    'cropland_usda': ['cropland'],
    'dynamic_world': ['water', 'trees', 'grass', 'flooded_vegetation',
                      'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'],
    'water_jrc': ['waterClass'],
    'dem': ['DEM'],

    # NEW: Additional products
    'modis_ndvi': ['NDVI', 'EVI'],
    'modis_lst': ['LST_Day_1km', 'LST_Night_1km'],
    'soil_texture': ['b0'],  # Surface soil texture
    'nightlights': ['avg_rad'],  # Proxy for economic activity (optional)
}

# ------------------- EXPANDED ASSET MAP -------------------
ASSET_MAP = {
    # Existing
    'landsat89':     {'enabled': True, 'id': 'LANDSAT/LC08/C02/T1_L2', 'temporal': 'median', 'add_ndvi': True},
    's2':            {'enabled': True, 'id': 'COPERNICUS/S2_SR_HARMONIZED', 'temporal': 'median', 'add_ndvi': True},
    's1':            {'enabled': True, 'id': 'COPERNICUS/S1_GRD', 'temporal': 'median'},
    'cropland_usda': {'enabled': False, 'id': 'USDA/NASS/CDL', 'temporal': 'mode', 'categorical': True},
    'dynamic_world': {'enabled': False, 'id': 'GOOGLE/DYNAMICWORLD/V1', 'temporal': 'mode', 'categorical': True},
    'water_jrc':     {'enabled': False, 'id': 'JRC/GSW1_4/YearlyHistory', 'temporal': 'mode', 'categorical': True},
    'dem':           {'enabled': False, 'id': 'COPERNICUS/DEM/GLO30', 'static': True},

    # NEW: MODIS for longer time series
    'modis_ndvi': {
        'enabled': True,
        'id': 'MODIS/061/MOD13A1',
        'temporal': 'median',
        'scale_override': 500  # MODIS native resolution
    },

    'modis_lst': {
        'enabled': True,
        'id': 'MODIS/061/MOD11A2',
        'temporal': 'mean',
        'scale_override': 1000,
        'preprocessing': lambda img: img.multiply(0.02).subtract(273.15)  # Kelvin to Celsius
    },

    # NEW: Soil properties (static)
    'soil_texture': {
        'enabled': True,
        'id': 'OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02',
        'static': True
    },

    # OPTIONAL: VIIRS Nightlights (economic activity proxy)
    'nightlights': {
        'enabled': False,  # Set True if you want urban-rural gradient
        'id': 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG',
        'temporal': 'mean'
    }
}

# =====================================================
#               HELPER FUNCTIONS (ADDED FIXES)
# =====================================================

def mask_landsat(image):
    """
    Cloud masking for Landsat 8/9 using QA_PIXEL band.
    Scales optical bands to 0-1 surface reflectance.
    """
    qa = image.select('QA_PIXEL')
    # Mask Dilated Cloud, Cirrus, Cloud, and Cloud Shadow
    mask = qa.bitwiseAnd(1 << 1).eq(0) \
        .And(qa.bitwiseAnd(1 << 2).eq(0)) \
        .And(qa.bitwiseAnd(1 << 3).eq(0)) \
        .And(qa.bitwiseAnd(1 << 4).eq(0))

    # Apply scaling factors for Landsat Collection 2
    return image.updateMask(mask) \
        .multiply(0.0000275).add(-0.2) \
        .copyProperties(image, image.propertyNames())

def mask_s2(image):
    """
    Cloud masking for Sentinel-2 using QA60 band.
    Scales to 0-1 surface reflectance.
    """
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
             qa.bitwiseAnd(cirrusBitMask).eq(0))

    # Scale to 0-1 (S2 data is typically 0-10000)
    return image.updateMask(mask).divide(10000) \
                .copyProperties(image, image.propertyNames())

def fix_cdl(image):
    """Ensure CDL has the correct band name."""
    return image.select(['cropland'])

# =====================================================

# ------------------- TEXTURE FEATURES (NEW) -------------------
def add_texture_features(img):
    """
    Compute GLCM texture metrics for Sentinel-1.
    Useful for crop structure characterization.
    """
    glcm = img.select(['VV']).glcmTexture(size=3)

    texture_bands = [
        'VV_asm',   # Angular Second Moment (uniformity)
        'VV_contrast',
        'VV_corr',  # Correlation
        'VV_ent'    # Entropy (randomness)
    ]

    return img.addBands(glcm.select(texture_bands))

# ------------------- SPECTRAL INDICES (EXPANDED) -------------------
def add_spectral_indices(img, sensor='landsat'):
    """
    Add comprehensive spectral indices for agricultural monitoring.
    """
    if sensor == 'landsat':
        nir, red, green, swir1 = 'SR_B5', 'SR_B4', 'SR_B3', 'SR_B6'
    else:  # sentinel-2
        nir, red, green, swir1 = 'B8', 'B4', 'B3', 'B11'

    # Vegetation indices
    ndvi = img.normalizedDifference([nir, red]).rename('NDVI')
    evi = img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': img.select(nir), 'RED': img.select(red), 'BLUE': img.select('SR_B2' if sensor=='landsat' else 'B2')}
    ).rename('EVI')

    savi = img.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
        {'NIR': img.select(nir), 'RED': img.select(red)}
    ).rename('SAVI')

    # Moisture/water indices
    ndwi = img.normalizedDifference([green, nir]).rename('NDWI')
    ndmi = img.normalizedDifference([nir, swir1]).rename('NDMI')  # Normalized Difference Moisture Index

    # Soil/bare ground
    bsi = img.expression(
        '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
        {
            'SWIR': img.select(swir1),
            'RED': img.select(red),
            'NIR': img.select(nir),
            'BLUE': img.select('SR_B2' if sensor=='landsat' else 'B2')
        }
    ).rename('BSI')

    return img.addBands([ndvi, evi, savi, ndwi, ndmi, bsi])

# ------------------- UPDATED PIPELINE -------------------
class AgricultureRSPipeline:
    def __init__(self):
        drive.mount('/content/drive')
        self.fc = ee.FeatureCollection('TIGER/2018/Counties')\
                  .filter(ee.Filter.inList('STATEFP',
                      ['02','15','60','66','69','72','78']).Not())
        self.states = STATES or self.fc.aggregate_array('STATEFP').distinct().getInfo()

    def export(self, table, name, year=None):
        """Export to Drive with error handling."""
        suffix = f"_{year}" if year else ""
        task_name = f"{name}{suffix}"

        task = ee.batch.Export.table.toDrive(
            collection=table,
            folder=FOLDER,
            description=task_name,
            fileNamePrefix=task_name,
            fileFormat='CSV'
        )

        try:
            task.start()
            print(f"→ Task submitted: {task_name}")
            return task
        except ee.EEException as e:
            print(f"!! FAILED: {task_name}. Error: {e}")
            return None

    def run(self):
        """Main execution loop."""
        active_task_count = 0
        QUEUE_LIMIT = 2900

        for asset, cfg in ASSET_MAP.items():
            if not cfg['enabled']:
                continue

            print(f"\n=== {asset.upper()} ===")
            bands = BANDS[asset]
            is_static = cfg.get('static', False)
            scale = cfg.get('scale_override', SCALE)

            for year in (YEARS if not is_static else [None]):
                for state in self.states:
                    # Check existing files
                    suffix = f"_{year}" if year else ""
                    file_path = os.path.join(BASE, f"{asset}_{state}{suffix}.csv")
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        print(f"  -> File exists: {asset}_{state}{suffix}.csv. Skipping.")
                        continue

                    # Queue management
                    while active_task_count >= QUEUE_LIMIT:
                        print(f"  ...Queue full. Waiting 5 min...")
                        time.sleep(300)
                        active_tasks = [t for t in ee.batch.Task.list() if t.state in ['RUNNING', 'READY']]
                        active_task_count = len(active_tasks)

                    # Build image
                    counties = self.fc.filter(ee.Filter.eq('STATEFP', state))
                    geom = counties.geometry().bounds()

                    try:
                        if is_static:
                            img = ee.Image(cfg['id']).select(bands)
                        else:
                            start, end = f'{year}-01-01', f'{year}-12-31'
                            col = ee.ImageCollection(cfg['id']).filterDate(start, end).filterBounds(geom)

                            # Sensor-specific preprocessing
                            if asset == 'landsat89':
                                if year >= 2022:
                                    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterDate(start, end).filterBounds(geom)
                                    col = col.merge(l9)
                                # --- ERROR FIXED: mask_landsat was missing ---
                                col = col.map(mask_landsat)
                                col = col.map(lambda i: add_spectral_indices(i, 'landsat'))

                            elif asset == 's2':
                                # --- ERROR FIXED: mask_s2 was missing ---
                                col = col.map(mask_s2)
                                col = col.map(lambda i: add_spectral_indices(i, 's2'))

                            elif asset == 's1':
                                col = col.map(add_texture_features)

                            elif asset == 'modis_lst' and 'preprocessing' in cfg:
                                col = col.map(cfg['preprocessing'])

                            elif asset == 'cropland_usda':
                                # --- ERROR FIXED: fix_cdl was missing ---
                                col = col.map(fix_cdl)

                            # Temporal reduction
                            if cfg['temporal'] == 'median':
                                img = col.median()
                            elif cfg['temporal'] == 'mean':
                                img = col.mean()
                            else:
                                img = col.mode()

                            # Select final bands
                            all_bands = bands + (['NDVI', 'EVI', 'SAVI', 'NDWI', 'NDMI', 'BSI'] if cfg.get('add_ndvi') else [])
                            if asset == 's1':
                                all_bands += ['VV_asm', 'VV_contrast', 'VV_corr', 'VV_ent']

                            img = img.select([b for b in all_bands if b in img.bandNames().getInfo()])

                        # Reducers
                        red = ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True)
                        if cfg.get('categorical'):
                            red = red.combine(ee.Reducer.frequencyHistogram(), '', True)
                        else:
                            red = red.combine(ee.Reducer.percentile([10, 25, 50, 75, 90]), '', True)

                        # Spatial aggregation
                        stats = img.reduceRegions(
                            collection=counties,
                            reducer=red,
                            scale=scale,
                            tileScale=4
                        )

                        # Tag and export
                        def tag(f):
                            props = {'STATEFP': f.get('STATEFP'), 'COUNTYFP': f.get('COUNTYFP'), 'NAME': f.get('NAME')}
                            if year: props['year'] = year
                            return ee.Feature(None, props).copyProperties(f, f.propertyNames())

                        stats = stats.map(tag)
                        new_task = self.export(stats, f"{asset}_{state}", year)
                        if new_task:
                            active_task_count += 1

                    except Exception as e:
                        print(f"!! ERROR: {asset}_{state}_{year}: {e}")

        print(f"\nAll tasks queued! → Drive → {FOLDER}")

# RUN
pipeline = AgricultureRSPipeline()
pipeline.run()


# =====================================================
# 2C
# S1 data collection code is the same as above, but with the following modifications:
# =====================================================
# ------------------- CONFIG -------------------
TEST_MODE = False
SCALE = 250
FOLDER = 'Agricultural_RS_LE_2025'
BASE = f'/content/drive/MyDrive/{FOLDER}'
os.makedirs(BASE, exist_ok=True)

if TEST_MODE:
    YEARS = range(2020, 2021) # Reduced for testing
    STATES = ['06']  # California
else:
    YEARS = range(2000, 2025)
    STATES = None

# ------------------- BAND CATALOG -------------------
BANDS = {
    'landsat89': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
    's2': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
    's1': ['VV', 'VH'], # We will strictly enforce VV/VH to prevent errors
}

# ------------------- ASSET MAP -------------------
ASSET_MAP = {
    'landsat89': {'enabled': False, 'id': 'LANDSAT/LC08/C02/T1_L2', 'temporal': 'median', 'add_ndvi': True},
    's2': {'enabled': False, 'id': 'COPERNICUS/S2_SR_HARMONIZED', 'temporal': 'median', 'add_ndvi': True},
    's1': {'enabled': True, 'id': 'COPERNICUS/S1_GRD', 'temporal': 'median'},
    'cropland_usda': {'enabled': False, 'id': 'USDA/NASS/CDL', 'temporal': 'mode', 'categorical': True},
    'dynamic_world': {'enabled': False, 'id': 'GOOGLE/DYNAMICWORLD/V1', 'temporal': 'mode', 'categorical': True},
    'water_jrc': {'enabled': False, 'id': 'JRC/GSW1_4/YearlyHistory', 'temporal': 'mode', 'categorical': True},
    'dem': {'enabled': False, 'id': 'COPERNICUS/DEM/GLO30', 'static': True},
    'modis_ndvi': {'enabled': False, 'id': 'MODIS/061/MOD13A1', 'temporal': 'median', 'scale_override': 500},
    'modis_lst': {'enabled': False, 'id': 'MODIS/061/MOD11A2', 'temporal': 'mean', 'scale_override': 1000, 'preprocessing': lambda img: img.multiply(0.02).subtract(273.15)},
    'nightlights': {'enabled': False, 'id': 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG', 'temporal': 'mean'}
}

# ------------------- HELPER FUNCTIONS -------------------
def mask_landsat(image):
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 1).eq(0) \
        .And(qa.bitwiseAnd(1 << 2).eq(0)) \
        .And(qa.bitwiseAnd(1 << 3).eq(0)) \
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(mask) \
        .multiply(0.0000275).add(-0.2) \
        .copyProperties(image, image.propertyNames())

def mask_s2(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
                 qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000) \
        .copyProperties(image, image.propertyNames())

def add_spectral_indices(img, sensor='landsat'):
    if sensor == 'landsat':
        nir, red = 'SR_B5', 'SR_B4'
    else:
        nir, red = 'B8', 'B4'

    ndvi = img.normalizedDifference([nir, red]).rename('NDVI')
    return img.addBands([ndvi])

# ------------------- ROBUST S1 TEXTURE FUNCTION -------------------
def add_s1_texture(img):
    """
    Computes GLCM on VV band.
    Assumes input image has 'VV' band (guaranteed by filter in main loop).
    """
    # 1. Select VV band
    vv = img.select('VV')

    # 2. Fix for "Only 32-bit integers":
    # Scale float (-20 to 0) to integer (-2000 to 0) so GLCM works.
    # .toInt32() is CRITICAL here.
    vv_int = vv.multiply(100).toInt32()

    # 3. Compute GLCM
    # Neighborhood size 3 is standard/efficient
    glcm = vv_int.glcmTexture(size=3)

    # 4. Select only the useful metrics and rename them cleanly
    # Default names are usually 'VV_asm', 'VV_contrast' etc.
    bands_to_keep = ['VV_asm', 'VV_contrast', 'VV_corr', 'VV_ent']

    return img.addBands(glcm.select(bands_to_keep))

# ------------------- PIPELINE CLASS -------------------
class AgricultureRSPipeline:
    def __init__(self):
        try:
            drive.mount('/content/drive')
        except:
            pass
        # Exclude territories for cleaner US data
        self.fc = ee.FeatureCollection('TIGER/2018/Counties') \
            .filter(ee.Filter.inList('STATEFP', ['02', '15', '60', '66', '69', '72', '78']).Not())

        if STATES:
             self.states = STATES
        else:
             # Get list of states dynamically
             self.states = self.fc.aggregate_array('STATEFP').distinct().getInfo()

    def export(self, table, name, year=None):
        """Export to Drive with error handling."""
        suffix = f"_{year}" if year else ""
        task_name = f"{name}{suffix}"

        task = ee.batch.Export.table.toDrive(
            collection=table,
            folder=FOLDER,
            description=task_name,
            fileNamePrefix=task_name,
            fileFormat='CSV'
        )

        try:
            task.start()
            print(f"→ Task submitted: {task_name}")
            return task
        except ee.EEException as e:
            print(f"!! FAILED: {task_name}. Error: {e}")
            return None

    def run(self):
        """Main execution loop."""
        active_task_count = 0

        for asset, cfg in ASSET_MAP.items():
            if not cfg['enabled']:
                continue

            print(f"\n=== {asset.upper()} ===")
            bands = BANDS.get(asset, [])
            is_static = cfg.get('static', False)
            scale = cfg.get('scale_override', SCALE)

            for year in (YEARS if not is_static else [None]):
                for state in self.states:
                    # Check existing files to skip processing
                    suffix = f"_{year}" if year else ""
                    file_path = os.path.join(BASE, f"{asset}_{state}{suffix}.csv")
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        print(f"  -> File exists: {asset}_{state}{suffix}.csv. Skipping.")
                        continue

                    # Define ROI
                    counties = self.fc.filter(ee.Filter.eq('STATEFP', state))
                    geom = counties.geometry().bounds()

                    try:
                        if is_static:
                            img = ee.Image(cfg['id']).select(bands)
                        else:
                            start, end = f'{year}-01-01', f'{year}-12-31'
                            col = ee.ImageCollection(cfg['id']).filterDate(start, end).filterBounds(geom)

                            # --- SENSOR SPECIFIC PROCESSING ---
                            if asset == 'landsat89':
                                if year >= 2022:
                                    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterDate(start, end).filterBounds(geom)
                                    col = col.merge(l9)
                                col = col.map(mask_landsat).map(lambda i: add_spectral_indices(i, 'landsat'))

                            elif asset == 's2':
                                col = col.map(mask_s2).map(lambda i: add_spectral_indices(i, 's2'))

                            elif asset == 's1':
                                # --- FIX: PRE-FILTERING IS KEY ---
                                # Filter for IW mode (standard over land)
                                col = col.filter(ee.Filter.eq('instrumentMode', 'IW'))
                                # Filter for images that actually have VV and VH bands
                                col = col.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                                col = col.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                                # Direction Pass (Ascending/Descending mix can cause noise, but usually handled by median)
                                # Now map the texture function safely
                                col = col.map(add_s1_texture)

                            elif asset == 'modis_lst' and 'preprocessing' in cfg:
                                col = col.map(cfg['preprocessing'])

                            elif asset == 'cropland_usda':
                                def fix_cdl(image): return image.select(['cropland'])
                                col = col.map(fix_cdl)

                            # Check if collection is empty after filtering
                            # (We use .first() to check without downloading everything)
                            if col.size().getInfo() == 0:
                                print(f"  !! No data found for {asset} in state {state} for {year}")
                                continue

                            # Temporal reduction
                            if cfg['temporal'] == 'median':
                                img = col.median()
                            elif cfg['temporal'] == 'mean':
                                img = col.mean()
                            else:
                                img = col.mode()

                            # --- DYNAMIC BAND SELECTION ---
                            # We construct the list of bands we expect to be present
                            expected_bands = list(bands)
                            if cfg.get('add_ndvi'):
                                expected_bands += ['NDVI']
                            if asset == 's1':
                                # Add the texture bands we created
                                expected_bands += ['VV_asm', 'VV_contrast', 'VV_corr', 'VV_ent']

                            # Select only bands that actually exist in the composite
                            available_bands = img.bandNames()
                            img = img.select(available_bands.filter(ee.Filter.inList('item', expected_bands)))

                        # Reducers
                        red = ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True)
                        if cfg.get('categorical'):
                            red = red.combine(ee.Reducer.frequencyHistogram(), '', True)
                        else:
                            red = red.combine(ee.Reducer.percentile([10, 25, 50, 75, 90]), '', True)

                        # Spatial aggregation
                        stats = img.reduceRegions(
                            collection=counties,
                            reducer=red,
                            scale=scale,
                            tileScale=4 # Increased tileScale for heavy S1 computation
                        )

                        # Tag and export
                        def tag(f):
                            props = {'STATEFP': f.get('STATEFP'), 'COUNTYFP': f.get('COUNTYFP'), 'NAME': f.get('NAME')}
                            if year: props['year'] = year
                            return ee.Feature(None, props).copyProperties(f, f.propertyNames())

                        stats = stats.map(tag)

                        # Final check: Don't export empty results
                        new_task = self.export(stats, f"{asset}_{state}", year)
                        if new_task:
                            active_task_count += 1

                    except Exception as e:
                        print(f"!! ERROR processing {asset} for State {state} in {year}: {e}")

        print(f"\nAll tasks queued! → Drive → {FOLDER}")

# RUN
pipeline = AgricultureRSPipeline()
pipeline.run()


# =====================================================
# 2D
# Merging code with normalization and error fixes for histogram parsing in the master CSV processing step.
# =====================================================

import pandas as pd
import numpy as np
import glob
import os
import re
import ast
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & CLASS MAPPINGS
# ==========================================
SOURCE_DIR = '/content/drive/MyDrive/Agricultural_RS_LE_2025'
DEST_DIR = os.path.join(SOURCE_DIR, 'master csv')

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

# --- CLASS DEFINITIONS (The Decoder Rings) ---
CLASS_MAPS = {
    'Cropland_USDA': {
        '1': 'Corn', '2': 'Cotton', '3': 'Rice', '4': 'Sorghum', '5': 'Soybeans',
        '6': 'Sunflower', '10': 'Peanuts', '11': 'Tobacco', '12': 'Sweet Corn',
        '21': 'Barley', '23': 'Spring Wheat', '24': 'Winter Wheat', '28': 'Oats',
        '36': 'Alfalfa', '37': 'Other Hay', '44': 'Other Crops', '61': 'Fallow',
        '111': 'Open Water', '121': 'Dev_Open', '122': 'Dev_Low',
        '123': 'Dev_Med', '124': 'Dev_High', '141': 'Forest_Deciduous',
        '142': 'Forest_Evergreen', '143': 'Forest_Mixed', '152': 'Shrubland',
        '176': 'Grassland', '190': 'Wetlands_Woody', '195': 'Wetlands_Herbaceous'
    },
    'Dynamic_World': {
        '0': 'Water', '1': 'Trees', '2': 'Grass', '3': 'FloodedVeg',
        '4': 'Crops', '5': 'ShrubScrub', '6': 'Built', '7': 'Bare', '8': 'SnowIce'
    },
    'Water_JRC': {
        '1': 'NotWater', '2': 'Seasonal', '3': 'Permanent'
    }
}

# ==========================================
# 2. PARSING ENGINE
# ==========================================
def parse_histogram_str(text):
    """Robustly parses string '{k=v, ...}' or JSON into a python dict."""
    if pd.isna(text) or str(text).strip() in ['', '{}', 'nan']:
        return {}

    # 1. Fast Path: Standard JSON
    if ':' in str(text) and '{' in str(text):
        try:
            return ast.literal_eval(str(text))
        except: pass

    # 2. Robust Path: GEE "key=value" format
    clean = str(text).replace('{', '').replace('}', '')
    if not clean: return {}

    # Regex handles scientific notation (e.g. 1.2e-5)
    pattern = r'([\d\w\.\-\+]+)\s*=\s*([\d\w\.\-\+eE]+)'
    matches = re.findall(pattern, clean)

    data = {}
    for k, v in matches:
        try:
            # Clean Key: Convert "1.0" -> "1" (Int) for cleaner mapping
            key_num = float(k)
            key = str(int(key_num)) if key_num.is_integer() else str(k)

            data[key] = float(v)
        except: continue
    return data

# ==========================================
# 3. GENERIC PROCESSING (One Code to Rule Them All)
# ==========================================
def process_sensor_master(sensor_name, file_pattern):
    print(f"\n🚜 Processing {sensor_name.upper()}...")

    files = glob.glob(os.path.join(SOURCE_DIR, file_pattern))
    if not files:
        print(f"   ⚠️ No files found for {sensor_name}")
        return

    all_rows = []

    # Get specific mapping for this sensor (if any)
    sensor_map = CLASS_MAPS.get(sensor_name, {})

    for f in tqdm(files, desc=f"Reading {sensor_name}"):
        try:
            # 1. Load File
            df = pd.read_csv(f, on_bad_lines='skip', low_memory=False)

            # 2. Metadata Cleanup
            if 'system:index' in df.columns:
                df = df.drop(columns=['system:index', '.geo', 'Unnamed: 0'], errors='ignore')

            # Ensure GEOID is padded string
            if 'GEOID' not in df.columns and 'STATEFP' in df.columns:
                 df['GEOID'] = df['STATEFP'].astype(str).str.zfill(2) + \
                               df['COUNTYFP'].astype(str).str.zfill(3)

            # Recover Year if missing
            if 'year' not in df.columns:
                parts = os.path.basename(f).replace('.csv','').split('_')
                if parts[-1].isdigit() and len(parts[-1]) == 4:
                    df['year'] = int(parts[-1])

            # 3. HISTOGRAM EXPLOSION
            # Find any column ending in 'histogram'
            hist_cols = [c for c in df.columns if 'histogram' in c.lower()]

            if hist_cols:
                for h_col in hist_cols:
                    # Skip Dynamic World "Confidence" histograms (Probability noise)
                    # We only want the Class Counts (Integer keys)
                    sample = df[h_col].dropna().iloc[0] if not df[h_col].dropna().empty else ""
                    if '0.1' in str(sample) and sensor_name == 'Dynamic_World':
                        df = df.drop(columns=[h_col]) # Drop noise
                        continue

                    # Parse
                    parsed_series = df[h_col].apply(parse_histogram_str)

                    # Explode
                    exploded = pd.json_normalize(parsed_series)
                    exploded.index = df.index

                    # --- INTELLIGENT RENAMING ---
                    new_names = {}
                    for col in exploded.columns:
                        # col is likely "1", "5", etc.
                        if col in sensor_map:
                            # Map "1" -> "Corn"
                            name_suffix = sensor_map[col]
                        else:
                            # Fallback "Class_1"
                            name_suffix = f"Class_{col}"

                        new_names[col] = f"{sensor_name}_{name_suffix}_pct"

                    exploded = exploded.rename(columns=new_names)

                    # --- CRITICAL: ZERO FILLING (LOCAL ONLY) ---
                    # Only fill 0s for crops *within this file*.
                    # If this file exists, we know missing keys = 0 pixels.
                    exploded = exploded.fillna(0)

                    # Normalize to Percentage
                    row_sums = exploded.sum(axis=1)
                    row_sums[row_sums == 0] = 1
                    exploded = exploded.div(row_sums, axis=0)

                    # Merge & Clean
                    df = pd.concat([df, exploded], axis=1)
                    df = df.drop(columns=[h_col])

            all_rows.append(df)

        except Exception as e:
            pass

    # 4. GRAND MERGE (PRESERVING NANs)
    if all_rows:
        # Stack all chunks
        master_df = pd.concat(all_rows, axis=0, ignore_index=True, sort=False)

        # --- CRITICAL: VACANT CELL HANDLING ---
        # We do NOT run master_df.fillna(0) globally.
        # This ensures that if a Year/County didn't exist in the input files,
        # its columns remain NaN (Vacant), indicating "No Data Collected".

        # Final ID Polish
        if 'GEOID' in master_df.columns:
            master_df['GEOID'] = master_df['GEOID'].astype(str).str.split('.').str[0].str.zfill(5)

        # Sort
        sort_cols = [c for c in ['GEOID', 'year'] if c in master_df.columns]
        if sort_cols:
            master_df = master_df.sort_values(sort_cols)

        # Save
        out_path = os.path.join(DEST_DIR, f'MASTER_{sensor_name}_vFinal.csv')
        master_df.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path}")
        print(f"   Shape: {master_df.shape} (Rows, Cols)")

        # Audit
        if 'year' in master_df.columns:
            years = sorted(master_df['year'].dropna().unique().astype(int))
            print(f"   Years: {min(years)} to {max(years)}")

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # Define datasets to process
    DATASETS = {
        'Cropland_USDA': 'cropland_usda_*.csv',
        'Dynamic_World': 'dynamic_world_*.csv',
        'Water_JRC':     'water_jrc_*.csv',
        'Landsat':       'landsat89_*.csv',
        'Sentinel2':     's2_*.csv',
        'MODIS_NDVI':    'modis_ndvi_*.csv',
        'MODIS_LST':     'modis_lst_*.csv',
        'DEM':           'dem_*.csv',
        'Soil':          'soil_texture_*.csv'
    }

    for name, pattern in DATASETS.items():
        process_sensor_master(name, pattern)


# =====================================================
# 2E
# Merging code for Sentinel-1 as skipped that in the previous step.
# =====================================================

import pandas as pd
import numpy as np
import glob
import os
import re
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
SOURCE_DIR = '/content/drive/MyDrive/Agricultural_RS_LE_2025'
DEST_DIR = os.path.join(SOURCE_DIR, 'master csv')

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

# ==========================================
# 2. SENTINEL-1 PROCESSOR
# ==========================================
def process_sentinel1_master():
    print(f"\n🛰️  Processing SENTINEL-1 (SAR Structure)...")

    # Pattern to match your files (e.g., s1_01_2015.csv)
    files = glob.glob(os.path.join(SOURCE_DIR, 's1_*.csv'))

    if not files:
        print("   ⚠️ No Sentinel-1 files found (checked 's1_*.csv').")
        return

    print(f"   -> Found {len(files)} file chunks. Stacking...")

    all_rows = []

    for f in tqdm(files, desc="Reading S1"):
        try:
            # 1. Load File (Low memory mode false to prevent type errors)
            df = pd.read_csv(f, on_bad_lines='skip', low_memory=False)

            # 2. Metadata Cleanup
            # Drop GEE system columns if they exist
            cols_to_drop = ['system:index', '.geo', 'Unnamed: 0']
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

            # 3. Ensure GEOID is correct
            # If GEOID is missing, try to build it from State/County FP
            if 'GEOID' not in df.columns and 'STATEFP' in df.columns:
                 df['GEOID'] = df['STATEFP'].astype(str).str.zfill(2) + \
                               df['COUNTYFP'].astype(str).str.zfill(3)

            # 4. Recover Year from Filename
            # Your naming convention is 's1_STATE_YEAR.csv'
            if 'year' not in df.columns:
                filename = os.path.basename(f)
                parts = filename.replace('.csv', '').split('_')
                # Check the last part for a 4-digit year
                if parts[-1].isdigit() and len(parts[-1]) == 4:
                    df['year'] = int(parts[-1])

            # 5. Add to Stack
            all_rows.append(df)

        except Exception as e:
            # Silent fail for corrupt individual files, just skip them
            pass

    # 3. GRAND MERGE
    if all_rows:
        # Concatenate all years/states
        master_df = pd.concat(all_rows, axis=0, ignore_index=True, sort=False)

        # Final ID Polish (Ensure 5-digit string '01001')
        if 'GEOID' in master_df.columns:
            master_df['GEOID'] = master_df['GEOID'].astype(str).str.split('.').str[0].str.zfill(5)

        # Sort by ID and Year
        sort_cols = [c for c in ['GEOID', 'year'] if c in master_df.columns]
        if sort_cols:
            master_df = master_df.sort_values(sort_cols)

        # 4. Save
        out_path = os.path.join(DEST_DIR, 'MASTER_Sentinel1_vFinal.csv')
        master_df.to_csv(out_path, index=False)

        print(f"✅ Saved: {out_path}")
        print(f"   Shape: {master_df.shape} (Rows, Cols)")

        # Validation Stats
        if 'year' in master_df.columns:
            years = sorted(master_df['year'].dropna().unique().astype(int))
            print(f"   Years Covered: {min(years)} to {max(years)}")
            print(f"   Columns Included: {len(master_df.columns)}")

if __name__ == "__main__":
    process_sentinel1_master()



# =====================================================
# 2F
# Final GEE data merging for individual sensors.
# =====================================================
import pandas as pd
import numpy as np
import glob
import os
import re
import ast
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & CLASS MAPPINGS
# ==========================================
SOURCE_DIR = '/content/drive/MyDrive/Agricultural_RS_LE_2025'
DEST_DIR = os.path.join(SOURCE_DIR, 'master csv')

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

# --- CLASS DEFINITIONS (The Decoder Rings) ---
CLASS_MAPS = {
    'Cropland_USDA': {
        '1': 'Corn', '2': 'Cotton', '3': 'Rice', '4': 'Sorghum', '5': 'Soybeans',
        '6': 'Sunflower', '10': 'Peanuts', '11': 'Tobacco', '12': 'Sweet Corn',
        '21': 'Barley', '23': 'Spring Wheat', '24': 'Winter Wheat', '28': 'Oats',
        '36': 'Alfalfa', '37': 'Other Hay', '44': 'Other Crops', '61': 'Fallow',
        '111': 'Open Water', '121': 'Dev_Open', '122': 'Dev_Low',
        '123': 'Dev_Med', '124': 'Dev_High', '141': 'Forest_Deciduous',
        '142': 'Forest_Evergreen', '143': 'Forest_Mixed', '152': 'Shrubland',
        '176': 'Grassland', '190': 'Wetlands_Woody', '195': 'Wetlands_Herbaceous'
    },
    'Dynamic_World': {
        '0': 'Water', '1': 'Trees', '2': 'Grass', '3': 'FloodedVeg',
        '4': 'Crops', '5': 'ShrubScrub', '6': 'Built', '7': 'Bare', '8': 'SnowIce'
    },
    'Water_JRC': {
        '1': 'NotWater', '2': 'Seasonal', '3': 'Permanent'
    }
}

# ==========================================
# 2. PARSING ENGINE
# ==========================================
def parse_histogram_str(text):
    """Robustly parses string '{k=v, ...}' or JSON into a python dict."""
    if pd.isna(text) or str(text).strip() in ['', '{}', 'nan']:
        return {}

    # 1. Fast Path: Standard JSON
    if ':' in str(text) and '{' in str(text):
        try:
            return ast.literal_eval(str(text))
        except: pass

    # 2. Robust Path: GEE "key=value" format
    clean = str(text).replace('{', '').replace('}', '')
    if not clean: return {}

    # Regex handles scientific notation (e.g. 1.2e-5)
    pattern = r'([\d\w\.\-\+]+)\s*=\s*([\d\w\.\-\+eE]+)'
    matches = re.findall(pattern, clean)

    data = {}
    for k, v in matches:
        try:
            # Clean Key: Convert "1.0" -> "1" (Int) for cleaner mapping
            key_num = float(k)
            key = str(int(key_num)) if key_num.is_integer() else str(k)

            data[key] = float(v)
        except: continue
    return data

# ==========================================
# 3. GENERIC PROCESSING (One Code to Rule Them All)
# ==========================================
def process_sensor_master(sensor_name, file_pattern):
    print(f"\n🚜 Processing {sensor_name.upper()}...")

    files = glob.glob(os.path.join(SOURCE_DIR, file_pattern))
    if not files:
        print(f"   ⚠️ No files found for {sensor_name}")
        return

    all_rows = []

    # Get specific mapping for this sensor (if any)
    sensor_map = CLASS_MAPS.get(sensor_name, {})

    for f in tqdm(files, desc=f"Reading {sensor_name}"):
        try:
            # 1. Load File
            df = pd.read_csv(f, on_bad_lines='skip', low_memory=False)

            # 2. Metadata Cleanup
            if 'system:index' in df.columns:
                df = df.drop(columns=['system:index', '.geo', 'Unnamed: 0'], errors='ignore')

            # Ensure GEOID is padded string
            if 'GEOID' not in df.columns and 'STATEFP' in df.columns:
                 df['GEOID'] = df['STATEFP'].astype(str).str.zfill(2) + \
                               df['COUNTYFP'].astype(str).str.zfill(3)

            # Recover Year if missing
            if 'year' not in df.columns:
                parts = os.path.basename(f).replace('.csv','').split('_')
                if parts[-1].isdigit() and len(parts[-1]) == 4:
                    df['year'] = int(parts[-1])

            # 3. HISTOGRAM EXPLOSION
            # Find any column ending in 'histogram'
            hist_cols = [c for c in df.columns if 'histogram' in c.lower()]

            if hist_cols:
                for h_col in hist_cols:
                    # Skip Dynamic World "Confidence" histograms (Probability noise)
                    # We only want the Class Counts (Integer keys)
                    sample = df[h_col].dropna().iloc[0] if not df[h_col].dropna().empty else ""
                    if '0.1' in str(sample) and sensor_name == 'Dynamic_World':
                        df = df.drop(columns=[h_col]) # Drop noise
                        continue

                    # Parse
                    parsed_series = df[h_col].apply(parse_histogram_str)

                    # Explode
                    exploded = pd.json_normalize(parsed_series)
                    exploded.index = df.index

                    # --- INTELLIGENT RENAMING ---
                    new_names = {}
                    for col in exploded.columns:
                        # col is likely "1", "5", etc.
                        if col in sensor_map:
                            # Map "1" -> "Corn"
                            name_suffix = sensor_map[col]
                        else:
                            # Fallback "Class_1"
                            name_suffix = f"Class_{col}"

                        new_names[col] = f"{sensor_name}_{name_suffix}_pct"

                    exploded = exploded.rename(columns=new_names)

                    # --- CRITICAL: ZERO FILLING (LOCAL ONLY) ---
                    # Only fill 0s for crops *within this file*.
                    # If this file exists, we know missing keys = 0 pixels.
                    exploded = exploded.fillna(0)

                    # Normalize to Percentage
                    row_sums = exploded.sum(axis=1)
                    row_sums[row_sums == 0] = 1
                    exploded = exploded.div(row_sums, axis=0)

                    # Merge & Clean
                    df = pd.concat([df, exploded], axis=1)
                    df = df.drop(columns=[h_col])

            all_rows.append(df)

        except Exception as e:
            pass

    # 4. GRAND MERGE (PRESERVING NANs)
    if all_rows:
        # Stack all chunks
        master_df = pd.concat(all_rows, axis=0, ignore_index=True, sort=False)

        # --- CRITICAL: VACANT CELL HANDLING ---
        # We do NOT run master_df.fillna(0) globally.
        # This ensures that if a Year/County didn't exist in the input files,
        # its columns remain NaN (Vacant), indicating "No Data Collected".

        # Final ID Polish
        if 'GEOID' in master_df.columns:
            master_df['GEOID'] = master_df['GEOID'].astype(str).str.split('.').str[0].str.zfill(5)

        # Sort
        sort_cols = [c for c in ['GEOID', 'year'] if c in master_df.columns]
        if sort_cols:
            master_df = master_df.sort_values(sort_cols)

        # Save
        out_path = os.path.join(DEST_DIR, f'MASTER_{sensor_name}_vFinal.csv')
        master_df.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path}")
        print(f"   Shape: {master_df.shape} (Rows, Cols)")

        # Audit
        if 'year' in master_df.columns:
            years = sorted(master_df['year'].dropna().unique().astype(int))
            print(f"   Years: {min(years)} to {max(years)}")

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # Define datasets to process
    DATASETS = {
        'Cropland_USDA': 'cropland_usda_*.csv',
        'Dynamic_World': 'dynamic_world_*.csv',
        'Water_JRC':     'water_jrc_*.csv',
        'Landsat':       'landsat89_*.csv',
        'Sentinel2':     's2_*.csv',
        'MODIS_NDVI':    'modis_ndvi_*.csv',
        'MODIS_LST':     'modis_lst_*.csv',
        'DEM':           'dem_*.csv',
        'Soil':          'soil_texture_*.csv'
    }

    for name, pattern in DATASETS.items():
        process_sensor_master(name, pattern)



###################################
# 3. Livestock dataset preparation
###################################


# ==========================================
# 3A Calculating livestock counts using zonal statistics on county shapefile and raster data for 2010, 2015, and 2020. This code reads the county shapefile, processes each raster file in parallel to compute mean density and head count for each species, and saves the results to a CSV file.
# ==========================================
import os, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
from tqdm import tqdm

RASTER_DIR = Path("/Users/faizahmad/Desktop/NewLiv/data")
COUNTY_SHP = Path("/Users/faizahmad/Desktop/NewLiv/data/cb_2023_us_county_500k.shp")
OUT_CSV    = Path("county_livestock_2010_2015_2020.csv")
CORES      = max(os.cpu_count() - 1, 1)

def load_counties():
    g = gpd.read_file(COUNTY_SHP).to_crs("EPSG:4326")
    geoids = [c for c in g.columns if c.upper().startswith("GEOID")]
    col = "GEOID" if "GEOID" in geoids else sorted(geoids, key=len)[0]
    if col != "GEOID":
        g = g.rename(columns={col: "GEOID"})
    if "ALAND" in g.columns:
        g["area_km2"] = g["ALAND"] / 1e6
    else:
        g["area_km2"] = g.to_crs(5070).geometry.area / 1e6
    return g[["GEOID","area_km2","geometry"]]

def one_raster(tif, counties):
    year = re.search(r"(\d{4})", tif.name).group(1)
    species = tif.stem.split('.')[-1]
    stats = zonal_stats(
        counties, tif,
        stats=["mean","count"],
        nodata=None,
        all_touched=True
    )
    df = pd.DataFrame(stats)
    df["GEOID"] = counties["GEOID"].values
    df["year"]  = year
    df["species"] = species
    df["head_count"] = df["mean"] * counties["area_km2"]
    return df[["GEOID","year","species","head_count","mean","count"]].rename(
        columns={"mean":"mean_density","count":"cell_count"}
    )

def run_batch():
    counties = load_counties()
    tifs = sorted(RASTER_DIR.glob("*.tif"))
    pieces = []
    with ThreadPoolExecutor(max_workers=CORES) as ex:
        futs = {ex.submit(one_raster, tif, counties): tif for tif in tifs}
        for f in tqdm(as_completed(futs), total=len(futs), desc="processing"):
            pieces.append(f.result())
    out = pd.concat(pieces, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    return out

df = run_batch()

# ==========================================
# 3B Using NLCD and  rasterio directly for more control (e.g., handling nodata, weighting by cell area, etc.)
# ==========================================
! pip install rasterio

import os
import re
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds, intersection
from tqdm import tqdm
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import radians, cos

# Configuration
DATA_DIR = Path("/content/drive/MyDrive/NewLiv/data")
COUNTY_SHP = DATA_DIR / "cb_2023_us_county_500k.shp"
NLCD_RASTER = DATA_DIR / "Annual_NLCD_LndCov_2019_CU_C1V1.tif"
OUT_DIR = DATA_DIR / "temp_county_stats"
OUT_DIR.mkdir(exist_ok=True)
CORES = os.cpu_count() or 1
COMBINED_OUT = DATA_DIR / "combined_livestock_stats.csv"

# Land cover classes (NLCD codes) based on metadata
LAND_USE_CLASSES = {
    "cropland": [82],  # Cultivated Crops
    "grassland": [71],  # Grassland/Herbaceous
    "pasture": [81],   # Pasture/Hay
    "permanent_crops": [],
    "water_bodies": [11, 12],
    "high_pop": [24],
    "other_exclusions": [21, 22, 23, 31, 52, 90, 95]
}

# Species mapping
code2name = {
    "CH": "chicken", "CHK": "chicken", "PG": "pig", "PGS": "pig",
    "CT": "cattle", "CTL": "cattle", "GT": "goat", "GTS": "goat",
    "SH": "sheep", "SHP": "sheep", "BF": "buffalo", "BFL": "buffalo",
    "DK": "duck", "HO": "horse"
}

# Helper Functions
def load_counties():
    """Load and prepare county shapefile."""
    g = gpd.read_file(COUNTY_SHP)
    geoids = [c for c in g.columns if c.upper().startswith("GEOID")]
    col = "GEOID" if "GEOID" in geoids else sorted(geoids, key=len)[0]
    if col != "GEOID":
        g = g.rename(columns={col: "GEOID"})
    g["GEOID"] = g["GEOID"].str.zfill(5)
    g["area_km2"] = g["ALAND"] / 1e6 if "ALAND" in g.columns else g.to_crs(5070).geometry.area / 1e6
    g = g.to_crs("EPSG:4326")
    return g[["GEOID", "area_km2", "geometry"]]

def compute_weighted_mean(data, nodata, geometry, transform, county_area_km2):
    """Compute weighted mean density, standard deviation, and standard error."""
    valid_data = data[data != nodata]
    if len(valid_data) == 0:
        return np.nan, np.nan, np.nan
    # Use accurate cell area based on latitude
    centroid_lat = geometry.centroid.y
    cell_area = (transform.a * 111.32 * np.cos(np.radians(centroid_lat))) * (abs(transform.e) * 111.32)
    weights = np.ones_like(valid_data) * cell_area
    weighted_mean = np.average(valid_data, weights=weights) if len(valid_data) > 0 else np.nan
    std_dev = np.std(valid_data) if len(valid_data) > 0 else np.nan
    std_error = std_dev / np.sqrt(len(valid_data)) if len(valid_data) > 0 else np.nan
    return weighted_mean, std_dev, std_error

def bounds_overlap(bounds1, bounds2):
    """Check if two bounding boxes overlap."""
    return not (bounds1[0] > bounds2[2] or bounds1[2] < bounds2[0] or bounds1[1] > bounds2[3] or bounds1[3] < bounds2[1])

def extract_species_from_filename(filename):
    """Extract species code from raster filename based on known patterns."""
    match_glw4 = re.search(r"GLW4-\d{4}\.D-DA\.([A-Z]{3})\.tif", filename)
    if match_glw4:
        return match_glw4.group(1)
    match_5_da = re.search(r"5_([A-Za-z]{2})_\d{4}_Da\.tif", filename)
    if match_5_da:
        return match_5_da.group(1).upper()
    return ""

def compute_idw_density(geoid, species, year, counties, livestock_raster_paths):
    """Compute IDW-based density for counties with zero agricultural area."""
    county = counties[counties["GEOID"] == geoid].iloc[0]
    centroid = county.geometry.centroid
    counties_proj = counties.to_crs("EPSG:5070")
    county_proj = gpd.GeoSeries([centroid], crs="EPSG:4326").to_crs("EPSG:5070").iloc[0]
    neighbors = counties_proj[counties_proj.geometry.distance(county_proj) < 11000]
    weights = []
    densities = []
    for _, neighbor in neighbors.iterrows():
        if neighbor["GEOID"] == geoid:
            continue
        neighbor_csv = OUT_DIR / f"county_{neighbor['GEOID']}_stats.csv"
        if neighbor_csv.exists():
            df = pd.read_csv(neighbor_csv)
            density_val = df[(df["year"] == year) & (df["species"] == species)]["weighted_mean"]
            if not density_val.empty and not np.isnan(density_val.iloc[0]):
                dist = county_proj.distance(neighbor.geometry)
                if dist > 0:
                    weights.append(1 / dist)
                    densities.append(density_val.iloc[0])
    if weights:
        return np.average(densities, weights=weights)
    return np.nan

def resample_nlcd_to_livestock(nlcd_src, livestock_src):
    """Resample NLCD raster to match livestock raster resolution."""
    transform, width, height = calculate_default_transform(
        nlcd_src.crs, livestock_src.crs, livestock_src.width, livestock_src.height,
        *nlcd_src.bounds
    )
    nlcd_resampled = np.zeros((livestock_src.height, livestock_src.width), dtype=nlcd_src.dtypes[0])
    reproject(
        source=rasterio.band(nlcd_src, 1),
        destination=nlcd_resampled,
        src_transform=nlcd_src.transform,
        src_crs=nlcd_src.crs,
        dst_transform=transform,
        dst_crs=livestock_src.crs,
        resampling=Resampling.nearest
    )
    return nlcd_resampled, transform

def process_county_chunk(county_chunk, nlcd_raster_path, livestock_raster_paths, counties):
    """Process a chunk of counties with optimized raster handling."""
    results = []

    # --- START of efficiency fix ---
    # Perform expensive raster operations once per chunk.
    # The CRS is consistently EPSG:4326 for all rasters, making this simpler.
    with rasterio.open(nlcd_raster_path) as nlcd_src:
        with rasterio.open(livestock_raster_paths[0]) as liv_src:
            # Check for CRS consistency before resampling
            if nlcd_src.crs != liv_src.crs:
                print(f"Warning: NLCD ({nlcd_src.crs}) and livestock ({liv_src.crs}) CRSs don't match. Reprojecting NLCD.")

            nlcd_resampled, nlcd_transform_resampled = resample_nlcd_to_livestock(nlcd_src, liv_src)
            nlcd_bounds = nlcd_src.bounds

    # --- END of efficiency fix ---

    for _, county in county_chunk.iterrows():
        gc.collect()
        geoid = county["GEOID"]
        geometry_4326 = county["geometry"]

        # Use the CRS from the resampled NLCD/livestock raster for consistency.
        county_geom_proj = gpd.GeoSeries([geometry_4326], crs="EPSG:4326").to_crs(rasterio.open(livestock_raster_paths[0]).crs).iloc[0]
        county_bounds_proj = county_geom_proj.bounds

        usable_area_km2 = 0.0

        if bounds_overlap(county_bounds_proj, nlcd_bounds):
            try:
                # Use the resampled NLCD data and its transform
                county_window = from_bounds(*county_bounds_proj, nlcd_transform_resampled)
                clamped_window = intersection(county_window, from_bounds(*nlcd_bounds, nlcd_transform_resampled))

                if clamped_window.width > 0 and clamped_window.height > 0:
                    chunk = nlcd_resampled[
                        int(clamped_window.row_off):int(clamped_window.row_off + clamped_window.height),
                        int(clamped_window.col_off):int(clamped_window.col_off + clamped_window.width)
                    ]

                    centroid_lat = county_geom_proj.centroid.y
                    # Correct cell area calculation
                    cell_area_km2 = (nlcd_transform_resampled.a * 111.32 * np.cos(np.radians(centroid_lat))) * (abs(nlcd_transform_resampled.e) * 111.32)

                    exclude_mask = (
                        np.isin(chunk, LAND_USE_CLASSES["water_bodies"]) |
                        np.isin(chunk, LAND_USE_CLASSES["high_pop"]) |
                        np.isin(chunk, LAND_USE_CLASSES["other_exclusions"])
                    )

                    agri_cells = (
                        np.isin(chunk, LAND_USE_CLASSES["cropland"]) |
                        np.isin(chunk, LAND_USE_CLASSES["grassland"]) |
                        np.isin(chunk, LAND_USE_CLASSES["pasture"])
                    ) & ~exclude_mask

                    usable_area_km2 = np.sum(agri_cells) * cell_area_km2

                    # More robust check for usable area
                    if usable_area_km2 < 0.1 or usable_area_km2 > county["area_km2"] * 1.5:
                        print(f"Warning: Unusual usable area ({usable_area_km2:.2f} km²) for county {geoid}, using full county area.")
                        usable_area_km2 = county["area_km2"]

            except (ValueError, rasterio.errors.RasterioIOError) as e:
                print(f"Error processing NLCD for county {geoid}: {e}. Falling back to total area.")
                usable_area_km2 = county["area_km2"]
        else:
            usable_area_km2 = county["area_km2"]

        stats = []
        for tif_path in livestock_raster_paths:
            gc.collect()
            with rasterio.open(tif_path) as src:
                year = re.search(r"(\d{4})", Path(tif_path).name).group(1)
                species_code = extract_species_from_filename(Path(tif_path).name)
                species = code2name.get(species_code, species_code.lower())
                nodata = src.nodata if src.nodata is not None else np.nan

                # Check for overlap and process
                weighted_mean, std_dev, std_error = np.nan, np.nan, np.nan
                if bounds_overlap(county_bounds_proj, src.bounds):
                    try:
                        # Use rio_mask for more accurate clipping
                        out_image, out_transform = rio_mask(src, [county_geom_proj], crop=True, all_touched=True)
                        if out_image.size > 0:
                            weighted_mean, std_dev, std_error = compute_weighted_mean(
                                out_image[0], nodata, county_geom_proj, out_transform, county["area_km2"]
                            )
                    except (ValueError, rasterio.errors.RasterioIOError) as e:
                        print(f"Error processing {species} raster for county {geoid}: {e}")
                        weighted_mean = np.nan

                # Handle zero agricultural area with IDW
                if usable_area_km2 < 0.1 and not np.isnan(weighted_mean):
                    weighted_mean = compute_idw_density(geoid, species, year, counties, livestock_raster_paths)
                    std_dev, std_error = np.nan, np.nan

                area_for_calc = usable_area_km2 if usable_area_km2 > 0.1 else county["area_km2"]
                head_count = weighted_mean * area_for_calc if not np.isnan(weighted_mean) else np.nan

                stats.append({
                    "GEOID": geoid, "year": year, "species": species,
                    "weighted_mean": weighted_mean, "std_dev": std_dev, "std_error": std_error,
                    "head_count": head_count, "area_usable_km2": usable_area_km2
                })

        df = pd.DataFrame(stats)
        out_path = OUT_DIR / f"county_{geoid}_stats.csv"
        df.to_csv(out_path, index=False)
        results.append(out_path)
    return results

def combine_csvs(out_paths):
    """Combine all county CSV files into a single CSV."""
    if not out_paths:
        print("No county stats files were generated to combine.")
        return
    combined_df = pd.concat([pd.read_csv(f) for f in out_paths], ignore_index=True)
    combined_df.to_csv(COMBINED_OUT, index=False)
    print(f"\n✅ Combined CSV saved to: {COMBINED_OUT}")

def main_processing(livestock_raster_paths, counties, chunk_size=5):
    """Orchestrate parallel processing of all counties."""
    out_paths = []
    county_chunks = [counties[i:i + chunk_size] for i in range(0, len(counties), chunk_size)]

    with ProcessPoolExecutor(max_workers=CORES) as executor:
        futures = {
            executor.submit(
                process_county_chunk, chunk, NLCD_RASTER, livestock_raster_paths, counties
            ): chunk for chunk in county_chunks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            out_paths.extend(future.result())

    print(f"\n✅ Saved {len(out_paths)} county CSVs to: {OUT_DIR}")
    combine_csvs(out_paths)
    return out_paths

if __name__ == "__main__":
    print("Loading counties...")
    counties = load_counties()
    print("Finding raster file paths...")
    livestock_raster_paths = [
        str(f) for f in sorted(DATA_DIR.glob("*.tif"))
        if "NLCD" not in f.name and "pixel_area" not in f.name
    ]
    if not livestock_raster_paths:
        print("No livestock raster files found. Exiting.")
    else:
        print(f"Found {len(livestock_raster_paths)} livestock rasters.")
        chunk_size = CORES * 2
        print(f"Using a chunk size of {chunk_size} counties.")
        main_processing(livestock_raster_paths, counties, chunk_size)


# ==========================================
# 3C Feature engineering for the interpolated livestock dataset. This code reads the wide-format CSV, computes new features such as usable area percentage, growth rates, and aggregated statistics, and saves the engineered dataset to a new CSV file.
# ==========================================
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
DATA_DIR = Path("/content/drive/MyDrive/NewLiv/data")
input_file = DATA_DIR / "interpolated_livestock_wide.csv"
output_file = DATA_DIR / "interpolated_livestock_wide_engineered.csv"

# Load the wide-format CSV
df = pd.read_csv(input_file, na_values=['', 'NA', '-'])

def engineer_features_fast(df):
    # Dynamically extract species from column names
    area_cols = [col for col in df.columns if col.startswith('area_usable_km2_')]
    species = [col.replace('area_usable_km2_', '') for col in area_cols]

    # Compute usable_area_percent for each species (vectorized)
    # The np.where logic correctly handles division by zero.
    for species_name in species:
        area_col = f'area_usable_km2_{species_name}'
        percent_col = f'usable_area_percent_{species_name}'
        if area_col in df.columns and 'total_area_km2' in df.columns:
            df[percent_col] = np.where(df['total_area_km2'] != 0, (df[area_col] / df['total_area_km2']) * 100, np.nan)

    # Compute growth rates for weighted_mean and head_count (vectorized)
    # This replaces the slow loops with a groupby and vectorized pct_change.
    df = df.sort_values(by=['GEOID', 'year']).set_index(['GEOID', 'year'])

    for species_name in species:
        wm_col = f'weighted_mean_{species_name}'
        hc_col = f'head_count_{species_name}'

        # Calculate percent change grouped by GEOID
        if wm_col in df.columns:
            df[f'growth_rate_wm_{species_name}'] = df.groupby(level='GEOID')[wm_col].pct_change().mul(100)

        if hc_col in df.columns:
            df[f'growth_rate_hc_{species_name}'] = df.groupby(level='GEOID')[hc_col].pct_change().mul(100)

    # Compute aggregated features (vectorized)
    # Sum and mean are already vectorized functions in pandas.
    wm_cols = [f'weighted_mean_{s}' for s in species if f'weighted_mean_{s}' in df.columns]
    hc_cols = [f'head_count_{s}' for s in species if f'head_count_{s}' in df.columns]

    if wm_cols:
        df['total_weighted_mean'] = df[wm_cols].sum(axis=1)
        df['avg_weighted_mean'] = df[wm_cols].mean(axis=1)

    if hc_cols:
        df['total_head_count'] = df[hc_cols].sum(axis=1)

    return df.reset_index()

# Apply the fast feature engineering
engineered_df = engineer_features_fast(df.copy()) # Use a copy to avoid chained assignment warnings

# Save the result
engineered_df.to_csv(output_file, index=False)
print(f"Engineered features saved to {output_file} at {datetime.now().strftime('%I:%M %p CDT on %B %d, %Y')}")

# ==========================================
# 3D Interpolation of livestock density for missing county-year-species combinations. This code reads the long-format CSV, identifies missing combinations, performs linear interpolation based on available data, and saves the interpolated dataset to a new CSV file.
# ==========================================
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import geopandas as gpd

# Configuration
DATA_DIR = Path("/content/drive/MyDrive/NewLiv/data")
input_file = DATA_DIR / "combined_livestock_stats.csv"
output_file = DATA_DIR / "interpolated_livestock_wide.csv"
county_shp = DATA_DIR / "cb_2023_us_county_500k.shp"

# Load county data for total area
counties_gdf = gpd.read_file(county_shp)
counties_gdf = counties_gdf[["GEOID", "ALAND", "AWATER"]].copy()

# --- FIX: Standardize GEOID in shapefile data to a 5-digit string ---
counties_gdf["GEOID"] = counties_gdf["GEOID"].astype(str).str.zfill(5)

counties_gdf["total_area_km2"] = (counties_gdf["ALAND"] + counties_gdf["AWATER"]) / 1e6
counties_gdf = counties_gdf.drop(columns=["ALAND", "AWATER"])
county_area_map = counties_gdf.set_index('GEOID')['total_area_km2'].to_dict()

# Load the combined CSV
df = pd.read_csv(input_file, na_values=['', 'NA', '-'])

# --- FIX: Standardize GEOID in the CSV data to a 5-digit string ---
df["GEOID"] = df["GEOID"].astype(str).str.zfill(5)

# Define the year range for interpolation
start_year = 2003
end_year = 2020
years = range(start_year, end_year + 1)

# Function to interpolate values for a given GEOID and species
def interpolate_group(group):
    full_index = pd.Index(years, name='year')
    full_df = group.set_index('year').reindex(full_index)

    full_df['GEOID'] = group['GEOID'].iloc[0]
    full_df['species'] = group['species'].iloc[0]

    if 'area_usable_km2' in group.columns and not group['area_usable_km2'].empty:
        full_df['area_usable_km2'] = group['area_usable_km2'].iloc[0]

    cols_to_interpolate = ['weighted_mean', 'std_dev', 'std_error', 'head_count']

    for col in cols_to_interpolate:
        if col in full_df.columns:
            if 2010 in full_df.index and not pd.isna(full_df.loc[2010, col]):
                val_2010 = full_df.loc[2010, col]
                full_df.loc[start_year:2010, col] = full_df.loc[start_year:2010, col].fillna(val_2010)

            full_df[col] = full_df[col].interpolate(method='linear', limit_area='inside')

    return full_df.reset_index()

# Step 1: Perform interpolation on the long-format data
interpolated_long_df = df.groupby(['GEOID', 'species'], group_keys=False).apply(interpolate_group).reset_index(drop=True)

# Step 2: Pivot the interpolated data to achieve the wide format
interpolated_wide_df = interpolated_long_df.pivot_table(
    index=['GEOID', 'year'],
    columns='species',
    values=['weighted_mean', 'std_dev', 'std_error', 'head_count', 'area_usable_km2']
).reset_index()

# Step 3: Flatten the multi-level columns
interpolated_wide_df.columns = [f'{col[0]}_{col[1]}' if col[0] in ['weighted_mean', 'std_dev', 'std_error', 'head_count', 'area_usable_km2'] else col[0] for col in interpolated_wide_df.columns]

# Step 4: Add the total_area_km2 column to the wide DataFrame
# This will now work correctly after standardizing the GEOID columns
interpolated_wide_df['total_area_km2'] = interpolated_wide_df['GEOID'].map(county_area_map)

# Reorder columns for readability
cols = ['GEOID', 'year', 'total_area_km2'] + [col for col in interpolated_wide_df.columns if col not in ['GEOID', 'year', 'total_area_km2']]
interpolated_wide_df = interpolated_wide_df[cols]

# Save the final result
interpolated_wide_df.to_csv(output_file, index=False)
print(f"Interpolated wide-format data saved to {output_file} at {datetime.now().strftime('%I:%M %p CDT on %B %d, %Y')}")



# 4 Engineering additional features for the final dataset. This code reads the wide-format CSV, computes 25 new cross-modal and engineered features based on existing columns, and saves the enhanced dataset to a new CSV file.
    # ============================================================================
    # STEP 1: DATA LOADING AND FEATURE ENGINEERING
    # ============================================================================

    print("\n[STEP 1] Loading data and engineering features...")
    start_time = time.time()

    df = pd.read_csv(DATA_FILE)

    # Clean Target
    if 'MeanLifeExpectency_x' in df.columns:
        df['MeanLifeExpectency'] = df['MeanLifeExpectency_x']
        df = df.drop(columns=['MeanLifeExpectency_x', 'MeanLifeExpectency_y'], errors='ignore')

    df['fips'] = df['fips'].astype(str).str.zfill(5)

    def create_engineered_features(data):
        """Generates the 25 cross-modal/engineered features"""
        df_eng = data.copy()
        epsilon = 1e-5
        
        # Missing columns fallback to 0 temporarily for math if needed, but keeping NaNs intact
        df_eng['ENG_TVI'] = df_eng['LST_Night_1km_mean'] * (1 - df_eng.get('NDVI_mean', 0))
        df_eng['ENG_Diurnal_Temp_Range'] = df_eng['LST_Day_1km_mean'] - df_eng['LST_Night_1km_mean']
        df_eng['ENG_Night_Cooling_Eff'] = (df_eng['LST_Day_1km_p90'] - df_eng['LST_Night_1km_p10']) / (df_eng['LST_Day_1km_p90'] + epsilon)
        df_eng['ENG_NDVI_LST_Divergence'] = df_eng.get('NDVI_stdDev', 0) / (df_eng['LST_Day_1km_stdDev'] + epsilon)
        df_eng['ENG_Ag_Greenness_Ratio'] = df_eng.get('NDVI_mean', 0) / (df_eng['USDA_Cropland_USDA_Corn_pct'] + df_eng['USDA_Cropland_USDA_Soybeans_pct'] + epsilon)
        df_eng['ENG_Livestock_Heat_Exposure'] = df_eng.get('sum_cattle', 0) * df_eng['LST_Night_1km_mean']
        df_eng['ENG_Impervious_Heat_Index'] = (df_eng['USDA_Cropland_USDA_Dev_Med_pct'] + df_eng['USDA_Cropland_USDA_Dev_High_pct']) * df_eng['LST_Night_1km_mean']
        df_eng['ENG_Forest_Heat_Buffer'] = df_eng['USDA_Cropland_USDA_Forest_Deciduous_pct'] * np.maximum(0, df_eng['LST_Day_1km_mean'] - 20)
        df_eng['ENG_Soil_Moisture_Deficit'] = np.maximum(0, 6.0 - df_eng.get('Soil_mean', 0))
        df_eng['ENG_Soil_Moisture_Excess'] = np.maximum(0, df_eng.get('Soil_mean', 0) - 8.5)
        df_eng['ENG_Wetland_Flood_Risk'] = (df_eng['USDA_Cropland_USDA_Wetlands_Woody_pct'] + df_eng['USDA_Cropland_USDA_Wetlands_Herbaceous_pct']) * df_eng.get('Soil_mean', 0)
        df_eng['ENG_SAR_NDVI_Struct'] = df_eng.get('S1_VV_mean', 0) / (df_eng.get('NDVI_mean', 0) + epsilon)
        df_eng['ENG_Water_Permanence'] = df_eng['JRC_Water_JRC_Permanent_pct'] / (df_eng['JRC_Water_JRC_Permanent_pct'] + df_eng.get('JRC_Water_JRC_Seasonal_pct', 0) + epsilon)
        df_eng['ENG_Elevation_Thermal_Mod'] = df_eng['LST_Night_1km_mean'] - (df_eng.get('DEM_mean', 0) * 0.0065)
        df_eng['ENG_Topo_Roughness_LST'] = df_eng.get('DEM_stdDev', 0) * df_eng['LST_Day_1km_stdDev']
        
        livestock_cols = ['sum_buffalo', 'sum_cattle', 'sum_chicken', 'sum_goat', 'sum_horse', 'sum_pig', 'sum_sheep']
        valid_livestock = [c for c in livestock_cols if c in df_eng.columns]
        df_eng['ENG_Livestock_Diversity'] = df_eng[valid_livestock].apply(lambda row: np.nan if row.isna().all() else entropy(row.fillna(0) + epsilon), axis=1)
        
        usda_cols = [c for c in df_eng.columns if c.startswith('USDA_Cropland_USDA_') and c.endswith('_pct')]
        df_eng['ENG_Crop_Diversity'] = df_eng[usda_cols].apply(lambda row: np.nan if row.isna().all() else entropy(row.fillna(0) + epsilon), axis=1)
        
        df_eng['ENG_Seasonal_NDVI_Amp'] = df_eng.get('NDVI_p90', 0) - df_eng.get('NDVI_p10', 0)
        df_eng['ENG_Cross_Sensor_NDVI'] = df_eng.get('Landsat_NDVI_mean', 0) - df_eng.get('S2_NDVI_mean', 0)
        df_eng['ENG_Wet_Bulb_Proxy'] = df_eng['LST_Day_1km_mean'] * (1 + df_eng.get('Landsat_NDMI_mean', 0))
        df_eng['ENG_Blue_Green_Proximity'] = (df_eng['USDA_Cropland_USDA_Forest_Evergreen_pct'] + df_eng['USDA_Cropland_USDA_Forest_Deciduous_pct'] + df_eng['USDA_Cropland_USDA_Forest_Mixed_pct'] + df_eng['JRC_Water_JRC_Permanent_pct'] + df_eng['USDA_Cropland_USDA_Dev_Open_pct'])
        df_eng['ENG_Urban_Stressor'] = df_eng['USDA_Cropland_USDA_Dev_High_pct'] / (df_eng['USDA_Cropland_USDA_Dev_Low_pct'] + 1)
        
        other_crops = df_eng.get('USDA_Cropland_USDA_Other Crops_pct', df_eng.get('USDA_Cropland_USDA_Other_Crops_pct', 0))
        sweet_corn = df_eng.get('USDA_Cropland_USDA_Sweet Corn_pct', df_eng.get('USDA_Cropland_USDA_Sweet_Corn_pct', 0))
        df_eng['ENG_Industrial_Ag_Proxy'] = (df_eng['USDA_Cropland_USDA_Corn_pct'] + df_eng['USDA_Cropland_USDA_Soybeans_pct']) / (other_crops + sweet_corn + 1)
        
        usable_area = df_eng.get('area_usable_km2_buffalo', 1)
        df_eng['ENG_Livestock_Pollution_Load'] = (df_eng.get('sum_pig',0) + df_eng.get('sum_chicken',0) + df_eng.get('sum_cattle',0)) / usable_area.replace(0, np.nan)
        df_eng['ENG_Landscape_Complexity'] = 1 / (df_eng.get('USDA_stdDev', 1) + epsilon)

        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
        return df_eng

    df = create_engineered_features(df)
    print(f"  ✓ Added 25 cross-modal features.")

    # Target & Excluded columns
    cols_to_drop = ['MeanLifeExpectency', 'year', 'fips', 'location_name']
    cols_to_drop.extend([c for c in df.columns if c.startswith('count_') or c.startswith('sum_') or c.startswith('area_')])

    # Keep only rows where Target is not null for training
    df_trainable = df.dropna(subset=['MeanLifeExpectency']).copy()

    X_raw = df_trainable.drop(columns=[c for c in cols_to_drop if c in df_trainable.columns], errors='ignore')
    y = df_trainable['MeanLifeExpectency']
    groups = df_trainable['fips']

    # Handle Missing Data
    X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_raw = X_raw.fillna(X_raw.median())
    X_raw = X_raw.clip(lower=X_raw.quantile(0.01), upper=X_raw.quantile(0.99), axis=1)

    print(f"  Pre-pruning dataset: {len(X_raw):,} rows × {X_raw.shape[1]} features")





    #We are using the ALREADY ENGINEERED dataset directly



#This is the collum heading after all the above code from our IHME LE, FAO Livestock and GEE Multimodal dataset including the engineered features. We will use this dataset to train our model and predict LE for all county-year combinations, including those with missing data.

# location_name	fips	year	USDA_mean	USDA_stdDev	USDA_Cropland_USDA_Corn_pct	USDA_Cropland_USDA_Fallow_pct	USDA_Cropland_USDA_Class_88_pct	USDA_Cropland_USDA_Soybeans_pct	USDA_Cropland_USDA_Class_83_pct	USDA_Cropland_USDA_Class_82_pct	USDA_Cropland_USDA_Class_81_pct	USDA_Cropland_USDA_Grassland_pct	USDA_Cropland_USDA_Class_63_pct	USDA_Cropland_USDA_Class_29_pct	USDA_Cropland_USDA_Class_25_pct	USDA_Cropland_USDA_Sunflower_pct	USDA_Cropland_USDA_Alfalfa_pct	USDA_Cropland_USDA_Spring Wheat_pct	USDA_Cropland_USDA_Barley_pct	USDA_Cropland_USDA_Class_22_pct	USDA_Cropland_USDA_Oats_pct	USDA_Cropland_USDA_Other Crops_pct	USDA_Cropland_USDA_Class_42_pct	USDA_Cropland_USDA_Winter Wheat_pct	USDA_Cropland_USDA_Class_47_pct	USDA_Cropland_USDA_Class_87_pct	USDA_Cropland_USDA_Class_53_pct	USDA_Cropland_USDA_Cotton_pct	USDA_Cropland_USDA_Class_92_pct	USDA_Cropland_USDA_Rice_pct	USDA_Cropland_USDA_Other Hay_pct	USDA_Cropland_USDA_Sweet Corn_pct	USDA_Cropland_USDA_Class_43_pct	USDA_Cropland_USDA_Class_27_pct	USDA_Cropland_USDA_Class_41_pct	USDA_Cropland_USDA_Class_57_pct	USDA_Cropland_USDA_Forest_Evergreen_pct	USDA_Cropland_USDA_Dev_Low_pct	USDA_Cropland_USDA_Sorghum_pct	USDA_Cropland_USDA_Wetlands_Herbaceous_pct	USDA_Cropland_USDA_Forest_Deciduous_pct	USDA_Cropland_USDA_Forest_Mixed_pct	USDA_Cropland_USDA_Class_131_pct	USDA_Cropland_USDA_Class_26_pct	USDA_Cropland_USDA_Wetlands_Woody_pct	USDA_Cropland_USDA_Dev_Open_pct	USDA_Cropland_USDA_Dev_Med_pct	USDA_Cropland_USDA_Dev_High_pct	USDA_Cropland_USDA_Class_45_pct	USDA_Cropland_USDA_Class_48_pct	USDA_Cropland_USDA_Class_74_pct	USDA_Cropland_USDA_Class_14_pct	USDA_Cropland_USDA_Class_33_pct	USDA_Cropland_USDA_Class_59_pct	USDA_Cropland_USDA_Class_31_pct	USDA_Cropland_USDA_Class_229_pct	USDA_Cropland_USDA_Class_70_pct	USDA_Cropland_USDA_Class_32_pct	USDA_Cropland_USDA_Class_52_pct	USDA_Cropland_USDA_Class_35_pct	USDA_Cropland_USDA_Class_68_pct	USDA_Cropland_USDA_Class_71_pct	USDA_Cropland_USDA_Open Water_pct	USDA_Cropland_USDA_Shrubland_pct	USDA_Cropland_USDA_Class_58_pct	USDA_Cropland_USDA_Class_67_pct	USDA_Cropland_USDA_Peanuts_pct	USDA_Cropland_USDA_Class_46_pct	USDA_Cropland_USDA_Class_39_pct	USDA_Cropland_USDA_Class_30_pct	USDA_Cropland_USDA_Class_112_pct	USDA_Cropland_USDA_Class_75_pct	USDA_Cropland_USDA_Class_69_pct	USDA_Cropland_USDA_Class_76_pct	USDA_Cropland_USDA_Class_211_pct	USDA_Cropland_USDA_Class_210_pct	USDA_Cropland_USDA_Class_54_pct	USDA_Cropland_USDA_Class_207_pct	USDA_Cropland_USDA_Class_223_pct	USDA_Cropland_USDA_Class_226_pct	USDA_Cropland_USDA_Class_72_pct	USDA_Cropland_USDA_Class_221_pct	USDA_Cropland_USDA_Class_209_pct	USDA_Cropland_USDA_Class_212_pct	USDA_Cropland_USDA_Class_213_pct	USDA_Cropland_USDA_Class_66_pct	USDA_Cropland_USDA_Class_204_pct	USDA_Cropland_USDA_Class_225_pct	USDA_Cropland_USDA_Class_217_pct	USDA_Cropland_USDA_Class_205_pct	USDA_Cropland_USDA_Class_224_pct	USDA_Cropland_USDA_Class_13_pct	USDA_Cropland_USDA_Class_208_pct	USDA_Cropland_USDA_Class_206_pct	USDA_Cropland_USDA_Class_218_pct	USDA_Cropland_USDA_Class_49_pct	USDA_Cropland_USDA_Class_220_pct	USDA_Cropland_USDA_Class_222_pct	USDA_Cropland_USDA_Class_216_pct	USDA_Cropland_USDA_Class_219_pct	USDA_Cropland_USDA_Class_214_pct	USDA_Cropland_USDA_Class_34_pct	USDA_Cropland_USDA_Class_56_pct	USDA_Cropland_USDA_Class_38_pct	USDA_Cropland_USDA_Tobacco_pct	USDA_Cropland_USDA_Class_236_pct	USDA_Cropland_USDA_Class_254_pct	USDA_Cropland_USDA_Class_241_pct	USDA_Cropland_USDA_Class_50_pct	USDA_Cropland_USDA_Class_227_pct	USDA_Cropland_USDA_Class_240_pct	USDA_Cropland_USDA_Class_238_pct	USDA_Cropland_USDA_Class_60_pct	USDA_Cropland_USDA_Class_242_pct	USDA_Cropland_USDA_Class_237_pct	USDA_Cropland_USDA_Class_249_pct	USDA_Cropland_USDA_Class_243_pct	USDA_Cropland_USDA_Class_244_pct	USDA_Cropland_USDA_Class_234_pct	USDA_Cropland_USDA_Class_232_pct	USDA_Cropland_USDA_Class_230_pct	USDA_Cropland_USDA_Class_231_pct	USDA_Cropland_USDA_Class_246_pct	USDA_Cropland_USDA_Class_77_pct	USDA_Cropland_USDA_Class_245_pct	USDA_Cropland_USDA_Class_55_pct	USDA_Cropland_USDA_Class_247_pct	USDA_Cropland_USDA_Class_250_pct	USDA_Cropland_USDA_Class_248_pct	USDA_Cropland_USDA_Class_235_pct	USDA_Cropland_USDA_Class_239_pct	USDA_Cropland_USDA_Class_233_pct	USDA_Cropland_USDA_Class_51_pct	USDA_Cropland_USDA_Class_215_pct	USDA_Cropland_USDA_Class_228_pct	USDA_Cropland_USDA_Class_64_pct	USDA_Cropland_USDA_Class_65_pct	JRC_mean	JRC_stdDev	JRC_Water_JRC_NotWater_pct	JRC_Water_JRC_Permanent_pct	JRC_Water_JRC_Seasonal_pct	JRC_Water_JRC_Class_0_pct	DEM_mean	DEM_p10	DEM_p25	DEM_p50	DEM_p75	DEM_p90	DEM_stdDev	Landsat_BSI_mean	Landsat_BSI_p10	Landsat_BSI_p25	Landsat_BSI_p50	Landsat_BSI_p75	Landsat_BSI_p90	Landsat_BSI_stdDev	Landsat_EVI_mean	Landsat_EVI_p10	Landsat_EVI_p25	Landsat_EVI_p50	Landsat_EVI_p75	Landsat_EVI_p90	Landsat_EVI_stdDev	Landsat_NDMI_mean	Landsat_NDMI_p10	Landsat_NDMI_p25	Landsat_NDMI_p50	Landsat_NDMI_p75	Landsat_NDMI_p90	Landsat_NDMI_stdDev	Landsat_NDVI_mean	Landsat_NDVI_p10	Landsat_NDVI_p25	Landsat_NDVI_p50	Landsat_NDVI_p75	Landsat_NDVI_p90	Landsat_NDVI_stdDev	Landsat_NDWI_mean	Landsat_NDWI_p10	Landsat_NDWI_p25	Landsat_NDWI_p50	Landsat_NDWI_p75	Landsat_NDWI_p90	Landsat_NDWI_stdDev	Landsat_SAVI_mean	Landsat_SAVI_p10	Landsat_SAVI_p25	Landsat_SAVI_p50	Landsat_SAVI_p75	Landsat_SAVI_p90	Landsat_SAVI_stdDev	Landsat_SR_B2_mean	Landsat_SR_B2_p10	Landsat_SR_B2_p25	Landsat_SR_B2_p50	Landsat_SR_B2_p75	Landsat_SR_B2_p90	Landsat_SR_B2_stdDev	Landsat_SR_B3_mean	Landsat_SR_B3_p10	Landsat_SR_B3_p25	Landsat_SR_B3_p50	Landsat_SR_B3_p75	Landsat_SR_B3_p90	Landsat_SR_B3_stdDev	Landsat_SR_B4_mean	Landsat_SR_B4_p10	Landsat_SR_B4_p25	Landsat_SR_B4_p50	Landsat_SR_B4_p75	Landsat_SR_B4_p90	Landsat_SR_B4_stdDev	Landsat_SR_B5_mean	Landsat_SR_B5_p10	Landsat_SR_B5_p25	Landsat_SR_B5_p50	Landsat_SR_B5_p75	Landsat_SR_B5_p90	Landsat_SR_B5_stdDev	Landsat_SR_B6_mean	Landsat_SR_B6_p10	Landsat_SR_B6_p25	Landsat_SR_B6_p50	Landsat_SR_B6_p75	Landsat_SR_B6_p90	Landsat_SR_B6_stdDev	Landsat_SR_B7_mean	Landsat_SR_B7_p10	Landsat_SR_B7_p25	Landsat_SR_B7_p50	Landsat_SR_B7_p75	Landsat_SR_B7_p90	Landsat_SR_B7_stdDev	Soil_mean	Soil_p10	Soil_p25	Soil_p50	Soil_p75	Soil_p90	Soil_stdDev	S1_VH_mean	S1_VH_p10	S1_VH_p25	S1_VH_p50	S1_VH_p75	S1_VH_p90	S1_VH_stdDev	S1_VV_asm_mean	S1_VV_asm_p10	S1_VV_asm_p25	S1_VV_asm_p50	S1_VV_asm_p75	S1_VV_asm_p90	S1_VV_asm_stdDev	S1_VV_contrast_mean	S1_VV_contrast_p10	S1_VV_contrast_p25	S1_VV_contrast_p50	S1_VV_contrast_p75	S1_VV_contrast_p90	S1_VV_contrast_stdDev	S1_VV_corr_mean	S1_VV_corr_p10	S1_VV_corr_p25	S1_VV_corr_p50	S1_VV_corr_p75	S1_VV_corr_p90	S1_VV_corr_stdDev	S1_VV_ent_mean	S1_VV_ent_p10	S1_VV_ent_p25	S1_VV_ent_p50	S1_VV_ent_p75	S1_VV_ent_p90	S1_VV_ent_stdDev	S1_VV_mean	S1_VV_p10	S1_VV_p25	S1_VV_p50	S1_VV_p75	S1_VV_p90	S1_VV_stdDev	S2_B11_mean	S2_B11_p10	S2_B11_p25	S2_B11_p50	S2_B11_p75	S2_B11_p90	S2_B11_stdDev	S2_B12_mean	S2_B12_p10	S2_B12_p25	S2_B12_p50	S2_B12_p75	S2_B12_p90	S2_B12_stdDev	S2_B2_mean	S2_B2_p10	S2_B2_p25	S2_B2_p50	S2_B2_p75	S2_B2_p90	S2_B2_stdDev	S2_B3_mean	S2_B3_p10	S2_B3_p25	S2_B3_p50	S2_B3_p75	S2_B3_p90	S2_B3_stdDev	S2_B4_mean	S2_B4_p10	S2_B4_p25	S2_B4_p50	S2_B4_p75	S2_B4_p90	S2_B4_stdDev	S2_B8_mean	S2_B8_p10	S2_B8_p25	S2_B8_p50	S2_B8_p75	S2_B8_p90	S2_B8_stdDev	S2_BSI_mean	S2_BSI_p10	S2_BSI_p25	S2_BSI_p50	S2_BSI_p75	S2_BSI_p90	S2_BSI_stdDev	S2_EVI_mean	S2_EVI_p10	S2_EVI_p25	S2_EVI_p50	S2_EVI_p75	S2_EVI_p90	S2_EVI_stdDev	S2_NDMI_mean	S2_NDMI_p10	S2_NDMI_p25	S2_NDMI_p50	S2_NDMI_p75	S2_NDMI_p90	S2_NDMI_stdDev	S2_NDVI_mean	S2_NDVI_p10	S2_NDVI_p25	S2_NDVI_p50	S2_NDVI_p75	S2_NDVI_p90	S2_NDVI_stdDev	S2_NDWI_mean	S2_NDWI_p10	S2_NDWI_p25	S2_NDWI_p50	S2_NDWI_p75	S2_NDWI_p90	S2_NDWI_stdDev	S2_SAVI_mean	S2_SAVI_p10	S2_SAVI_p25	S2_SAVI_p50	S2_SAVI_p75	S2_SAVI_p90	S2_SAVI_stdDev	LST_Day_1km_mean	LST_Day_1km_p10	LST_Day_1km_p25	LST_Day_1km_p50	LST_Day_1km_p75	LST_Day_1km_p90	LST_Day_1km_stdDev	LST_Night_1km_mean	LST_Night_1km_p10	LST_Night_1km_p25	LST_Night_1km_p50	LST_Night_1km_p75	LST_Night_1km_p90	LST_Night_1km_stdDev	NDVI_EVI_mean	NDVI_EVI_p10	NDVI_EVI_p25	NDVI_EVI_p50	NDVI_EVI_p75	NDVI_EVI_p90	NDVI_EVI_stdDev	NDVI_mean	NDVI_p10	NDVI_p25	NDVI_p50	NDVI_p75	NDVI_p90	NDVI_stdDev	MeanLifeExpectency	count_buffalo	count_cattle	mean_buffalo	mean_cattle	mean_chicken	mean_duck	mean_goat	mean_horse	mean_pig	mean_sheep	sum_buffalo	sum_cattle	sum_chicken	sum_goat	sum_horse	sum_pig	sum_sheep	area_usable_km2_buffalo	head_count_cattle	head_count_chicken	head_count_goat	head_count_horse	head_count_sheep	std_dev_cattle	std_dev_chicken	std_dev_duck	std_dev_goat	std_dev_horse	std_dev_pig	std_dev_sheep	std_error_cattle	std_error_goat	std_error_sheep	usable_area_percent_buffalo	growth_rate_wm_buffalo	growth_rate_wm_cattle	growth_rate_wm_chicken	growth_rate_wm_duck	growth_rate_wm_goat	growth_rate_wm_horse	growth_rate_wm_pig	growth_rate_wm_sheep	ENG_TVI	ENG_Diurnal_Temp_Range	ENG_Night_Cooling_Eff	ENG_NDVI_LST_Divergence	ENG_Ag_Greenness_Ratio	ENG_Livestock_Heat_Exposure	ENG_Impervious_Heat_Index	ENG_Forest_Heat_Buffer	ENG_Soil_Moisture_Deficit	ENG_Soil_Moisture_Excess	ENG_Wetland_Flood_Risk	ENG_SAR_NDVI_Struct	ENG_Water_Permanence	ENG_Elevation_Thermal_Mod	ENG_Topo_Roughness_LST	ENG_Livestock_Diversity	ENG_Crop_Diversity	ENG_Seasonal_NDVI_Amp	ENG_Cross_Sensor_NDVI	ENG_Wet_Bulb_Proxy	ENG_Blue_Green_Proximity	ENG_Urban_Stressor	ENG_Industrial_Ag_Proxy	ENG_Livestock_Pollution_Load	ENG_Landscape_Complexity
# Autauga County (Alabama)	1001	2000			0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	2.4110150646494	0.8244678606829210	0.1741693418616120	0.5871251020886750	0.2377352082608940	0.0009703477888199																																																																																												7.927099963107340	7.000000000000000	7.000000000000000	7.000000000000000	9.0	9.0	1.1005415425548800																																																																																																																															24.860289895913400	23.79770958766030	24.234012641845900	24.73793319276310	25.39593187857660	26.11024517732050	0.8922821460003820	13.590786322942500	12.960363169316300	13.2269732329579	13.554937427830200	13.914163516939900	14.270777802093600	0.5244791721424150	4001.7607020854400	3376.8677483586300	3696.451291331820	4015.372512847540	4335.825866775520	4623.346262701250	517.4623699237000	6870.76187649268	5649.146691946910	6317.955049009600	7023.097415213080	7568.768393238630	7887.12472442422	877.523660810202																																												-93365.46575290820	11.269503572970900	0.5036290116163360	983.4488230604980	687076187.6492680		0.0	0.0	0.0	0.0	0.0		0.7117786817176870				4.890349128221750	2237.978032477310			0.5871251020886750	0.0	0.0		

