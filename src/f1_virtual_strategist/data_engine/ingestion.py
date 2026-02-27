import os
import fastf1
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TelemetryIngestionEngine:
    """The Interaction Data Engine: Spatial Telemetry Pipeline.

    Responsible for downloading F1 session data, normalizing telemetry features,
    and slicing/resampling them spatially (by Track Distance in meters) rather than by time.
    """
    
    def __init__(self, cache_dir: str = 'f1_cache'):
        """Initializes the engine and enables the fastf1 cache.

        Args:
            cache_dir (str, optional): The directory to store fastf1 cache. Defaults to 'f1_cache'.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(self.cache_dir)
        self.session = None

    def load_session(self, year: int, event: str, session_identifier: str = 'R') -> fastf1.core.Session:
        """Loads the F1 session data and loads telemetry.

        Args:
            year (int): The championship year (e.g., 2023).
            event (str): The event name or location (e.g., 'Silverstone').
            session_identifier (str, optional): The session to load. Defaults to 'R' (Race).

        Returns:
            fastf1.core.Session: The loaded fastf1 session object.
        """
        logger.info(f"Loading session: {year} {event} {session_identifier}")
        self.session = fastf1.get_session(year, event, session_identifier)
        self.session.load(telemetry=True, weather=False, messages=False)
        return self.session

    def process_driver_lap(self, driver_identifier: str | int, lap_number: int, distance_interval: float = 3.0) -> pd.DataFrame:
        """Extracts, normalizes, and spatially resamples the complete lap telemetry.

        Args:
            driver_identifier (str | int): The driver's 3-letter identifier (e.g., 'NOR') or number (e.g., 4).
            lap_number (int): The specific lap number to process.
            distance_interval (float, optional): The spatial interval in meters for resampling. Defaults to 3.0.

        Raises:
            ValueError: If the session has not been loaded prior to calling.
            ValueError: If no laps are found for the driver.
            ValueError: If the specified lap number is not found for the driver.

        Returns:
            pd.DataFrame: A spatially resampled DataFrame containing normalized telemetry.
        """
        if self.session is None:
            raise ValueError("Session not loaded. Call load_session() first.")

        # Get the driver's lap
        laps = self.session.laps.pick_drivers(driver_identifier)
        if laps.empty:
            raise ValueError(f"No laps found for driver {driver_identifier}")
            
        lap_subset = laps[laps['LapNumber'] == lap_number]
        if lap_subset.empty:
            raise ValueError(f"Lap {lap_number} not found for driver {driver_identifier}")
            
        lap = lap_subset.iloc[0]
        
        # Get raw telemetry for the lap
        telemetry = lap.get_telemetry()
        
        # 1. Extract core features
        df = telemetry[['Distance', 'RPM', 'Speed', 'Throttle', 'Brake', 'nGear']].copy()
        df = df.dropna().copy()
        
        # Ensure 'Distance' is strictly increasing (critical for Cubic Spline interpolation)
        df = df.drop_duplicates(subset=['Distance'], keep='first').copy()
        
        # Sometimes telemetry has micro-fluctuations where distance goes backwards minimally due to GPS jitter
        # We enforce strictly monotonic increasing distance by comparing against the cumulative maximum
        df = df[df['Distance'] > df['Distance'].cummax().shift().fillna(-1)].copy()
        
        # 2. Normalization
        df = self._normalize_features(df)
        
        # 3. Spatial Resampling (Distance-based slicing)
        resampled_df = self._resample_spatial(df, distance_interval)
        
        return resampled_df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes telemetry features using Fixed Scaling.

        Because we know the absolute physical boundaries of an F1 car, we use fixed scaling 
        to bind all inputs between [0.0, 1.0] to ensure stable neural network training, 
        with the exception of Gear which is kept as a discrete integer.

        Args:
            df (pd.DataFrame): The raw telemetry DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with normalized feature columns.
        """
        # Brake is boolean in fastf1 (True/False or 1/0 as int), map to 1.0/0.0
        df['Brake'] = df['Brake'].astype(float)
        
        # Throttle: Fastf1 provides 0-100, but can be 104 on error. Clip and normalize to [0.0, 1.0]
        df['Throttle'] = df['Throttle'].clip(upper=100.0) / 100.0
        
        # RPM: Standardize against a theoretical hard limiter (15,000 for modern turbo-hybrids)
        MAX_RPM = 15_000.0
        df['RPM'] = (df['RPM'] / MAX_RPM).clip(upper=1.0)
        
        # Speed: Standardize against a theoretical max speed (400 km/h)
        MAX_SPEED = 400.0
        df['Speed'] = (df['Speed'] / MAX_SPEED).clip(upper=1.0)
        
        # Gear: Unnormalized. Kept as absolute integer [0, 8].
        # Neural networks can handle small integer ranges without exploding gradients,
        # and treating gear as an embedding or absolute value later is easier if unscaled.
        df['nGear'] = df['nGear'].astype(int)
        
        return df

    def _resample_spatial(self, df: pd.DataFrame, interval_meters: float) -> pd.DataFrame:
        """Applies interpolation to resample time-based telemetry into exact distance intervals.

        Continuous variables use Cubic Spline interpolation. Categorical variables use 
        previous-neighbor interpolation.

        Args:
            df (pd.DataFrame): The normalized telemetry DataFrame with strictly increasing 'Distance'.
            interval_meters (float): The target spatial interval in meters.

        Returns:
            pd.DataFrame: A spatially resampled DataFrame.
        """
        max_distance = df['Distance'].max()
        min_distance = df['Distance'].min()
        
        # Create uniform distance grid (from 0 to max lap distance)
        # F1 laps start at Distance = 0 (approximately)
        target_distances = np.arange(min_distance, max_distance, interval_meters)
        
        resampled_data = {'Distance': target_distances}
        
        # Continuous variables: use Cubic Spline
        continuous_cols = ['RPM', 'Speed', 'Throttle']
        for col in continuous_cols:
            # Create cubic spline interpolator (natural boundary conditions)
            cs = CubicSpline(df['Distance'], df[col], bc_type='natural')
            
            # Evaluate at target distances
            interpolated = cs(target_distances)
            
            # Clip values to prevent spline overshoot outside physical limits
            if col in ['Throttle', 'RPM']:
                interpolated = np.clip(interpolated, 0.0, 1.0)
            elif col == 'Speed':
                interpolated = np.clip(interpolated, 0.0, None)
                
            resampled_data[col] = interpolated
            
        # Categorical variables (nGear, Brake): use nearest/previous neighbor interpolation
        # A car cannot be in gear 6.5, and Brake is boolean (0 or 1), so we use 'previous' fill
        gear_interpolator = interp1d(df['Distance'], df['nGear'], kind='previous', fill_value='extrapolate')
        resampled_data['nGear'] = np.round(gear_interpolator(target_distances)).astype(int)
        
        brake_interpolator = interp1d(df['Distance'], df['Brake'], kind='previous', fill_value='extrapolate')
        resampled_data['Brake'] = np.round(brake_interpolator(target_distances)).astype(float)
        
        return pd.DataFrame(resampled_data)

    def extract_micro_sector(self, resampled_df: pd.DataFrame, start_distance: float = 0.0, length_meters: float = None, interval_meters: float = 3.0) -> np.ndarray:
        """Extracts a specific micro-sector from a pre-processed lap DataFrame.

        If `length_meters` is not provided, the sector extends from `start_distance` 
        to the end of the lap.

        Args:
            resampled_df (pd.DataFrame): The spatially resampled lap DataFrame.
            start_distance (float, optional): The track distance in meters where the micro-sector begins. Defaults to 0.0.
            length_meters (float, optional): The length of the micro-sector in meters. Defaults to None (full lap).
            interval_meters (float, optional): The spatial interval in meters, used to calculate expected shape. Defaults to 3.0.

        Returns:
            np.ndarray: A PyTorch-ready numpy array of shape (N_steps, Features) 
            containing [Throttle, Brake, RPM, Speed, Gear].
        """
        # Filter for the specific distance window
        if length_meters is None:
            end_distance = resampled_df['Distance'].max()
            length_meters = end_distance - start_distance
        else:
            end_distance = start_distance + length_meters
            
        micro_sector_df = resampled_df[(resampled_df['Distance'] >= start_distance) & (resampled_df['Distance'] < end_distance)]
        
        # Ensure we return the exact number of expected steps
        expected_steps = int(length_meters / interval_meters)
        
        # Convert to tensor-ready numpy array [Throttle, Brake, RPM, Speed, Gear]
        features = micro_sector_df[['Throttle', 'Brake', 'RPM', 'Speed', 'nGear']].values
        
        # Pad or trim to exactly match the expected sequence length
        if len(features) > expected_steps:
            features = features[:expected_steps]
        elif len(features) < expected_steps:
            # If the lap ended before the micro-sector finished, pad with the last known value
            if len(features) == 0:
                features = np.zeros((expected_steps, 5))
            else:
                padding = np.tile(features[-1], (expected_steps - len(features), 1))
                features = np.vstack([features, padding])
                
        return features

def main(args_list=None):
    """Parses command line arguments and runs the ingestion pipeline."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Extract and resample F1 telemetry into spatial micro-sectors.")
    parser.add_argument('--year', type=int, default=2023, help='F1 Season Year (e.g. 2023)')
    parser.add_argument('--event', type=str, default='Silverstone', help='Race Name or Location (e.g. Silverstone)')
    
    # Mutually exclusive group for driver selection
    driver_group = parser.add_mutually_exclusive_group()
    driver_group.add_argument('--driver_identifier', type=str, default='NOR', help='Driver 3-letter identifier (e.g. NOR)')
    driver_group.add_argument('--driver_number', type=int, default=None, help='Driver number (e.g. 4)')
    
    parser.add_argument('--lap', type=int, default=15, help='Lap number to process')
    parser.add_argument('--interval', type=float, default=3.0, help='Distance interval in meters for resampling')
    parser.add_argument('--micro_start', type=float, default=0.0, help='Start distance for micro-sector extraction (0.0 for start of lap)')
    parser.add_argument('--micro_length', type=float, default=None, help='Length of micro-sector in meters. Default is None (extracts full lap from start)')
    parser.add_argument('--csv_output', type=str, default=None, help='Optional filename to save the resampled lap data as CSV (e.g. lap_data.csv)')

    args = parser.parse_args(args_list)
    
    driver = args.driver_number if args.driver_number is not None else args.driver_identifier

    print(f"Initializing The Interaction Data Engine...")
    print(f"Configuration: {args.year} {args.event} (Race) | Driver: {driver} | Lap: {args.lap}")
    
    engine = TelemetryIngestionEngine()
    
    try:
        # 1. Load the session (Hardcoded to 'R' for Race)
        engine.load_session(args.year, args.event, 'R')
        
        # 2. Resample full lap
        print(f"\nProcessing {driver} - Lap {args.lap} with {args.interval}m spatial resolution...")
        resampled_lap = engine.process_driver_lap(driver, args.lap, distance_interval=args.interval)
        
        print(f"\n✅ Full Lap Resampled Shape: {resampled_lap.shape}")
        print(f"✅ Columns: {resampled_lap.columns.tolist()}")
        
        print(f"\nHead (First 5 spatial steps - 0m, {args.interval}m, {args.interval*2}m, {args.interval*3}m, {args.interval*4}m):")
        print(resampled_lap.head())
        
        # Extract Micro-Sector
        # If micro_length is not provided, calculate it from the resampled lap to show in print
        actual_length = args.micro_length if args.micro_length is not None else resampled_lap['Distance'].max() - args.micro_start
        
        print(f"\nExtracting Micro-Sector (Start: {args.micro_start}m, Length: {actual_length:.1f}m)...")
        micro_sector_tensor = engine.extract_micro_sector(resampled_lap, 
                                                        start_distance=args.micro_start, 
                                                        length_meters=args.micro_length,
                                                        interval_meters=args.interval)
        
        expected_steps = int(actual_length / args.interval)
        print(f"✅ Micro-Sector Tensor Shape: {micro_sector_tensor.shape} (Expected: {expected_steps} steps x 5 features)")
        print("✅ Ready for 1D-CNN Encoder.")
        
        # Save to CSV if requested
        if args.csv_output:
            resampled_lap.to_csv(args.csv_output, index=False)
            print(f"✅ Saved full resampled lap to {args.csv_output}")
            
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()