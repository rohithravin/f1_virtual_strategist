import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from f1_virtual_strategist.data_engine.ingestion import TelemetryIngestionEngine

@pytest.fixture
def mock_telemetry_data():
    """Provides a sample of raw telemetry data for testing."""
    return pd.DataFrame({
        'Distance': [0.0, 5.0, 10.0, 15.0, 15.0, 20.0],  # Includes a duplicate to test dropping
        'RPM': [10000, 12000, 14000, 15000, 15000, 16000],  # 16000 should clip to 1.0
        'Speed': [200, 250, 300, 350, 350, 450],  # 450 should clip to 1.0
        'Throttle': [50, 80, 100, 104, 104, 0],  # 104 should clip to 1.0
        'Brake': [False, False, False, True, True, True],
        'nGear': [6, 7, 8, 8, 8, 7]
    })

@pytest.fixture
def ingestion_engine(tmp_path):
    """Returns an instance of TelemetryIngestionEngine with a temporary cache directory."""
    cache_dir = tmp_path / "dummy_cache"
    engine = TelemetryIngestionEngine(cache_dir=str(cache_dir))
    return engine

def test_normalization(ingestion_engine, mock_telemetry_data):
    """Test that features are correctly normalized via Fixed Scaling."""
    normalized_df = ingestion_engine._normalize_features(mock_telemetry_data.copy())
    
    # Brake tests
    assert (normalized_df['Brake'].isin([0.0, 1.0])).all()
    assert normalized_df['Brake'].iloc[0] == 0.0
    assert normalized_df['Brake'].iloc[3] == 1.0
    
    # Throttle tests
    assert normalized_df['Throttle'].max() <= 1.0
    assert normalized_df['Throttle'].iloc[0] == 0.5
    assert normalized_df['Throttle'].iloc[2] == 1.0
    assert normalized_df['Throttle'].iloc[3] == 1.0 # The 104 error value should be clipped to 1.0
    
    # RPM tests
    assert normalized_df['RPM'].max() <= 1.0
    assert normalized_df['RPM'].iloc[5] == 1.0 # 16000 should clip to 1.0
    
    # Speed tests
    assert normalized_df['Speed'].max() <= 1.0
    assert normalized_df['Speed'].iloc[5] == 1.0 # 450 should clip to 1.0
    
    # nGear tests
    assert pd.api.types.is_integer_dtype(normalized_df['nGear'])
    assert normalized_df['nGear'].max() == 8

def test_spatial_resampling(ingestion_engine, mock_telemetry_data):
    """Test that cubic spline and previous-neighbor interpolation work over distance."""
    # First clean the data like `process_driver_lap` does
    df = mock_telemetry_data.drop_duplicates(subset=['Distance'], keep='first').copy()
    df = df[df['Distance'] > df['Distance'].cummax().shift().fillna(-1)].copy()
    
    df = ingestion_engine._normalize_features(df)
    
    # Resample with interval of 2.0 meters
    resampled_df = ingestion_engine._resample_spatial(df, interval_meters=2.0)
    
    # Distance checks
    assert resampled_df['Distance'].iloc[0] == 0.0
    assert resampled_df['Distance'].iloc[1] == 2.0
    assert resampled_df['Distance'].iloc[-1] <= 20.0
    
    # Check that continuous variables exist and are bounded
    for col in ['RPM', 'Speed', 'Throttle']:
        assert resampled_df[col].max() <= 1.0
        assert resampled_df[col].min() >= 0.0
        
    # Check that categorical variables exist and maintain discrete values
    assert (resampled_df['Brake'].isin([0.0, 1.0])).all()
    assert pd.api.types.is_integer_dtype(resampled_df['nGear'])
    
@pytest.mark.parametrize("start_distance,length_meters,expected_steps", [
    (10.0, 30.0, 15),     # Normal extraction
    (0.0, 100.0, 50),     # Full extraction
    (50.0, 10.0, 5),      # Small extraction
    (0.0, None, 49),      # None length defaults to max_dist - start_dist = 98 - 0 = 98 / 2 = 49
    (20.0, None, 39),     # None length with start > 0: max_dist = 98 - 20 = 78 / 2 = 39
])
def test_extract_micro_sector(ingestion_engine, start_distance, length_meters, expected_steps):
    """Test extracting a specific length micro-sector tensor from resampled data."""
    # Create a fake resampled lap (100 meters long, 2 meter intervals)
    resampled_df = pd.DataFrame({
        'Distance': np.arange(0, 100, 2.0),
        'Throttle': np.ones(50),
        'Brake': np.zeros(50),
        'RPM': np.ones(50) * 0.5,
        'Speed': np.ones(50) * 0.5,
        'nGear': np.ones(50, dtype=int) * 5
    })
    
    tensor = ingestion_engine.extract_micro_sector(
        resampled_df, 
        start_distance=start_distance, 
        length_meters=length_meters, 
        interval_meters=2.0
    )
    
    # Check shape
    assert tensor.shape == (expected_steps, 5)
    
@pytest.mark.parametrize("start_distance,length_meters,expected_steps", [
    (0.0, 40.0, 20),      # Padding required (20m available, 40m requested)
    (10.0, 30.0, 15),     # Padding required (10m available after start, 30m requested)
    (18.0, 10.0, 5),      # Padding required (2m available after start, 10m requested)
])
def test_extract_micro_sector_padding(ingestion_engine, start_distance, length_meters, expected_steps):
    """Test extracting a micro-sector that goes past the end of the data, ensuring it pads."""
    resampled_df = pd.DataFrame({
        'Distance': np.arange(0, 20, 2.0), # only 20m of data (10 steps)
        'Throttle': np.ones(10),
        'Brake': np.zeros(10),
        'RPM': np.ones(10) * 0.5,
        'Speed': np.ones(10) * 0.5,
        'nGear': np.ones(10, dtype=int) * 5
    })
    
    tensor = ingestion_engine.extract_micro_sector(
        resampled_df, 
        start_distance=start_distance, 
        length_meters=length_meters, 
        interval_meters=2.0
    )
    
    assert tensor.shape == (expected_steps, 5)

def test_process_driver_lap_with_mocking(ingestion_engine, mock_telemetry_data):
    """Test process_driver_lap by mocking the FastF1 session to avoid network calls."""
    # Create mock session and laps
    mock_session = MagicMock()
    mock_laps = MagicMock()
    
    # Create a mock lap that has LapNumber 15
    mock_lap = MagicMock()
    # We must mock getting telemetry to return our mock dataframe
    mock_lap.get_telemetry.return_value = mock_telemetry_data
    
    # Rebuild a proper mock for session.laps.pick_drivers
    mock_driver_laps = MagicMock()
    mock_driver_laps.empty = False
    
    # When filtering by lap_subset = laps[laps['LapNumber'] == lap_number]
    mock_lap_subset = MagicMock()
    mock_lap_subset.empty = False
    mock_lap_subset.iloc = [mock_lap]
    
    # Mock the boolean indexing: mock_driver_laps[...] returns mock_lap_subset
    mock_driver_laps.__getitem__.return_value = mock_lap_subset
    
    mock_session.laps.pick_drivers.return_value = mock_driver_laps
    
    # Assign the mock session to our engine
    ingestion_engine.session = mock_session
    
    # Run the function
    resampled_df = ingestion_engine.process_driver_lap('NOR', 15, distance_interval=5.0)
    
    # Assertions
    mock_session.laps.pick_drivers.assert_called_once_with('NOR')
    mock_lap.get_telemetry.assert_called_once()
    
    # Check the resulting dataframe has the expected columns after resampling
    expected_cols = ['Distance', 'RPM', 'Speed', 'Throttle', 'Brake', 'nGear']
    for col in expected_cols:
        assert col in resampled_df.columns
        
    # Check shape/values (mock telemetry distance goes from 0 to 20, interval=5.0 -> 0, 5, 10, 15 = 4 rows since arange stops before max)
    assert len(resampled_df) == 4
    assert resampled_df['Distance'].iloc[-1] == 15.0
    
def test_process_driver_lap_exceptions(ingestion_engine):
    """Test process_driver_lap raises correct exceptions."""
    # 1. Session not loaded
    with pytest.raises(ValueError, match="Session not loaded"):
        ingestion_engine.process_driver_lap('NOR', 1)
        
    # Mock session for remaining tests
    ingestion_engine.session = MagicMock()
    
    # 2. No laps found for driver
    empty_laps = MagicMock()
    empty_laps.empty = True
    ingestion_engine.session.laps.pick_drivers.return_value = empty_laps
    
    with pytest.raises(ValueError, match="No laps found for driver NOR"):
        ingestion_engine.process_driver_lap('NOR', 1)
        
    # 3. Lap not found
    valid_laps = MagicMock()
    valid_laps.empty = False
    
    empty_subset = MagicMock()
    empty_subset.empty = True
    valid_laps.__getitem__.return_value = empty_subset
    
    ingestion_engine.session.laps.pick_drivers.return_value = valid_laps
    
    with pytest.raises(ValueError, match="Lap 1 not found for driver NOR"):
        ingestion_engine.process_driver_lap('NOR', 1)

def test_main_function(tmp_path, capsys):
    """Test the main() function by mocking sys.argv."""
    from f1_virtual_strategist.data_engine.ingestion import main
    import sys
    
    csv_out = tmp_path / "main_output.csv"
    
    # Create fake CLI arguments
    test_args = [
        "--year", "2023",
        "--event", "Silverstone",
        "--driver_identifier", "NOR",
        "--lap", "15",
        "--interval", "5.0",
        "--micro_start", "1500.0",
        "--micro_length", "500.0",
        "--csv_output", str(csv_out)
    ]
    
    # Call main with the arguments explicitly passed
    main(test_args)
    
    # Capture output
    captured = capsys.readouterr()
    
    # Validate stdout logs
    assert "Initializing The Interaction Data Engine..." in captured.out
    assert "Extracting Micro-Sector (Start: 1500.0m, Length: 500.0m)..." in captured.out
    assert "Micro-Sector Tensor Shape: (100, 5)" in captured.out
    
    # Check that CSV was created
    assert csv_out.exists(), "CSV output file was not created by main()"
    
@pytest.mark.parametrize("bad_args", [
    ["--year", "2023", "--event", "Silverstone", "--driver_identifier", "NOR", "--lap", "999"], # Invalid lap
    ["--year", "1950", "--event", "Silverstone", "--driver_identifier", "NOR"], # Invalid year/no data
    ["--year", "2050", "--event", "Silverstone", "--driver_identifier", "NOR"], # Future year (avoids auto-correct)
    ["--year", "2023", "--event", "Silverstone", "--driver_identifier", "INVALID_DRIVER"], # Invalid driver
])
def test_main_function_exception(capsys, bad_args):
    """Test the main() function handles exceptions correctly by forcing a failure."""
    from f1_virtual_strategist.data_engine.ingestion import main
    import pytest
    
    # The main function has a sys.exit(1) on failure, so we must catch SystemExit
    with pytest.raises(SystemExit) as exc_info:
        main(bad_args)
        
    assert exc_info.value.code == 1

def test_process_driver_lap_exceptions(ingestion_engine):
    """Test process_driver_lap raises correct exceptions."""
    # 1. Session not loaded
    with pytest.raises(ValueError, match="Session not loaded"):
        ingestion_engine.process_driver_lap('NOR', 1)
        
    # Mock session for remaining tests
    ingestion_engine.session = MagicMock()
    
    # 2. No laps found for driver
    empty_laps = MagicMock()
    empty_laps.empty = True
    ingestion_engine.session.laps.pick_drivers.return_value = empty_laps
    
    with pytest.raises(ValueError, match="No laps found for driver NOR"):
        ingestion_engine.process_driver_lap('NOR', 1)
        
    # 3. Lap not found
    valid_laps = MagicMock()
    valid_laps.empty = False
    
    empty_subset = MagicMock()
    empty_subset.empty = True
    valid_laps.__getitem__.return_value = empty_subset
    
    ingestion_engine.session.laps.pick_drivers.return_value = valid_laps
    
    with pytest.raises(ValueError, match="Lap 1 not found for driver NOR"):
        ingestion_engine.process_driver_lap('NOR', 1)

@pytest.mark.parametrize("year,event,driver,lap,interval,micro_start,micro_length", [
    ("2023", "Silverstone", "NOR", "15", "5.0", "1500.0", "500.0"),
    ("2024", "Monza", "VER", "10", "3.0", "0.0", "300.0"),
    ("2023", "Monaco", "16", "30", "10.0", "2000.0", "1000.0"),
])
def test_integration_script(tmp_path, year, event, driver, lap, interval, micro_start, micro_length):
    """Test the full ingestion.py script end-to-end via subprocess with various parameters."""
    import subprocess
    import os
    
    # We will output the CSV to a temporary directory
    csv_out = tmp_path / f"test_output_{year}_{event}_{driver}.csv"
    
    # Run the script with parameterized arguments
    cmd = [
        "python", "src/f1_virtual_strategist/data_engine/ingestion.py",
        "--year", year,
        "--event", event,
        "--lap", lap,
        "--interval", interval,
        "--micro_start", micro_start,
        "--micro_length", micro_length,
        "--csv_output", str(csv_out)
    ]
    
    # Add driver flag depending on whether it's an identifier (e.g. NOR) or number (e.g. 16)
    if driver.isdigit():
        cmd.extend(["--driver_number", driver])
    else:
        cmd.extend(["--driver_identifier", driver])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Ensure script executed successfully
    assert result.returncode == 0, f"Script failed with output:\n{result.stderr}\n{result.stdout}"
    
    # Check that CSV was created
    assert csv_out.exists(), "CSV output file was not created by the script"
    
    # Validate CSV contents
    df = pd.read_csv(csv_out)
    expected_cols = ['Distance', 'RPM', 'Speed', 'Throttle', 'Brake', 'nGear']
    for col in expected_cols:
        assert col in df.columns, f"Column {col} missing from output CSV"
        
    # Check minimum row count (depends on track, but >100 is safe for any F1 track)
    assert len(df) > 100, "Resampled lap CSV is suspiciously short"
    
    # Ensure distance is strictly increasing in the CSV
    assert df['Distance'].is_monotonic_increasing, "Distance in final CSV is not strictly increasing"
    
    # Check that the interval is approximately correct between each row
    distances = df['Distance'].values
    intervals = np.diff(distances)
    assert np.allclose(intervals, float(interval), atol=1e-3), f"Distance interval is not consistently {interval} meters"
    
    # Verify that Gears are integers strictly between 0 and 8
    assert df['nGear'].between(0, 8).all(), "Gears are out of bounds [0, 8]"
    assert pd.api.types.is_integer_dtype(df['nGear']), "Gears are not exact integers"
    
    # Verify no missing data
    assert df.isnull().sum().sum() == 0, "Resampled dataframe contains NaN values"
    
    # Check that Brake is strictly boolean representation (0.0 or 1.0)
    assert df['Brake'].isin([0.0, 1.0]).all(), "Brake must be strictly 0.0 or 1.0"
    
    # Verify bounds for continuous normalized features
    for col in ['Throttle', 'RPM', 'Speed']:
        assert df[col].between(0.0, 1.0).all(), f"{col} must be bounded between 0.0 and 1.0"
        
    # Verify lap starts at or near 0
    assert df['Distance'].iloc[0] < float(interval) * 2, "First distance value should be near 0"
    
    # Validate stdout logs for micro-sector extraction
    assert f"Extracting Micro-Sector (Start: {float(micro_start)}m, Length: {float(micro_length)}m)..." in result.stdout
    expected_steps = int(float(micro_length) / float(interval))
    assert f"Micro-Sector Tensor Shape: ({expected_steps}, 5)" in result.stdout
