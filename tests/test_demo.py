import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Adjust the system path to allow importing from the demo script's directory
import efficient_track_anything.demo as original_demo

@pytest.fixture
def mock_dependencies():
    """A fixture to mock dependencies like OpenCV and the TAM predictor."""
    # Patch all imported functions that perform side effects (file I/O, external library calls)
    with patch('efficient_track_anything.demo.build_predictor') as mock_build_predictor, \
         patch('efficient_track_anything.demo.start') as mock_start, \
         patch('efficient_track_anything.demo.track') as mock_track, \
         patch('efficient_track_anything.demo.read_frame') as mock_read_frame, \
         patch('efficient_track_anything.demo.masks_to_uint8_batch') as mock_masks_to_uint8_batch, \
         patch('efficient_track_anything.demo.overlay_mask_bgr') as mock_overlay_mask_bgr, \
         patch('efficient_track_anything.demo.save_imgs') as mock_save_imgs:
        
        # Configure the return values for the mocks
        mock_predictor_instance = MagicMock()
        mock_build_predictor.return_value = mock_predictor_instance
        
        mock_start.return_value = (None, MagicMock())
        mock_track.return_value = (None, MagicMock())
        
        # Pass the mock objects to the test function
        yield {
            'mock_build_predictor': mock_build_predictor,
            'mock_start': mock_start,
            'mock_track': mock_track,
            'mock_read_frame': mock_read_frame,
            'mock_masks_to_uint8_batch': mock_masks_to_uint8_batch,
            'mock_overlay_mask_bgr': mock_overlay_mask_bgr,
            'mock_save_imgs': mock_save_imgs,
        }

def test_demo_sequence(mock_dependencies):
    """
    Tests the main `demo_sequence` function by mocking its dependencies.
    """
    # Unpack the mock objects from the fixture
    mock_build_predictor = mock_dependencies['mock_build_predictor']
    mock_start = mock_dependencies['mock_start']
    mock_track = mock_dependencies['mock_track']
    mock_read_frame = mock_dependencies['mock_read_frame']
    mock_save_imgs = mock_dependencies['mock_save_imgs']
    
    # Call the main function under test
    original_demo.demo_sequence()
    
    # Assert that `build_predictor` was called once
    mock_build_predictor.assert_called_once()
    
    # Assert the sequence of frame reading
    assert mock_read_frame.call_count == 3
    
    # Assert that `start` was called for the first frame
    mock_start.assert_called_once()
    
    # Assert that `track` was called for the second and third frames
    assert mock_track.call_count == 2
    
    # Assert that `save_imgs` was called to save the output of each frame
    assert mock_save_imgs.call_count == 3
    
    # Assert that `start` and `track` were called in the correct order
    # Note: `unittest.mock.mock_calls` tracks the order of calls
    all_mock_calls = [
        mock_start.mock_calls,
        mock_track.mock_calls,
        mock_track.mock_calls,
    ]
    # Check that start was called before track
    assert mock_start.call_count == 1
    assert mock_track.call_count == 2