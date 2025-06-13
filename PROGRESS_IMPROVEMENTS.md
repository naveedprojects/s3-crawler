# CloudNest S3 Manager - Progress Bar and ETA Fixes

## Issues Fixed

### 1. Progress Bar Jumping from 0% to 100%
**Problem**: Progress updates were too infrequent, causing the progress bar to jump directly from 0% to 100% for small files.

**Solution**: 
- Improved progress callback logic with more frequent updates
- Added smooth animation when progress jumps significantly (>15%)
- Better update conditions based on time intervals, percentage changes, and byte thresholds
- Added intermediate steps for visual smoothness

### 2. ETA (Estimated Time) Calculation Issues
**Problem**: ETA was showing "∞" (infinity) for even very small files due to poor speed calculation and edge case handling.

**Solution**:
- Completely rewrote the `format_eta()` method with better logic
- Proper handling of edge cases (no speed data, very small files, etc.)
- Improved time formatting (seconds, minutes, hours)
- Better threshold handling for when to show "∞"

### 3. Download Speed Reporting Issues
**Problem**: Speed was often showing "0 B/s" during active downloads.

**Solution**:
- Improved transfer rate calculation with exponential moving average
- Better smoothing factor (0.3) for more responsive speed updates
- Removed artificial minimum speed enforcement that was causing issues

## Key Changes Made

### Progress Callback Logic (`SingleFileDownloader.run()`)
```python
def progress_callback(bytes_transferred):
    # Calculate progress with float precision
    progress_float = (bytes_transferred / self.task.size) * 100
    progress = min(int(progress_float), 100)
    
    # Better update conditions
    should_update = (
        progress != last_emitted_progress or 
        time_diff >= 0.2 or 
        bytes_diff >= max(1024, self.task.size * 0.001) or
        progress >= 95 or
        (progress == 0 and last_emitted_progress == 0 and bytes_transferred > 0)
    )
    
    # Exponential moving average for speed smoothing
    if time_diff > 0 and bytes_diff > 0:
        current_rate = bytes_diff / time_diff
        alpha = 0.3  # Smoothing factor
        self.task.transfer_rate = (alpha * current_rate) + ((1 - alpha) * self.task.transfer_rate)
```

### ETA Calculation (`format_eta()`)
```python
def format_eta(self, bytes_remaining: int, bytes_per_sec: float) -> str:
    if bytes_remaining <= 0:
        return "0s"
    
    if bytes_per_sec <= 0:
        return "calculating..."
    
    seconds_remaining = bytes_remaining / bytes_per_sec
    
    if seconds_remaining < 1:
        return "< 1s"
    elif seconds_remaining < 60:
        return f"{int(seconds_remaining)}s"
    elif seconds_remaining < 3600:
        return f"{int(seconds_remaining / 60)}m"
    elif seconds_remaining < 86400:
        hours = int(seconds_remaining / 3600)
        minutes = int((seconds_remaining % 3600) / 60)
        return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
    else:
        return "∞"
```

### UI Progress Bar Updates (`update_download_progress()`)
```python
if progress_bar:
    current_value = progress_bar.value()
    new_value = progress
    
    # Smooth animation for large jumps
    if new_value - current_value > 15 and current_value < 80:
        steps = min(3, (new_value - current_value) // 5)
        step_size = (new_value - current_value) / (steps + 1)
        
        for i in range(1, steps + 1):
            intermediate = int(current_value + (step_size * i))
            progress_bar.setValue(intermediate)
            QApplication.processEvents()
            time.sleep(0.02)
    
    progress_bar.setValue(new_value)
    progress_bar.setFormat(f"{progress}%")
    progress_bar.repaint()
```

## Testing

Created `test_progress_logic.py` to verify:
- ✅ ETA calculation for various file sizes and speeds
- ✅ Progress percentage calculation accuracy
- ✅ Transfer rate formatting

## Expected Behavior Now

1. **Smooth Progress**: Progress bar will update smoothly from 0% to 100% with intermediate steps
2. **Accurate ETA**: Shows realistic time estimates (e.g., "5s", "2m", "1h 30m") instead of always "∞"
3. **Responsive Speed**: Shows actual download speeds instead of "0 B/s"
4. **Better Visual Feedback**: Animations and immediate UI updates for better user experience

## Files Modified
- `main.py` - Core logic improvements
- `test_progress_logic.py` - New test file for verification
