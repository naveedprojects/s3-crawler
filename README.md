# CloudNest S3 Manager

A professional PyQt6-based application for managing AWS S3 buckets with CloudNest branding.

## Features

- Connect to S3 buckets using AWS credentials
- Browse bucket contents with folder navigation
- Download files and folders with real-time progress tracking
- Delete files and folders from S3
- Parallel download management based on connection speed
- Detailed progress monitoring with transfer rates and ETAs

## Recent Updates

### Download Progress Improvements (2023-07-15)

- Simplified download progress UI with a cleaner, more compact single progress bar per file
- Fixed real-time progress reporting with more frequent updates
- Added special handling for extremely long ETAs (shows "∞" for ETAs longer than 24 hours)
- Fixed incorrect progress calculation that was showing 0% until completion
- Improved download speed reporting to show accurate speeds during active downloads
- Enhanced UI responsiveness with forced UI updates
- Added detailed logging to track download progress

## Requirements

- Python 3.8+
- PyQt6
- boto3
- humanize

## Installation

1. Clone the repository
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

```
python main.py
```

Or use the provided shell script:

```
./run_s3_gui.sh
```

## Credits

CloudNest S3 Manager GUI 
