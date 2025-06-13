#!/usr/bin/env python3
"""
CloudNest S3 Manager GUI
A professional PyQt6-based application for managing AWS S3 buckets with CloudNest branding
"""

import sys
import os
import platform
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import humanize

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTreeWidget, QTreeWidgetItem,
    QProgressBar, QTextEdit, QMessageBox, QSplitter, QFrame,
    QCheckBox, QScrollArea, QGridLayout, QTabWidget, QStatusBar,
    QDialog, QDialogButtonBox, QComboBox, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation,
    QEasingCurve, QRect, QSize
)
from PyQt6.QtGui import (
    QIcon, QPixmap, QFont, QPalette, QColor, QLinearGradient,
    QPainter, QBrush, QPen
)


# CloudNest S3 Manager - Dark Theme AWS-inspired color palette
CLOUDNEST_COLORS = {
    'primary': '#1F2937',          # Charcoal/Gray-900 (Primary Background)
    'primary_dark': '#111827',     # Darker charcoal for cards
    'accent': '#FF9900',           # AWS Orange (Accent)
    'accent_hover': '#D97706',     # Darker orange for hover
    'accent_dark': '#D97706',      # Alias for darker orange (hover state)
    'background': '#1F2937',       # Charcoal background
    'surface': '#111827',          # Card background (slightly lighter charcoal)
    'surface_alt': '#374151',      # Alternative surface color (gray-700)
    'input_border': '#4B5563',     # Input border color (gray-600)
    'input_focus': '#FF9900',      # Input focus border (accent orange)
    'border_focus': '#6B7280',     # Focus border color (gray-500)
    'text_primary': '#F9FAFB',     # Off-white text (gray-50)
    'text_secondary': '#D1D5DB',   # Light gray text (gray-300)
    'text_tertiary': '#9CA3AF',    # Placeholder text (gray-400)
    'text_error': '#F87171',       # Light red for errors (red-400)
    'overlay': 'rgba(0,0,0,0.6)',  # Modal overlay (darker for dark theme)
    'shadow': 'rgba(0,0,0,0.4)',   # Drop shadow for dark theme
    'success': '#10B981',          # Modern green
    'success_light': '#065F46',    # Dark green background
    'danger': '#EF4444',           # Modern red
    'danger_light': '#7F1D1D',     # Dark red background
    'warning': '#F59E0B',          # Modern amber
    'warning_light': '#78350F',    # Dark amber background
    'info': '#3B82F6',            # Modern blue
    'info_light': '#1E3A8A',      # Dark blue background
    'header_gradient_start': '#FF9900',  # AWS Orange for header gradient
    'header_gradient_end': '#0073BB',    # AWS Blue for header gradient
}

# AWS Regions
AWS_REGIONS = {
    'us-east-1': 'US East (N. Virginia)',
    'us-east-2': 'US East (Ohio)',
    'us-west-1': 'US West (N. California)',
    'us-west-2': 'US West (Oregon)',
    'ca-central-1': 'Canada (Central)',
    'eu-west-1': 'Europe (Ireland)',
    'eu-west-2': 'Europe (London)',
    'eu-west-3': 'Europe (Paris)',
    'eu-central-1': 'Europe (Frankfurt)',
    'eu-north-1': 'Europe (Stockholm)',
    'ap-northeast-1': 'Asia Pacific (Tokyo)',
    'ap-northeast-2': 'Asia Pacific (Seoul)',
    'ap-northeast-3': 'Asia Pacific (Osaka)',
    'ap-southeast-1': 'Asia Pacific (Singapore)',
    'ap-southeast-2': 'Asia Pacific (Sydney)',
    'ap-south-1': 'Asia Pacific (Mumbai)',
    'sa-east-1': 'South America (São Paulo)',
    'af-south-1': 'Africa (Cape Town)',
    'me-south-1': 'Middle East (Bahrain)',
    'ap-east-1': 'Asia Pacific (Hong Kong)',
    'eu-south-1': 'Europe (Milan)'
}


@dataclass
class S3Object:
    """Represents an S3 object or folder"""
    key: str
    size: int
    last_modified: datetime
    is_folder: bool
    selected: bool = False


class LoginWorker(QThread):
    """Worker thread for login attempts"""
    
    login_success = pyqtSignal(object, str)  # s3_client, bucket_name
    login_error = pyqtSignal(str)  # error message
    login_complete = pyqtSignal()  # signal to restore button state
    
    def __init__(self, bucket_name: str, access_key: str, secret_key: str, region: str):
        super().__init__()
        self.bucket_name = bucket_name
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
    
    def run(self):
        try:
            print(f"DEBUG: Testing connection to bucket '{self.bucket_name}' in region '{self.region}'")
            print(f"DEBUG: Access Key: {self.access_key[:8]}...{self.access_key[-4:]}")
            print(f"DEBUG: Secret Key: {'*' * 8}...{self.secret_key[-4:]}")
            
            # Create S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
            
            # Test connection by checking bucket access
            print("DEBUG: Checking bucket access...")
            s3_client.head_bucket(Bucket=self.bucket_name)
            print("DEBUG: Connection successful!")
            
            # Success
            self.login_success.emit(s3_client, self.bucket_name)
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            print(f"DEBUG: ClientError - {error_code}: {e.response['Error']['Message']}")
            
            if error_code in ['NoSuchBucket', '404']:
                self.login_error.emit(f"Bucket '{self.bucket_name}' does not exist in region '{self.region}'")
            elif error_code == 'AccessDenied':
                self.login_error.emit("Access denied. Please check your credentials")
            else:
                self.login_error.emit(f"AWS Error: {e.response['Error']['Message']}")
        except NoCredentialsError:
            print("DEBUG: NoCredentialsError")
            self.login_error.emit("Invalid credentials")
        except Exception as e:
            print(f"DEBUG: Unexpected error: {str(e)}")
            self.login_error.emit(f"Connection failed: {str(e)}")
        finally:
            self.login_complete.emit()


class BucketContentWorker(QThread):
    """Worker thread for loading S3 bucket contents with lazy loading and pagination"""
    
    objects_loaded = pyqtSignal(list, bool, str)  # List of S3Object, has_more, next_token
    load_error = pyqtSignal(str)  # error message
    
    def __init__(self, s3_client, bucket_name: str, prefix: str = "", max_keys: int = 100, continuation_token: str = None):
        super().__init__()
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.max_keys = max_keys
        self.continuation_token = continuation_token
    
    def run(self):
        try:
            print(f"DEBUG: BucketContentWorker: Loading contents for prefix '{self.prefix}' (max {self.max_keys}, token: {self.continuation_token[:20] + '...' if self.continuation_token else 'None'})")
            
            # Prepare list_objects_v2 parameters
            list_params = {
                'Bucket': self.bucket_name,
                'Prefix': self.prefix,
                'Delimiter': '/',
                'MaxKeys': self.max_keys
            }
            
            # Add continuation token if provided (for pagination)
            if self.continuation_token:
                list_params['ContinuationToken'] = self.continuation_token
            
            # List objects with delimiter to get folders at current level
            response = self.s3_client.list_objects_v2(**list_params)
            
            objects = []
            
            # Add folders (common prefixes)
            if 'CommonPrefixes' in response:
                for prefix_info in response['CommonPrefixes']:
                    folder_key = prefix_info['Prefix']
                    objects.append(S3Object(
                        key=folder_key,
                        size=0,
                        last_modified=datetime.now(),
                        is_folder=True
                    ))
                    print(f"DEBUG: Found folder: {folder_key}")
            
            # Add files in current directory
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    # Skip if this is just the folder itself
                    if key == self.prefix:
                        continue
                        
                    # Only include files directly in current directory (not in subdirectories)
                    relative_key = key[len(self.prefix):]
                    if '/' not in relative_key:
                        objects.append(S3Object(
                            key=key,
                            size=obj['Size'],
                            last_modified=obj['LastModified'],
                            is_folder=False
                        ))
                        print(f"DEBUG: Found file: {key}")
            
            # Check if there are more objects to load (pagination)
            has_more = response.get('IsTruncated', False)
            next_token = response.get('NextContinuationToken', '')
            
            print(f"DEBUG: BucketContentWorker: Loaded {len(objects)} objects, has_more: {has_more}, next_token: {'Yes' if next_token else 'No'}")
            self.objects_loaded.emit(objects, has_more, next_token)
            
        except Exception as e:
            print(f"DEBUG: BucketContentWorker: Error loading contents: {str(e)}")
            self.load_error.emit(f"Error loading contents: {str(e)}")


class NotificationWidget(QLabel):
    """Custom notification widget with fade animation and modern design"""
    
    def __init__(self, message: str, notification_type: str = "success", parent=None):
        super().__init__(parent)
        self.setText(message)
        
        # Choose color based on type
        colors = {
            "success": (CLOUDNEST_COLORS['success'], CLOUDNEST_COLORS['success_light']),
            "error": (CLOUDNEST_COLORS['danger'], CLOUDNEST_COLORS['danger_light']),
            "info": (CLOUDNEST_COLORS['info'], CLOUDNEST_COLORS['info_light']),
            "warning": (CLOUDNEST_COLORS['warning'], CLOUDNEST_COLORS['warning_light'])
        }
        
        bg_color, border_color = colors.get(notification_type, colors["success"])
        
        self.setStyleSheet(f"""
            QLabel {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {bg_color}, stop:1 {border_color});
                color: white;
                padding: 16px 24px;
                border-radius: 12px;
                font-weight: 600;
                font-size: 14px;
                border: 2px solid {border_color};
            }}
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedHeight(60)
        self.setMinimumWidth(300)
        
        # Smooth animation
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(4000)
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.finished.connect(self.hide)
        
        # Auto-hide timer
        QTimer.singleShot(800, self.start_fade)
    
    def start_fade(self):
        self.animation.start()


@dataclass
class DownloadTask:
    """Represents a download task with progress tracking"""
    key: str
    filename: str
    size: int
    local_path: str
    start_time: Optional[float] = None
    bytes_downloaded: int = 0
    last_update_time: float = 0
    last_bytes: int = 0
    transfer_rate: float = 0.0  # bytes per second
    is_complete: bool = False
    is_error: bool = False


class SingleFileDownloader(QThread):
    """Worker thread for downloading a single file with detailed progress"""
    
    progress_updated = pyqtSignal(str, int, str, str, str, str)  # filename, progress, status, rate, size, eta
    download_completed = pyqtSignal(str, str)  # filename, local_path
    error_occurred = pyqtSignal(str, str)  # filename, error_message
    speed_measured = pyqtSignal(str, float)  # filename, speed_bytes_per_sec
    
    def __init__(self, s3_client, bucket_name: str, task: DownloadTask):
        super().__init__()
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.task = task
        self.is_cancelled = False
    
    def cancel(self):
        self.is_cancelled = True
    
    def format_rate(self, bytes_per_sec: float) -> str:
        """Format transfer rate with appropriate units"""
        if bytes_per_sec < 1024:
            return f"{bytes_per_sec:.0f} B/s"
        elif bytes_per_sec < 1024 * 1024:
            return f"{bytes_per_sec / 1024:.1f} KB/s"
        else:
            return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size with appropriate units"""
        return humanize.naturalsize(size_bytes)
    
    def format_eta(self, bytes_remaining: int, bytes_per_sec: float) -> str:
        """Format estimated time remaining with improved logic"""
        # Handle edge cases
        if bytes_remaining <= 0:
            return "0s"
        
        # If no speed data yet, show calculating
        if bytes_per_sec <= 0:
            return "calculating..."
        
        # Calculate seconds remaining
        seconds_remaining = bytes_remaining / bytes_per_sec
        
        # Handle very small remaining times (less than 1 second)
        if seconds_remaining < 1:
            return "< 1s"
        
        # Handle reasonable time ranges
        if seconds_remaining < 60:
            return f"{int(seconds_remaining)}s"
        elif seconds_remaining < 3600:  # Less than 1 hour
            minutes = int(seconds_remaining / 60)
            return f"{minutes}m"
        elif seconds_remaining < 86400:  # Less than 24 hours
            hours = int(seconds_remaining / 3600)
            minutes = int((seconds_remaining % 3600) / 60)
            if minutes > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{hours}h"
        else:
            # For very long times (>24 hours), just show infinity
            return "∞"
    
    def run(self):
        try:
            self.task.start_time = time.time()
            self.task.last_update_time = self.task.start_time
            self.task.last_bytes = 0
            
            # Create directory if needed
            os.makedirs(os.path.dirname(self.task.local_path), exist_ok=True)
            
            print(f"DEBUG: Starting download of {self.task.filename}, size: {self.task.size} bytes")
            
            # Display initial progress immediately
            self.progress_updated.emit(
                self.task.filename, 0, "Initializing...", 
                "0 B/s", f"0 B / {self.format_size(self.task.size)}", "Calculating..."
            )
            
            # Force UI update immediately
            QApplication.processEvents()
            
            # Track last emitted progress to avoid unnecessary updates
            last_emitted_progress = 0
            total_bytes_downloaded = 0
            
            def progress_callback(bytes_transferred):
                nonlocal last_emitted_progress, total_bytes_downloaded
                
                if self.is_cancelled:
                    return
                
                current_time = time.time()
                total_bytes_downloaded = bytes_transferred
                self.task.bytes_downloaded = bytes_transferred
                
                # Calculate progress percentage - always emit if it changed
                if self.task.size > 0:
                    progress = min(int((bytes_transferred / self.task.size) * 100), 100)
                else:
                    progress = 0
                
                # Calculate transfer rate
                time_diff = current_time - self.task.last_update_time
                bytes_diff = bytes_transferred - self.task.last_bytes
                
                # Force progress updates more frequently for better UI responsiveness
                should_update = (
                    progress != last_emitted_progress or 
                    time_diff >= 0.25 or  # Update every 0.25 seconds (more frequent)
                    bytes_transferred == 0 or  # Always update first call
                    progress in [25, 50, 75, 100]  # Always update key milestones
                )
                
                if should_update:
                    # Calculate speed
                    if time_diff > 0 and bytes_diff > 0:
                        current_rate = bytes_diff / time_diff
                        # Simple smoothing
                        if self.task.transfer_rate > 0:
                            self.task.transfer_rate = (self.task.transfer_rate * 0.7) + (current_rate * 0.3)
                        else:
                            self.task.transfer_rate = current_rate
                    
                    # Update tracking
                    self.task.last_update_time = current_time
                    self.task.last_bytes = bytes_transferred
                    last_emitted_progress = progress
                    
                    # Format information
                    rate_str = self.format_rate(self.task.transfer_rate)
                    size_str = f"{self.format_size(bytes_transferred)} / {self.format_size(self.task.size)}"
                    bytes_remaining = max(0, self.task.size - bytes_transferred)
                    eta_str = self.format_eta(bytes_remaining, self.task.transfer_rate)
                    status = "Downloading..." if progress < 100 else "Completing..."
                    
                    # ALWAYS emit progress - this is critical
                    print(f"DEBUG: Emitting progress: {self.task.filename} -> {progress}% ({bytes_transferred}/{self.task.size} bytes)")
                    self.progress_updated.emit(
                        self.task.filename, progress, status, rate_str, size_str, eta_str
                    )
                    
                    # Emit speed for parallel management
                    self.speed_measured.emit(self.task.filename, self.task.transfer_rate)
            
            # Use CUSTOM download method for complete progress control
            try:
                # Get object metadata to verify size
                response = self.s3_client.head_object(Bucket=self.bucket_name, Key=self.task.key)
                actual_size = response['ContentLength']
                
                # Update task size if it was estimated
                if actual_size != self.task.size:
                    print(f"DEBUG: Updating file size from {self.task.size} to {actual_size}")
                    self.task.size = actual_size
                
                # CUSTOM STREAMING DOWNLOAD with manual progress tracking
                print(f"DEBUG: Starting CUSTOM streaming download, total size: {actual_size}")
                
                # Get the S3 object
                s3_response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.task.key)
                
                # Open file for writing
                with open(self.task.local_path, 'wb') as f:
                    downloaded = 0
                    chunk_size = 8192  # 8KB chunks for frequent updates
                    
                    # Read the stream in chunks
                    while downloaded < actual_size:
                        if self.is_cancelled:
                            break
                        
                        # Calculate remaining bytes
                        remaining = actual_size - downloaded
                        read_size = min(chunk_size, remaining)
                        
                        # Read chunk from S3 stream
                        chunk = s3_response['Body'].read(read_size)
                        if not chunk:
                            break
                            
                        # Write to file
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Call progress callback for EVERY chunk
                        progress_callback(downloaded)
                        
                        # Add small delay to make progress visible for testing
                        time.sleep(0.01)  # 10ms delay
                
                print(f"DEBUG: Custom download completed: {downloaded}/{actual_size} bytes")
                
            except Exception as custom_error:
                print(f"DEBUG: Custom download failed, falling back to standard S3 download: {custom_error}")
                # Fallback to regular S3 download with callback
                self.s3_client.download_file(
                    self.bucket_name, self.task.key, self.task.local_path,
                    Callback=progress_callback
                )
            
            # Mark as complete and emit final progress
            self.task.is_complete = True
            final_rate = self.format_rate(self.task.transfer_rate)
            final_size = self.format_size(self.task.size)
            
            # Always emit 100% at the end to ensure UI shows completion
            self.progress_updated.emit(
                self.task.filename, 100, "Complete", final_rate, final_size, "0s"
            )
            self.download_completed.emit(self.task.filename, self.task.local_path)
            
        except Exception as e:
            print(f"DEBUG: Download error for {self.task.filename}: {str(e)}")
            self.task.is_error = True
            self.error_occurred.emit(self.task.filename, str(e))


class ParallelDownloadManager(QThread):
    """Manages parallel downloads based on internet speed"""
    
    progress_updated = pyqtSignal(str, int, str, str, str, str)  # filename, progress, status, rate, size, eta
    download_completed = pyqtSignal(str, str)  # filename, local_path
    error_occurred = pyqtSignal(str)
    all_completed = pyqtSignal()
    
    def __init__(self, s3_client, bucket_name: str, objects: List[S3Object], download_dir: str):
        super().__init__()
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.objects = objects
        self.download_dir = download_dir
        self.is_cancelled = False
        
        # Download management
        self.download_tasks: List[DownloadTask] = []
        self.active_downloaders: List[SingleFileDownloader] = []
        self.completed_downloads = 0
        self.total_downloads = 0
        
        # Speed monitoring (in bytes per second)
        self.speed_samples: Dict[str, List[float]] = {}
        self.min_speed_threshold = 1024  # 1 KB/s minimum per download
        self.max_concurrent_downloads = 8
        
        # Prepare download tasks
        self._prepare_download_tasks()
    
    def _prepare_download_tasks(self):
        """Prepare all download tasks from objects"""
        for obj in self.objects:
            if obj.is_folder:
                # Get all files in folder
                self._add_folder_tasks(obj.key)
            else:
                # Single file
                self._add_file_task(obj.key, obj.size)
        
        self.total_downloads = len(self.download_tasks)
    
    def _add_file_task(self, key: str, size: int):
        """Add a single file download task"""
        filename = os.path.basename(key) or key
        local_path = os.path.join(self.download_dir, key)
        
        # Small file handling - ensure minimum size for better progress tracking
        # AWS S3 SDK may report 0 for very small files
        effective_size = max(size, 1024) if size < 1024 else size
        
        task = DownloadTask(
            key=key,
            filename=filename,
            size=effective_size,
            local_path=local_path
        )
        self.download_tasks.append(task)
    
    def _add_folder_tasks(self, prefix: str):
        """Add download tasks for all files in a folder"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            for page in pages:
                if self.is_cancelled:
                    break
                    
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if self.is_cancelled:
                            break
                        self._add_file_task(obj['Key'], obj['Size'])
                        
        except Exception as e:
            self.error_occurred.emit(f"Failed to list folder {prefix}: {str(e)}")
    
    def cancel(self):
        """Cancel all downloads"""
        self.is_cancelled = True
        for downloader in self.active_downloaders:
            downloader.cancel()
    
    def _get_average_speed_per_download(self) -> float:
        """Calculate average speed per active download"""
        if not self.active_downloaders:
            return 0.0
        
        total_speed = 0.0
        active_count = 0
        
        for downloader in self.active_downloaders:
            filename = downloader.task.filename
            if filename in self.speed_samples and self.speed_samples[filename]:
                # Average of last 3 samples for smoothing
                recent_samples = self.speed_samples[filename][-3:]
                avg_speed = sum(recent_samples) / len(recent_samples)
                total_speed += avg_speed
                active_count += 1
        
        return total_speed / active_count if active_count > 0 else 0.0
    
    def _should_start_new_download(self) -> bool:
        """Determine if we should start a new parallel download"""
        # Don't exceed max concurrent downloads
        if len(self.active_downloaders) >= self.max_concurrent_downloads:
            return False
        
        # Always start first download
        if len(self.active_downloaders) == 0:
            return True
        
        # Check if current downloads are maintaining good speed
        avg_speed = self._get_average_speed_per_download()
        
        # If average speed per download is above threshold, start another
        return avg_speed > self.min_speed_threshold
    
    def _start_next_download(self):
        """Start the next download in queue"""
        # Find next pending task
        for task in self.download_tasks:
            if not task.start_time and not task.is_complete and not task.is_error:
                downloader = SingleFileDownloader(self.s3_client, self.bucket_name, task)
                
                # Connect signals properly to avoid lambda capture issues
                # This is crucial - use correct parameter forwarding with named parameters
                downloader.progress_updated.connect(self._forward_progress_update)
                downloader.download_completed.connect(self._on_download_completed)
                downloader.error_occurred.connect(self._on_download_error)
                downloader.speed_measured.connect(self._on_speed_measured)
                
                self.active_downloaders.append(downloader)
                downloader.start()
                
                # Log and debugging output when download starts
                print(f"DEBUG: Started download of {task.filename}, size: {task.size} bytes")
                break
    
    def _forward_progress_update(self, filename, progress, status, rate, size, eta):
        """Properly forward progress signals with all parameters"""
        # Debug the forwarding to ensure it's working
        print(f"DEBUG: Forwarding progress: {filename}, {progress}%, {status}, {rate}")
        self.progress_updated.emit(filename, progress, status, rate, size, eta)
    
    def _on_speed_measured(self, filename: str, speed: float):
        """Handle speed measurement from downloader"""
        if filename not in self.speed_samples:
            self.speed_samples[filename] = []
        
        # Keep last 5 samples for smoothing
        self.speed_samples[filename].append(speed)
        if len(self.speed_samples[filename]) > 5:
            self.speed_samples[filename].pop(0)
    
    def _on_download_completed(self, filename: str, local_path: str):
        """Handle individual download completion"""
        # Remove from active downloaders
        self.active_downloaders = [d for d in self.active_downloaders if d.task.filename != filename]
        
        # Clean up speed samples
        if filename in self.speed_samples:
            del self.speed_samples[filename]
        
        self.completed_downloads += 1
        self.download_completed.emit(filename, local_path)
        
        # Check if we can start more downloads or if all are complete
        self._manage_parallel_downloads()
    
    def _on_download_error(self, filename: str, error_message: str):
        """Handle individual download error"""
        # Remove from active downloaders
        self.active_downloaders = [d for d in self.active_downloaders if d.task.filename != filename]
        
        # Clean up speed samples
        if filename in self.speed_samples:
            del self.speed_samples[filename]
        
        self.completed_downloads += 1
        self.error_occurred.emit(f"Failed to download {filename}: {error_message}")
        
        # Continue with other downloads
        self._manage_parallel_downloads()
    
    def _manage_parallel_downloads(self):
        """Manage starting new downloads based on current performance"""
        if self.is_cancelled:
            return
        
        # Check if all downloads are complete
        if self.completed_downloads >= self.total_downloads:
            self.all_completed.emit()
            return
        
        # Start new downloads if conditions are met
        while (self._should_start_new_download() and 
               self.completed_downloads + len(self.active_downloaders) < self.total_downloads):
            self._start_next_download()
    
    def run(self):
        """Main download management loop"""
        try:
            if not self.download_tasks:
                self.all_completed.emit()
                return
            
            # Start initial download
            self._start_next_download()
            
            # Monitor and manage downloads
            while (not self.is_cancelled and 
                   self.completed_downloads < self.total_downloads):
                
                # Check every 2 seconds if we should start more downloads
                self.msleep(2000)
                self._manage_parallel_downloads()
            
        except Exception as e:
            self.error_occurred.emit(f"Download manager error: {str(e)}")


class DeleteWorker(QThread):
    """Worker thread for deleting files from S3"""
    
    progress_updated = pyqtSignal(str, str)  # filename, status
    deletion_completed = pyqtSignal(str)  # filename
    error_occurred = pyqtSignal(str)
    all_completed = pyqtSignal()
    
    def __init__(self, s3_client, bucket_name: str, objects: List[S3Object]):
        super().__init__()
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.objects = objects
    
    def run(self):
        try:
            for obj in self.objects:
                if obj.is_folder:
                    self._delete_folder(obj.key)
                else:
                    self._delete_file(obj.key)
            
            self.all_completed.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"Delete error: {str(e)}")
    
    def _delete_file(self, key: str):
        try:
            filename = os.path.basename(key) or key
            self.progress_updated.emit(filename, "Deleting...")
            
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            self.deletion_completed.emit(filename)
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to delete {key}: {str(e)}")
    
    def _delete_folder(self, prefix: str):
        try:
            # List all objects with this prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            objects_to_delete = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects_to_delete.append({'Key': obj['Key']})
            
            # Delete in batches
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i:i+1000]
                self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={'Objects': batch}
                )
            
            folder_name = prefix.rstrip('/')
            self.deletion_completed.emit(folder_name)
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to delete folder {prefix}: {str(e)}")


class LoginWindow(QMainWindow):
    """AWS S3 Login Window"""
    
    login_successful = pyqtSignal(object, str)  # s3_client, bucket_name
    
    def __init__(self):
        super().__init__()
        self.s3_client = None
        self.login_worker = None
        self.setup_ui()
        self.apply_styles()
    
    def setup_ui(self):
        self.setWindowTitle("CloudNest S3 Manager - Login")
        
        # Responsive window sizing (50% width, 70% height with constraints)
        screen = QApplication.primaryScreen().geometry()
        width = max(480, min(960, int(screen.width() * 0.5)))
        height = max(620, min(1000, int(screen.height() * 0.7)))
        self.resize(width, height)
        self.setMinimumSize(480, 620)
        self.setMaximumSize(960, 1000)
        
        # Create CloudNest icon
        self.setWindowIcon(self.create_cloudnest_icon())
        
        # Center window
        self.center_window()
        
        # Main widget with CloudNest background
        main_widget = QWidget()
        main_widget.setStyleSheet(f"background-color: {CLOUDNEST_COLORS['background']};")
        self.setCentralWidget(main_widget)
        
        # Main layout - center everything with responsive spacing
        main_layout = QVBoxLayout(main_widget)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(30)
        
        # Header Section
        title_label = QLabel("CloudNest S3 Manager")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFixedSize(360, 60)  # Exact dimensions from specs
        main_layout.addWidget(title_label, 0, Qt.AlignmentFlag.AlignHCenter)
        
        # Login Form Card
        form_frame = QFrame()
        form_frame.setObjectName("form_frame")
        form_frame.setFixedWidth(400)  # Centered with 400px width
        form_layout = QVBoxLayout(form_frame)
        form_layout.setSpacing(20)  # Equal vertical spacing
        form_layout.setContentsMargins(30, 30, 30, 30)  # 30px padding as specified
        
        # Field creation helper
        def create_field(label_text, placeholder_text, is_password=False):
            label = QLabel(label_text)
            label.setObjectName("field_label")
            
            if label_text == "AWS Region":
                field = QComboBox()
                field.setObjectName("input_field")
                # Populate region dropdown
                for region_code, region_name in AWS_REGIONS.items():
                    field.addItem(f"{region_name} ({region_code})", region_code)
                field.setCurrentText("US East (N. Virginia) (us-east-1)")
            else:
                field = QLineEdit()
                field.setObjectName("input_field")
                field.setPlaceholderText(placeholder_text)
                if is_password:
                    field.setEchoMode(QLineEdit.EchoMode.Password)
            
            field.setFixedSize(340, 40)  # Exact dimensions: 340x40
            return label, field
        
        # Create all fields
        bucket_label, self.bucket_input = create_field("Bucket Name", "Enter S3 bucket name")
        region_label, self.region_combo = create_field("AWS Region", "")
        access_key_label, self.access_key_input = create_field("Access Key ID", "Enter your AWS access key")
        secret_key_label, self.secret_key_input = create_field("Secret Access Key", "Enter your AWS secret key", True)
        
        # Add all fields to form
        for label, field in [(bucket_label, self.bucket_input), 
                           (region_label, self.region_combo),
                           (access_key_label, self.access_key_input), 
                           (secret_key_label, self.secret_key_input)]:
            form_layout.addWidget(label)
            form_layout.addWidget(field)
        
        # Login button with exact specifications
        form_layout.addSpacing(30)  # 30px top margin
        self.login_button = QPushButton("Login")
        self.login_button.setObjectName("login_button")
        self.login_button.setFixedSize(340, 44)  # Exact dimensions: 340x44
        self.login_button.clicked.connect(self.attempt_login)
        form_layout.addWidget(self.login_button)
        
        main_layout.addWidget(form_frame, 0, Qt.AlignmentFlag.AlignHCenter)
        main_layout.addStretch()
        
        # Connect Enter key to login
        self.secret_key_input.returnPressed.connect(self.attempt_login)
    
    def center_window(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
    
    def create_cloudnest_icon(self) -> QIcon:
        """Create CloudNest S3 bucket icon"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # CloudNest primary color background
        painter.setBrush(QBrush(QColor(CLOUDNEST_COLORS['accent'])))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(4, 4, 24, 24, 4, 4)
        
        # White "S3" text
        painter.setPen(QPen(QColor('white')))
        painter.setFont(QFont('Inter', 8, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "S3")
        
        painter.end()
        return QIcon(pixmap)
    
    def apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {CLOUDNEST_COLORS['background']};
            }}
            
            #title {{
                font-family: "Inter", "Open Sans", sans-serif;
                font-size: 24px;
                font-weight: bold;
                color: black;
                padding: 10px 30px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {CLOUDNEST_COLORS['header_gradient_start']}, stop:1 {CLOUDNEST_COLORS['header_gradient_end']});
                border-radius: 16px;
            }}
            
            #form_frame {{
                background-color: {CLOUDNEST_COLORS['surface']};
                border-radius: 16px;
                border: 1px solid {CLOUDNEST_COLORS['input_border']};
            }}
            
            #field_label {{
                font-family: "Inter", "Open Sans", sans-serif;
                font-size: 14px;
                font-weight: 500;
                color: {CLOUDNEST_COLORS['text_primary']};
                margin-bottom: 5px;
            }}
            
            #input_field {{
                font-family: "Inter", "Open Sans", sans-serif;
                font-size: 16px;
                font-weight: normal;
                padding: 8px 12px;
                border: 1px solid {CLOUDNEST_COLORS['input_border']};
                border-radius: 6px;
                background-color: {CLOUDNEST_COLORS['surface']};
                color: {CLOUDNEST_COLORS['text_primary']};
                selection-background-color: {CLOUDNEST_COLORS['accent']};
            }}
            
            #input_field:focus {{
                border-color: {CLOUDNEST_COLORS['input_focus']};
                outline: none;
            }}
            
            #input_field:hover {{
                border-color: {CLOUDNEST_COLORS['text_secondary']};
            }}
            
            #input_field::placeholder {{
                color: {CLOUDNEST_COLORS['text_tertiary']};
            }}
            
            QComboBox {{
                font-family: "Inter", "Open Sans", sans-serif;
                font-size: 16px;
                font-weight: normal;
                padding: 8px 12px;
                border: 1px solid {CLOUDNEST_COLORS['input_border']};
                border-radius: 6px;
                background-color: {CLOUDNEST_COLORS['surface']};
                color: {CLOUDNEST_COLORS['text_primary']};
            }}
            
            QComboBox:focus {{
                border-color: {CLOUDNEST_COLORS['input_focus']};
            }}
            
            QComboBox:hover {{
                border-color: {CLOUDNEST_COLORS['text_secondary']};
            }}
            
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {CLOUDNEST_COLORS['text_secondary']};
                margin-right: 10px;
            }}
            
            QComboBox QAbstractItemView {{
                border: 1px solid {CLOUDNEST_COLORS['input_border']};
                background-color: {CLOUDNEST_COLORS['surface']};
                selection-background-color: {CLOUDNEST_COLORS['accent']};
                selection-color: white;
                border-radius: 4px;
            }}
            
            #login_button {{
                font-family: "Inter", "Open Sans", sans-serif;
                font-size: 16px;
                font-weight: 600;
                background-color: {CLOUDNEST_COLORS['accent']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
            }}
            
            #login_button:hover {{
                background-color: {CLOUDNEST_COLORS['accent_hover']};
            }}
            
            #login_button:pressed {{
                background-color: {CLOUDNEST_COLORS['accent_dark']};
            }}
            
            #login_button:disabled {{
                background-color: {CLOUDNEST_COLORS['text_tertiary']};
                color: white;
                opacity: 0.5;
            }}
        """)
    
    def show_error(self, message: str):
        """Show error modal with fade in/out animation and CloudNest styling"""
        # Create custom modal dialog
        modal = QDialog(self)
        modal.setWindowTitle("Login Failed")
        modal.setFixedSize(400, 200)
        modal.setModal(True)
        
        # Center on parent
        parent_pos = self.pos()
        parent_size = self.size()
        x = parent_pos.x() + (parent_size.width() - 400) // 2
        y = parent_pos.y() + (parent_size.height() - 200) // 2
        modal.move(x, y)
        
        layout = QVBoxLayout(modal)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Error message
        error_label = QLabel("Invalid credentials or access denied to bucket.")
        error_label.setWordWrap(True)
        error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Details
        details_label = QLabel(message)
        details_label.setWordWrap(True)
        details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Buttons
        button_layout = QHBoxLayout()
        try_again_btn = QPushButton("Try Again")
        try_again_btn.clicked.connect(modal.accept)
        exit_btn = QPushButton("Exit")
        exit_btn.clicked.connect(lambda: (modal.reject(), self.close()))
        
        button_layout.addWidget(try_again_btn)
        button_layout.addWidget(exit_btn)
        
        layout.addWidget(error_label)
        layout.addWidget(details_label)
        layout.addLayout(button_layout)
        
        # Apply CloudNest styling to modal
        modal.setStyleSheet(f"""
            QDialog {{
                background-color: {CLOUDNEST_COLORS['surface']};
                border-radius: 8px;
                border: 2px solid {CLOUDNEST_COLORS['text_error']};
            }}
            QLabel {{
                font-family: "Inter", "Open Sans", sans-serif;
                font-size: 14px;
                color: {CLOUDNEST_COLORS['text_error']};
                font-weight: 500;
            }}
            QPushButton {{
                font-family: "Inter", "Open Sans", sans-serif;
                font-size: 14px;
                font-weight: 600;
                padding: 8px 16px;
                border-radius: 6px;
                border: none;
                min-width: 80px;
            }}
            QPushButton:first-child {{
                background-color: {CLOUDNEST_COLORS['accent']};
                color: white;
            }}
            QPushButton:first-child:hover {{
                background-color: {CLOUDNEST_COLORS['accent_hover']};
            }}
            QPushButton:last-child {{
                background-color: {CLOUDNEST_COLORS['text_secondary']};
                color: white;
            }}
            QPushButton:last-child:hover {{
                background-color: {CLOUDNEST_COLORS['text_primary']};
            }}
        """)
        
        # Add fade-in animation
        modal.setWindowOpacity(0)
        fade_in = QPropertyAnimation(modal, b"windowOpacity")
        fade_in.setDuration(200)
        fade_in.setStartValue(0)
        fade_in.setEndValue(1)
        fade_in.start()
        
        modal.exec()
    
    def attempt_login(self):
        bucket_name = self.bucket_input.text().strip()
        access_key = self.access_key_input.text().strip()
        secret_key = self.secret_key_input.text().strip()
        region = self.region_combo.currentData()  # Get selected region code
        
        if not all([bucket_name, access_key, secret_key]):
            self.show_error("Please fill in all fields")
            return
        
        # Disable login button and show loading
        self.login_button.setText("Connecting...")
        self.login_button.setEnabled(False)
        
        # Create and start login worker thread
        self.login_worker = LoginWorker(bucket_name, access_key, secret_key, region)
        self.login_worker.login_success.connect(self.on_login_success)
        self.login_worker.login_error.connect(self.on_login_error)
        self.login_worker.login_complete.connect(self.on_login_complete)
        self.login_worker.start()
    
    def on_login_success(self, s3_client, bucket_name):
        """Handle successful login"""
        print("DEBUG: Login successful, emitting signal")
        self.login_successful.emit(s3_client, bucket_name)
    
    def on_login_error(self, error_message):
        """Handle login error"""
        print(f"DEBUG: Login error: {error_message}")
        self.show_error(error_message)
    
    def on_login_complete(self):
        """Handle login completion (restore button state)"""
        print("DEBUG: Login attempt complete, restoring button")
        self.login_button.setText("Login")
        self.login_button.setEnabled(True)


class S3BrowserWindow(QMainWindow):
    """Main S3 Browser Window"""
    
    def __init__(self, s3_client, bucket_name: str):
        try:
            print(f"DEBUG: Initializing S3BrowserWindow for bucket: {bucket_name}")
            super().__init__()
            self.s3_client = s3_client
            self.bucket_name = bucket_name
            self.objects: List[S3Object] = []
            self.selected_objects: List[S3Object] = []
            self.download_worker = None
            self.delete_worker = None
            self.content_worker = None
            self.current_prefix = ""  # Track current folder path
            self.navigation_history = []  # For back/forward navigation
            
            # Pagination properties
            self.has_more_objects = False
            self.next_continuation_token = ""
            self.is_loading_more = False
            self.objects_per_page = 100
            
            print("DEBUG: Setting up UI...")
            self.setup_ui()
            print("DEBUG: Applying styles...")
            self.apply_styles()
            print("DEBUG: Loading bucket contents...")
            self.load_bucket_contents()
            print("DEBUG: S3BrowserWindow initialization complete")
        except Exception as e:
            print(f"DEBUG: Error in S3BrowserWindow.__init__: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def setup_ui(self):
        self.setWindowTitle(f"CloudNest S3 Manager - {self.bucket_name}")
        self.setMinimumSize(1200, 800)
        self.setWindowIcon(self.create_aws_icon())
        
        # Center window
        self.center_window()
        
        # Main widget and splitter
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - File browser
        left_panel = self.create_file_browser()
        splitter.addWidget(left_panel)
        
        # Right panel - Progress and info
        right_panel = self.create_progress_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([800, 400])
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f"Connected to bucket: {self.bucket_name}")
    
    def create_file_browser(self) -> QWidget:
        """Create the file browser panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Header with bucket info
        header = QLabel(f"📦 S3 Bucket: {self.bucket_name}")
        header.setObjectName("bucket_header")
        header.setStyleSheet(f"""
            #bucket_header {{
                font-size: 20px;
                font-weight: 700;
                color: {CLOUDNEST_COLORS['text_primary']};
                padding: 16px 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {CLOUDNEST_COLORS['primary']}, stop:1 {CLOUDNEST_COLORS['accent']});
                color: white;
                border-radius: 12px;
                margin-bottom: 8px;
            }}
        """)
        layout.addWidget(header)
        
        # Toolbar
        toolbar_frame = QFrame()
        toolbar_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {CLOUDNEST_COLORS['surface']};
                border: 1px solid {CLOUDNEST_COLORS['input_border']};
                border-radius: 10px;
                padding: 8px;
            }}
        """)
        toolbar = QHBoxLayout(toolbar_frame)
        toolbar.setSpacing(12)
        
        # Navigation buttons
        self.back_btn = QPushButton("⬅️ Back")
        self.back_btn.setObjectName("toolbar_button")
        self.back_btn.clicked.connect(self.navigate_back)
        self.back_btn.setEnabled(False)
        self.back_btn.setMinimumHeight(40)
        toolbar.addWidget(self.back_btn)
        
        self.up_btn = QPushButton("⬆️ Up")
        self.up_btn.setObjectName("toolbar_button")
        self.up_btn.clicked.connect(self.navigate_up)
        self.up_btn.setEnabled(False)
        self.up_btn.setMinimumHeight(40)
        toolbar.addWidget(self.up_btn)
        
        # Current path label
        self.path_label = QLabel("📁 Root")
        self.path_label.setStyleSheet(f"""
            color: {CLOUDNEST_COLORS['text_primary']};
            font-size: 14px;
            font-weight: 600;
            padding: 8px 12px;
            background-color: {CLOUDNEST_COLORS['surface_alt']};
            border-radius: 6px;
        """)
        toolbar.addWidget(self.path_label)
        
        # Add spacing instead of separator
        toolbar.addSpacing(20)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setObjectName("toolbar_button")
        refresh_btn.clicked.connect(self.load_bucket_contents)
        refresh_btn.setMinimumHeight(40)
        toolbar.addWidget(refresh_btn)
        
        # Select all checkbox
        self.select_all_cb = QCheckBox("Select All")
        self.select_all_cb.stateChanged.connect(self.toggle_select_all)
        toolbar.addWidget(self.select_all_cb)
        
        toolbar.addStretch()
        
        # Selection info label
        self.selection_info = QLabel("No items selected")
        self.selection_info.setStyleSheet(f"""
            color: {CLOUDNEST_COLORS['text_secondary']};
            font-size: 13px;
            font-weight: 500;
            padding: 8px 12px;
        """)
        toolbar.addWidget(self.selection_info)
        
        # Action buttons
        self.download_btn = QPushButton("⬇️ Download Selected")
        self.download_btn.setObjectName("action_button")
        self.download_btn.clicked.connect(self.download_selected)
        self.download_btn.setEnabled(False)
        self.download_btn.setMinimumHeight(40)
        
        self.delete_btn = QPushButton("🗑️ Delete Selected")
        self.delete_btn.setObjectName("delete_button")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setMinimumHeight(40)
        
        toolbar.addWidget(self.download_btn)
        toolbar.addWidget(self.delete_btn)
        
        layout.addWidget(toolbar_frame)
        
        # File tree with scroll-based lazy loading
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["📂 Name", "📏 Size", "📅 Modified", "🏷️ Type"])
        self.file_tree.setRootIsDecorated(False)
        self.file_tree.setAlternatingRowColors(True)
        self.file_tree.itemChanged.connect(self.on_item_changed)
        self.file_tree.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.file_tree.setMinimumHeight(400)
        
        # Connect scroll event for lazy loading
        scrollbar = self.file_tree.verticalScrollBar()
        scrollbar.valueChanged.connect(self.on_scroll_changed)
        
        layout.addWidget(self.file_tree)
        
        return panel
    
    def on_scroll_changed(self, value):
        """Handle scroll events for lazy loading"""
        if not self.has_more_objects or self.is_loading_more:
            return
            
        scrollbar = self.file_tree.verticalScrollBar()
        
        # Check if we're near the bottom (90% scrolled)
        if scrollbar.maximum() > 0:
            scroll_percentage = value / scrollbar.maximum()
            if scroll_percentage >= 0.9:
                print(f"DEBUG: Scroll near bottom ({scroll_percentage:.2%}), loading more objects...")
                self.load_bucket_contents(load_more=True)
    
    def create_progress_panel(self) -> QWidget:
        """Create the progress and info panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Title
        title = QLabel("📊 Operations")
        title.setObjectName("panel_title")
        layout.addWidget(title)
        
        # Progress area with better styling
        progress_frame = QFrame()
        progress_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {CLOUDNEST_COLORS['surface']};
                border: 1px solid {CLOUDNEST_COLORS['input_border']};
                border-radius: 12px;
                padding: 12px;
            }}
        """)
        progress_layout = QVBoxLayout(progress_frame)
        
        progress_header = QLabel("📥 Download Progress")
        progress_header.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 600;
            color: {CLOUDNEST_COLORS['text_primary']};
            margin-bottom: 8px;
        """)
        progress_layout.addWidget(progress_header)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMinimumHeight(300)
        
        self.progress_widget = QWidget()
        self.progress_layout = QVBoxLayout(self.progress_widget)
        self.progress_layout.addStretch()
        
        scroll_area.setWidget(self.progress_widget)
        progress_layout.addWidget(scroll_area)
        
        layout.addWidget(progress_frame)
        
        # Status info with better styling
        status_frame = QFrame()
        status_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {CLOUDNEST_COLORS['surface']};
                border: 1px solid {CLOUDNEST_COLORS['input_border']};
                border-radius: 12px;
                padding: 12px;
            }}
        """)
        status_layout = QVBoxLayout(status_frame)
        
        status_header = QLabel("📋 Activity Log")
        status_header.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 600;
            color: {CLOUDNEST_COLORS['text_primary']};
            margin-bottom: 8px;
        """)
        status_layout.addWidget(status_header)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(200)
        self.status_text.setReadOnly(True)
        self.status_text.setObjectName("status_text")
        status_layout.addWidget(self.status_text)
        
        layout.addWidget(status_frame)
        
        return panel
    
    def center_window(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
    
    def create_aws_icon(self) -> QIcon:
        """Create AWS icon (same as login window)"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.setBrush(QBrush(QColor(CLOUDNEST_COLORS['primary'])))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(4, 4, 24, 24, 4, 4)
        
        painter.setPen(QPen(QColor('white')))
        painter.setFont(QFont('Arial', 8, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "S3")
        
        painter.end()
        return QIcon(pixmap)
    
    def apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {CLOUDNEST_COLORS['background']}, stop:1 {CLOUDNEST_COLORS['surface_alt']});
            }}
            
            #panel_title {{
                font-size: 18px;
                font-weight: 700;
                color: {CLOUDNEST_COLORS['text_primary']};
                padding: 16px 20px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {CLOUDNEST_COLORS['surface']}, stop:1 {CLOUDNEST_COLORS['surface_alt']});
                border-radius: 12px;
                margin-bottom: 16px;
                border: 1px solid {CLOUDNEST_COLORS['input_border']};
            }}
            
            #toolbar_button {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {CLOUDNEST_COLORS['accent']}, stop:1 {CLOUDNEST_COLORS['accent_dark']});
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
            }}
            
            #toolbar_button:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {CLOUDNEST_COLORS['accent_dark']}, stop:1 #2A7AB0);
            }}
            
            #action_button {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {CLOUDNEST_COLORS['primary']}, stop:1 {CLOUDNEST_COLORS['primary_dark']});
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
                margin-left: 8px;
            }}
            
            #action_button:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {CLOUDNEST_COLORS['primary_dark']}, stop:1 #0F172A);
            }}
            
            #delete_button {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {CLOUDNEST_COLORS['danger']}, stop:1 #DC2626);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
                margin-left: 8px;
            }}
            
            #delete_button:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #DC2626, stop:1 #B91C1C);
            }}
            
            QCheckBox {{
                font-size: 14px;
                font-weight: 500;
                color: {CLOUDNEST_COLORS['text_primary']};
                spacing: 8px;
            }}
            
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid {CLOUDNEST_COLORS['input_border']};
                background-color: {CLOUDNEST_COLORS['surface']};
            }}
            
            QCheckBox::indicator:checked {{
                background-color: {CLOUDNEST_COLORS['accent']};
                border-color: {CLOUDNEST_COLORS['accent']};
                image: url(data:image/svg+xml;charset=utf8,<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20,6 9,17 4,12"></polyline></svg>);
            }}
            
            QTreeWidget {{
                background-color: {CLOUDNEST_COLORS['surface']};
                border: 1px solid {CLOUDNEST_COLORS['input_border']};
                border-radius: 12px;
                font-size: 14px;
                font-weight: 500;
                gridline-color: {CLOUDNEST_COLORS['input_border']};
                selection-background-color: {CLOUDNEST_COLORS['accent']};
                alternate-background-color: {CLOUDNEST_COLORS['surface_alt']};
                color: {CLOUDNEST_COLORS['text_primary']};
            }}
            
            QTreeWidget::item {{
                padding: 12px 8px;
                border-bottom: 1px solid {CLOUDNEST_COLORS['input_border']};
                color: {CLOUDNEST_COLORS['text_primary']};
            }}
            
            QTreeWidget::item:hover {{
                background-color: {CLOUDNEST_COLORS['surface_alt']};
            }}
            
            QTreeWidget::item:selected {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {CLOUDNEST_COLORS['accent']}, stop:1 {CLOUDNEST_COLORS['accent_dark']});
                color: white;
                border-radius: 6px;
            }}
            
            QHeaderView::section {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {CLOUDNEST_COLORS['surface']}, stop:1 {CLOUDNEST_COLORS['surface_alt']});
                color: {CLOUDNEST_COLORS['text_primary']};
                padding: 12px 8px;
                border: none;
                border-bottom: 2px solid {CLOUDNEST_COLORS['input_border']};
                font-weight: 600;
                font-size: 14px;
            }}
            
            #status_text {{
                background-color: {CLOUDNEST_COLORS['surface']};
                border: 1px solid {CLOUDNEST_COLORS['input_border']};
                border-radius: 12px;
                font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', 'Source Code Pro', monospace;
                font-size: 12px;
                padding: 12px;
                color: {CLOUDNEST_COLORS['text_secondary']};
            }}
            
            QProgressBar {{
                border: none;
                border-radius: 8px;
                text-align: center;
                font-weight: 600;
                font-size: 12px;
                background-color: {CLOUDNEST_COLORS['surface_alt']};
                color: {CLOUDNEST_COLORS['text_primary']};
                height: 20px;
            }}
            
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {CLOUDNEST_COLORS['success']}, stop:1 #059669);
                border-radius: 8px;
            }}
            
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            
            QScrollBar:vertical {{
                background-color: {CLOUDNEST_COLORS['surface_alt']};
                width: 12px;
                border-radius: 6px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {CLOUDNEST_COLORS['border_focus']};
                border-radius: 6px;
                min-height: 20px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {CLOUDNEST_COLORS['text_tertiary']};
            }}
            
            #filename_label {{
                font-weight: 600;
                color: {CLOUDNEST_COLORS['text_primary']};
                font-size: 13px;
            }}
            
            #status_label {{
                color: {CLOUDNEST_COLORS['text_secondary']};
                font-size: 12px;
                font-weight: 500;
            }}
            
            QStatusBar {{
                background-color: {CLOUDNEST_COLORS['surface']};
                border-top: 1px solid {CLOUDNEST_COLORS['input_border']};
                color: {CLOUDNEST_COLORS['text_secondary']};
                font-size: 13px;
                padding: 8px;
            }}
            
            QSplitter::handle {{
                background-color: {CLOUDNEST_COLORS['input_border']};
                width: 2px;
            }}
            
            QSplitter::handle:hover {{
                background-color: {CLOUDNEST_COLORS['accent']};
            }}
        """)
    
    def load_bucket_contents(self, prefix: str = None, load_more: bool = False):
        """Load S3 bucket contents using worker thread with pagination support"""
        if prefix is not None:
            self.current_prefix = prefix
            
        # Reset pagination state for new prefix
        if not load_more:
            self.has_more_objects = False
            self.next_continuation_token = ""
            self.objects.clear()
            self.file_tree.clear()
            
        print(f"DEBUG: Starting bucket content loading for prefix: '{self.current_prefix}', load_more: {load_more}")
        
        if load_more:
            self.status_text.append(f"Loading more objects...")
            self.is_loading_more = True
        else:
            self.status_text.append(f"Loading contents for: {self.current_prefix or 'Root'}...")
        
        # Update navigation buttons (only for initial loads)
        if not load_more:
            self.up_btn.setEnabled(bool(self.current_prefix))
            self.back_btn.setEnabled(len(self.navigation_history) > 0)
            
            # Update path label
            if self.current_prefix:
                self.path_label.setText(f"📁 {self.current_prefix.rstrip('/')}")
            else:
                self.path_label.setText("📁 Root")
        
        # Create and start content worker with pagination support
        continuation_token = self.next_continuation_token if load_more else None
        self.content_worker = BucketContentWorker(
            self.s3_client, self.bucket_name, self.current_prefix, 
            max_keys=self.objects_per_page, continuation_token=continuation_token
        )
        self.content_worker.objects_loaded.connect(self.on_objects_loaded)
        self.content_worker.load_error.connect(self.on_load_error)
        self.content_worker.start()
    
    def on_objects_loaded(self, objects, has_more=False, next_token=""):
        """Handle successful object loading with pagination support"""
        print(f"DEBUG: Received {len(objects)} objects from worker, has_more: {has_more}")
        
        # Update pagination state
        self.has_more_objects = has_more
        self.next_continuation_token = next_token
        
        if self.is_loading_more:
            # Append new objects to existing list (pagination)
            self.objects.extend(objects)
            print(f"DEBUG: Appended {len(objects)} objects, total now: {len(self.objects)}")
            self.is_loading_more = False
        else:
            # Replace objects (initial load)
            self.objects = objects
            print(f"DEBUG: Loaded {len(objects)} objects initially")
        
        self._update_file_tree()
    
    def on_load_error(self, error_message):
        """Handle loading error"""
        print(f"DEBUG: Load error: {error_message}")
        self.status_text.append(error_message)
    
    def on_item_double_clicked(self, item, column):
        """Handle double-click on folder to navigate into it"""
        obj = item.data(0, Qt.ItemDataRole.UserRole)
        if obj and obj.is_folder:
            # Add current path to navigation history
            self.navigation_history.append(self.current_prefix)
            # Navigate into folder
            self.load_bucket_contents(obj.key)
    
    def navigate_back(self):
        """Navigate back to previous folder"""
        if self.navigation_history:
            previous_prefix = self.navigation_history.pop()
            self.load_bucket_contents(previous_prefix)
    
    def navigate_up(self):
        """Navigate up one level"""
        if self.current_prefix:
            # Add current path to navigation history
            self.navigation_history.append(self.current_prefix)
            # Go up one level
            parts = self.current_prefix.rstrip('/').split('/')
            if len(parts) > 1:
                parent_prefix = '/'.join(parts[:-1]) + '/'
            else:
                parent_prefix = ""
            self.load_bucket_contents(parent_prefix)
    
    def _update_file_tree(self):
        """Update file tree widget with pagination support"""
        # Store current item count before adding new items
        current_item_count = self.file_tree.topLevelItemCount()
        
        # Determine which objects to add
        if self.is_loading_more:
            # For pagination, only add objects that aren't already in the tree
            objects_to_add = self.objects[current_item_count:]
        else:
            # For initial load, clear tree and add all objects
            self.file_tree.clear()
            objects_to_add = self.objects
        
        for obj in objects_to_add:
            item = QTreeWidgetItem()
            item.setCheckState(0, Qt.CheckState.Unchecked)
            
            # Name with modern icons
            if obj.is_folder:
                icon = "📁"
                # Show folder name without the full path for cleaner display
                folder_name = obj.key.rstrip('/').split('/')[-1]
                name = f"{icon} {folder_name}"
                size_text = "—"
                type_text = "Folder"
            else:
                # Get file extension for better icons
                file_ext = os.path.splitext(obj.key)[1].lower()
                if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg']:
                    icon = "🖼️"
                elif file_ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']:
                    icon = "🎥"
                elif file_ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                    icon = "🎵"
                elif file_ext in ['.pdf']:
                    icon = "📄"
                elif file_ext in ['.doc', '.docx', '.txt', '.rtf']:
                    icon = "📝"
                elif file_ext in ['.xls', '.xlsx', '.csv']:
                    icon = "📊"
                elif file_ext in ['.zip', '.rar', '.7z', '.tar', '.gz']:
                    icon = "📦"
                elif file_ext in ['.py', '.js', '.html', '.css', '.json', '.xml']:
                    icon = "💻"
                else:
                    icon = "📄"
                
                # Show only filename for cleaner display
                filename = os.path.basename(obj.key)
                name = f"{icon} {filename}"
                size_text = humanize.naturalsize(obj.size)
                type_text = "File"
            
            item.setText(0, name)
            item.setText(1, size_text)
            item.setText(2, obj.last_modified.strftime("%Y-%m-%d %H:%M"))
            item.setText(3, type_text)
            
            # Store object reference
            item.setData(0, Qt.ItemDataRole.UserRole, obj)
            
            self.file_tree.addTopLevelItem(item)
        
        path_display = self.current_prefix or "Root"
        if current_item_count > 0 and len(objects_to_add) > 0:
            # This was a pagination load
            self.status_text.append(f"📄 Loaded {len(objects_to_add)} more items ({len(self.objects)} total)")
            if self.has_more_objects:
                self.status_text.append("📜 Scroll down to load more objects...")
        else:
            # This was an initial load
            status_msg = f"✅ Loaded {len(self.objects)} items from {path_display}"
            if self.has_more_objects:
                status_msg += f" (showing first {self.objects_per_page}, scroll down for more)"
            self.status_text.append(status_msg)
        
        # Resize columns to fit content (only on initial load to avoid constant resizing)
        if current_item_count == 0:  # Initial load
            for i in range(4):
                self.file_tree.resizeColumnToContents(i)
            
            # Add some padding to columns
            for i in range(4):
                current_width = self.file_tree.columnWidth(i)
                self.file_tree.setColumnWidth(i, current_width + 20)
    
    def on_item_changed(self, item, column):
        """Handle item selection change"""
        if column == 0:  # Checkbox column
            obj = item.data(0, Qt.ItemDataRole.UserRole)
            if obj:
                obj.selected = item.checkState(0) == Qt.CheckState.Checked
                
                if obj.selected and obj not in self.selected_objects:
                    self.selected_objects.append(obj)
                elif not obj.selected and obj in self.selected_objects:
                    self.selected_objects.remove(obj)
        
        # Update button states
        has_selection = len(self.selected_objects) > 0
        self.download_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        
        # Update selection info
        if self.selected_objects:
            total_size = sum(obj.size for obj in self.selected_objects if not obj.is_folder)
            files_count = len([obj for obj in self.selected_objects if not obj.is_folder])
            folders_count = len([obj for obj in self.selected_objects if obj.is_folder])
            
            info_parts = []
            if files_count > 0:
                info_parts.append(f"{files_count} file{'s' if files_count != 1 else ''}")
            if folders_count > 0:
                info_parts.append(f"{folders_count} folder{'s' if folders_count != 1 else ''}")
            
            selection_text = f"Selected: {', '.join(info_parts)} ({humanize.naturalsize(total_size)})"
            self.selection_info.setText(selection_text)
            self.status_bar.showMessage(selection_text)
        else:
            self.selection_info.setText("No items selected")
            self.status_bar.showMessage(f"Connected to bucket: {self.bucket_name}")
    
    def toggle_select_all(self, state):
        """Toggle select all items"""
        check_state = Qt.CheckState.Checked if state == 2 else Qt.CheckState.Unchecked
        
        for i in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(i)
            item.setCheckState(0, check_state)
    
    def download_selected(self):
        """Download selected files and folders with adaptive parallel downloads"""
        if not self.selected_objects:
            return
        
        # Get download directory
        download_dir = self.get_download_directory()
        
        # Clear progress area
        self.clear_progress_widgets()
        
        # Create parallel download manager (replaces old DownloadWorker)
        self.download_manager = ParallelDownloadManager(
            self.s3_client, self.bucket_name, 
            self.selected_objects.copy(), download_dir
        )
        
        # Connect signals
        self.download_manager.progress_updated.connect(self.update_download_progress)
        self.download_manager.download_completed.connect(self.on_download_completed)
        self.download_manager.error_occurred.connect(self.on_download_error)
        self.download_manager.all_completed.connect(self.on_all_downloads_completed)
        
        # Add status text explaining adaptive parallel downloads
        parallel_text = (
            f"Starting download with adaptive parallel downloading:\n"
            f"• Starting with 1 download\n"
            f"• Will add parallel downloads when speed exceeds 1024 KB/s\n"
            f"• Maximum of {self.download_manager.max_concurrent_downloads} concurrent downloads"
        )
        self.status_text.append(parallel_text)
        
        # Start download manager
        self.download_manager.start()
        self.status_text.append(f"Preparing download of {len(self.selected_objects)} selected items...")
    
    def delete_selected(self):
        """Delete selected files and folders"""
        if not self.selected_objects:
            return
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete {len(self.selected_objects)} selected items?\n"
            "This action cannot be undone!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Clear progress area
        self.clear_progress_widgets()
        
        # Create delete worker
        self.delete_worker = DeleteWorker(
            self.s3_client, self.bucket_name, self.selected_objects.copy()
        )
        
        # Connect signals
        self.delete_worker.progress_updated.connect(self.update_delete_progress)
        self.delete_worker.deletion_completed.connect(self.on_deletion_completed)
        self.delete_worker.error_occurred.connect(self.on_delete_error)
        self.delete_worker.all_completed.connect(self.on_all_deletions_completed)
        
        # Start deletion
        self.delete_worker.start()
        self.status_text.append(f"Starting deletion of {len(self.selected_objects)} items...")
    
    def get_download_directory(self) -> str:
        """Get the appropriate download directory based on OS"""
        if platform.system() == "Windows":
            return str(Path.home() / "Downloads")
        elif platform.system() == "Darwin":  # macOS
            return str(Path.home() / "Downloads")
        else:  # Linux
            downloads_dir = Path.home() / "Downloads"
            if downloads_dir.exists():
                return str(downloads_dir)
            else:
                return str(Path.home())
    
    def clear_progress_widgets(self):
        """Clear all progress widgets"""
        for i in reversed(range(self.progress_layout.count())):
            child = self.progress_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # Add stretch back
        self.progress_layout.addStretch()
    
    def update_download_progress(self, filename: str, progress: int, status: str, 
                            rate: str = "", size: str = "", eta: str = ""):
        """Update download progress for a file with detailed information"""
        # Find or create progress widget for this file
        progress_widget = self.find_or_create_progress_widget(filename, "download")
        progress_bar = progress_widget.findChild(QProgressBar)
        status_label = progress_widget.findChild(QLabel, "status_label")
        transfer_rate_label = progress_widget.findChild(QLabel, "transfer_rate_label")
        size_progress_label = progress_widget.findChild(QLabel, "size_progress_label")
        eta_label = progress_widget.findChild(QLabel, "eta_label")
        
        # Debug output to help diagnose issues
        print(f"DEBUG UI UPDATE: {filename}: {progress}%, {status}, {rate}, {size}, {eta}")
        
        if progress_bar:
            print(f"DEBUG: Setting progress bar from {progress_bar.value()}% to {progress}%")
            # Update progress bar exactly like the test widget
            progress_bar.setValue(progress)
            progress_bar.setFormat(f"{progress}%")
            
            # Force immediate visual update
            progress_bar.repaint()
            QApplication.processEvents()
            print(f"DEBUG: Progress bar now shows {progress_bar.value()}%")
        
        if status_label:
            current_text = status_label.text()
            new_text = f"{status} ({progress}%)"
            
            # Only update if text changed to minimize repaints
            if current_text != new_text:
                status_label.setText(new_text)
                
                # Set text color based on status
                color = CLOUDNEST_COLORS['text_secondary']
                if "Complete" in status:
                    color = CLOUDNEST_COLORS['success']
                elif "Error" in status:
                    color = CLOUDNEST_COLORS['danger']
                
                status_label.setStyleSheet(f"""
                    font-size: 11px;
                    color: {color};
                    font-weight: 600;
                """)
        
        # Always update these fields, even if they're empty
        if transfer_rate_label:
            if not rate:
                rate = "0 B/s"  # Default value
            
            # Ensure we don't show 0 B/s when downloading is active
            if "0 B/s" in rate and progress > 0 and progress < 100:
                rate = "calculating..."
            
            transfer_rate_label.setText(rate)
            
            # Highlight transfer rate based on speed
            color = CLOUDNEST_COLORS['text_secondary']
            if "MB/s" in rate:
                color = CLOUDNEST_COLORS['success']
            elif "KB/s" in rate:
                try:
                    if float(rate.split()[0]) > 500:
                        color = CLOUDNEST_COLORS['info']
                except (ValueError, IndexError):
                    pass
                
            transfer_rate_label.setStyleSheet(f"""
                font-size: 11px;
                color: {color};
                font-weight: 600;
                text-align: right;
            """)
            
        if size_progress_label:
            if not size:
                size = "0 B / unknown"  # Default value
            size_progress_label.setText(size)
            
        if eta_label:
            if not eta:
                eta = "calculating..."  # Default value
            eta_label.setText(f"ETA: {eta}")
        
        # Simplify the progress display by making it more compact
        # (We'll rely on the detail in progress bar's text value)
        if progress_bar:
            # Set the progress bar format to just show percentage
            progress_bar.setFormat(f"{progress}%")
            
        # Force update the UI to ensure changes are visible immediately
        QApplication.processEvents()
    
    def update_delete_progress(self, filename: str, status: str):
        """Update delete progress for a file"""
        progress_widget = self.find_or_create_progress_widget(filename, "delete")
        status_label = progress_widget.findChild(QLabel, "status_label")
        
        if status_label:
            status_label.setText(status)
    
    def find_or_create_progress_widget(self, filename: str, operation: str) -> QWidget:
        """Find or create a progress widget for a file"""
        # Look for existing widget
        for i in range(self.progress_layout.count()):
            item = self.progress_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if widget.objectName() == f"progress_{filename}":
                    return widget
        
        # Create new progress widget with border and better visibility
        progress_widget = QFrame()
        progress_widget.setObjectName(f"progress_{filename}")
        progress_widget.setFrameShape(QFrame.Shape.StyledPanel)
        progress_widget.setStyleSheet(f"""
            QFrame {{
                border: 1px solid {CLOUDNEST_COLORS['input_border']};
                border-radius: 8px;
                background-color: {CLOUDNEST_COLORS['surface']};
                margin-bottom: 8px;
            }}
        """)
        
        # Simplify the layout to be more compact
        layout = QVBoxLayout(progress_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)  # Reduced spacing for more compact display
        
        if operation == "download":
            # Top row layout - filename with progress info
            top_row = QHBoxLayout()
            top_row.setSpacing(8)
            
            # File name label with better visibility - truncate if too long
            name_label = QLabel(os.path.basename(filename))  # Just show basename instead of full path
            name_label.setObjectName("filename_label")
            name_label.setStyleSheet(f"""
                font-weight: 600;
                color: {CLOUDNEST_COLORS['text_primary']};
                font-size: 13px;
            """)
            name_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            name_label.setToolTip(filename)  # Show full path on hover
            top_row.addWidget(name_label)
            
            # Size info on right side
            size_progress_label = QLabel("0 B / calculating...")
            size_progress_label.setObjectName("size_progress_label")
            size_progress_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            size_progress_label.setStyleSheet(f"""
                font-size: 11px;
                color: {CLOUDNEST_COLORS['text_secondary']};
                font-weight: 500;
            """)
            top_row.addWidget(size_progress_label)
            
            layout.addLayout(top_row)
            
            # Progress bar with better visibility - match test widget exactly
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)  # Start at 0
            progress_bar.setTextVisible(True)
            progress_bar.setFormat("0%")
            progress_bar.setFixedHeight(20)  # Match test height
            progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid {CLOUDNEST_COLORS['input_border']};
                    border-radius: 4px;
                    background-color: {CLOUDNEST_COLORS['surface']};
                    text-align: center;
                    color: {CLOUDNEST_COLORS['text_primary']};
                    font-weight: 600;
                    font-size: 11px;
                }}
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {CLOUDNEST_COLORS['accent']}, 
                        stop:1 {CLOUDNEST_COLORS['accent_hover']});
                    border-radius: 3px;
                }}
            """)
            layout.addWidget(progress_bar)
            
            # Bottom row with status and ETA
            bottom_row = QHBoxLayout()
            bottom_row.setSpacing(8)
            
            # Status label (left)
            status_label = QLabel("Preparing...")
            status_label.setObjectName("status_label")
            status_label.setStyleSheet(f"""
                font-size: 11px;
                color: {CLOUDNEST_COLORS['text_secondary']};
                font-weight: 600;
            """)
            bottom_row.addWidget(status_label)
            
            # Spacer to push items to sides
            bottom_row.addStretch()
            
            # Transfer rate (center)
            transfer_rate_label = QLabel("0 B/s")
            transfer_rate_label.setObjectName("transfer_rate_label")
            transfer_rate_label.setStyleSheet(f"""
                font-size: 11px;
                color: {CLOUDNEST_COLORS['text_secondary']};
                font-weight: 500;
            """)
            bottom_row.addWidget(transfer_rate_label)
            
            # Spacer between rate and ETA
            bottom_row.addSpacing(10)
            
            # ETA (right)
            eta_label = QLabel("ETA: calculating...")
            eta_label.setObjectName("eta_label")
            eta_label.setStyleSheet(f"""
                font-size: 11px;
                color: {CLOUDNEST_COLORS['text_secondary']};
                font-weight: 500;
            """)
            bottom_row.addWidget(eta_label)
            
            layout.addLayout(bottom_row)
            
            # Add a log line immediately to inform user
            self.status_text.append(f"Added {filename} to download queue")
            
        else:
            # For non-download operations, keep it simple
            name_label = QLabel(os.path.basename(filename))
            name_label.setObjectName("filename_label")
            name_label.setStyleSheet(f"""
                font-weight: 600;
                color: {CLOUDNEST_COLORS['text_primary']};
                font-size: 13px;
            """)
            layout.addWidget(name_label)
            
            # Status label
            status_label = QLabel("Initializing...")
            status_label.setObjectName("status_label")
            status_label.setStyleSheet(f"""
                font-size: 11px;
                color: {CLOUDNEST_COLORS['text_secondary']};
                font-weight: 500;
            """)
            layout.addWidget(status_label)
        
        # Insert before stretch
        self.progress_layout.insertWidget(self.progress_layout.count() - 1, progress_widget)
        
        return progress_widget
    
    def on_download_completed(self, filename: str, local_path: str):
        """Handle download completion"""
        self.status_text.append(f"✅ Downloaded: {filename}")
        
        # Show notification
        notification = NotificationWidget(
            f"{filename} downloaded to {local_path}", "success", self
        )
        notification.setGeometry(self.width() - 350, 50, 340, 60)
        notification.show()
        
        # Update progress widget
        progress_widget = self.find_or_create_progress_widget(filename, "download")
        status_label = progress_widget.findChild(QLabel, "status_label")
        if status_label:
            status_label.setText("✅ Completed")
            status_label.setStyleSheet(f"color: {CLOUDNEST_COLORS['success']};")
    
    def on_deletion_completed(self, filename: str):
        """Handle deletion completion"""
        self.status_text.append(f"🗑️ Deleted: {filename}")
        
        # Update progress widget
        progress_widget = self.find_or_create_progress_widget(filename, "delete")
        status_label = progress_widget.findChild(QLabel, "status_label")
        if status_label:
            status_label.setText("🗑️ Deleted")
            status_label.setStyleSheet(f"color: {CLOUDNEST_COLORS['success']};")
    
    def on_download_error(self, error_message: str):
        """Handle download error"""
        self.status_text.append(f"❌ Error: {error_message}")
        QMessageBox.warning(self, "Download Error", error_message)
    
    def on_delete_error(self, error_message: str):
        """Handle delete error"""
        self.status_text.append(f"❌ Error: {error_message}")
        QMessageBox.warning(self, "Delete Error", error_message)
    
    def on_all_downloads_completed(self):
        """Handle all downloads completion"""
        self.status_text.append("🎉 All downloads completed!")
        self.status_bar.showMessage("All downloads completed successfully")
        
        # Show success notification
        notification = NotificationWidget("All downloads completed!", "success", self)
        notification.setGeometry(self.width() - 350, 50, 340, 60)
        notification.show()
    
    def on_all_deletions_completed(self):
        """Handle all deletions completion"""
        self.status_text.append("🎉 All deletions completed!")
        self.status_bar.showMessage("All deletions completed successfully")
        
        # Refresh bucket contents
        self.selected_objects.clear()
        self.load_bucket_contents()


class S3BucketManager(QApplication):
    """Main application class"""
    
    def __init__(self):
        super().__init__(sys.argv)
        self.setApplicationName("CloudNest S3 Manager")
        self.setApplicationVersion("1.0.0")
        
        # Set application style
        self.setStyle('Fusion')
        
        # Apply dark palette if preferred
        self.apply_aws_palette()
        
        # Create and show login window
        self.login_window = LoginWindow()
        self.login_window.login_successful.connect(self.open_browser_window)
        print("DEBUG: Signal connection established")
        self.login_window.show()
        
        self.browser_window = None
    
    def apply_aws_palette(self):
        """Apply AWS-like color palette"""
        palette = QPalette()
        
        # Base colors
        palette.setColor(QPalette.ColorRole.Window, QColor(CLOUDNEST_COLORS['background']))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(CLOUDNEST_COLORS['text_primary']))
        palette.setColor(QPalette.ColorRole.Base, QColor(CLOUDNEST_COLORS['surface']))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(CLOUDNEST_COLORS['surface_alt']))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(CLOUDNEST_COLORS['surface']))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(CLOUDNEST_COLORS['text_primary']))
        palette.setColor(QPalette.ColorRole.Text, QColor(CLOUDNEST_COLORS['text_primary']))
        palette.setColor(QPalette.ColorRole.Button, QColor(CLOUDNEST_COLORS['surface']))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(CLOUDNEST_COLORS['text_primary']))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(CLOUDNEST_COLORS['primary']))
        palette.setColor(QPalette.ColorRole.Link, QColor(CLOUDNEST_COLORS['accent']))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(CLOUDNEST_COLORS['accent']))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(CLOUDNEST_COLORS['surface']))
        
        self.setPalette(palette)
    
    def open_browser_window(self, s3_client, bucket_name: str):
        """Open the main browser window"""
        try:
            print(f"DEBUG: Opening browser window for bucket: {bucket_name}")
            self.login_window.hide()
            
            print("DEBUG: Creating S3BrowserWindow...")
            self.browser_window = S3BrowserWindow(s3_client, bucket_name)
            print("DEBUG: S3BrowserWindow created successfully")
            
            print("DEBUG: Showing browser window...")
            self.browser_window.show()
            print("DEBUG: Browser window show() called")
            
            # Ensure the window is raised and activated
            self.browser_window.raise_()
            self.browser_window.activateWindow()
            print("DEBUG: Browser window should now be visible and active")
            
        except Exception as e:
            print(f"DEBUG: Error in open_browser_window: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    app = S3BucketManager()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()