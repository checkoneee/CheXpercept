"""
Logger utility for redirecting print statements to log files
"""
import sys
from datetime import datetime

class Logger:
    def __init__(self, log_file=None):
        """
        Initialize logger
        
        Args:
            log_file: Path to log file. If None, only print to terminal
        """
        self.terminal = sys.stdout
        self.log_file = log_file
        self.file_handle = None
        
        if self.log_file:
            self.file_handle = open(self.log_file, 'w', encoding='utf-8')
            self.write(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.write("=" * 80 + "\n\n")
    
    def write(self, message):
        """Write message to log file"""
        if self.file_handle:
            self.file_handle.write(message)
            self.file_handle.flush()
    
    def print(self, message):
        """Print message to log file only (not terminal)"""
        self.write(str(message) + '\n')
    
    def close(self):
        """Close log file"""
        if self.file_handle:
            self.write(f"\nLog ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_handle.close()
            self.file_handle = None
    
    def __del__(self):
        """Destructor to ensure file is closed"""
        self.close()

# Global logger instance
_logger = None

def init_logger(log_file):
    """Initialize global logger"""
    global _logger
    _logger = Logger(log_file)
    return _logger

def get_logger():
    """Get global logger instance"""
    return _logger

def log_print(*args, **kwargs):
    """Print to log file instead of terminal"""
    global _logger
    if _logger:
        # Convert all args to strings and join them
        message = ' '.join(str(arg) for arg in args)
        # Handle end parameter
        end = kwargs.get('end', '\n')
        _logger.print(message + ('' if end == '\n' else end))
    # else: silently ignore if logger not initialized (for multiprocessing workers)

def close_logger():
    """Close global logger"""
    global _logger
    if _logger:
        _logger.close()
        _logger = None
