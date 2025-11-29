"""
RateLimiter Utility

Provides rate limiting functionality for API calls and web scraping.
Ensures respectful usage of external services by enforcing delays between requests.

Example:
    # As a decorator
    @RateLimiter(delay_seconds=2.0)
    def fetch_data():
        return requests.get(url)
    
    # As a context manager
    limiter = RateLimiter(delay_seconds=1.0)
    limiter.wait()
    response = requests.get(url)
"""

import time
import functools
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for API calls and web scraping.
    
    Enforces a minimum delay between consecutive calls to prevent
    overwhelming external services.
    
    Attributes:
        delay_seconds: Minimum delay between calls in seconds
        last_request_time: Timestamp of last request
    
    Usage:
        # As decorator
        @RateLimiter(delay_seconds=2.0)
        def api_call():
            pass
        
        # Manual wait
        limiter = RateLimiter(delay_seconds=1.0)
        limiter.wait()
    """
    
    def __init__(self, delay_seconds: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            delay_seconds: Minimum delay between calls (default: 1.0 second)
        """
        self.delay_seconds = delay_seconds
        self.last_request_time = None
        logger.debug(f"RateLimiter initialized with {delay_seconds}s delay")
    
    def wait(self) -> None:
        """
        Wait if necessary to enforce rate limit.
        
        Calculates elapsed time since last request and sleeps if needed
        to maintain the configured delay.
        """
        if self.last_request_time is None:
            # First call, no delay needed
            logger.debug("First request, no delay")
            self.last_request_time = time.time()
            return
        
        # Calculate elapsed time
        elapsed = time.time() - self.last_request_time
        
        # Calculate remaining wait time
        remaining = self.delay_seconds - elapsed
        
        if remaining > 0:
            logger.debug(f"Rate limiting: waiting {remaining:.2f}s")
            time.sleep(remaining)
        else:
            logger.debug(f"No wait needed, {elapsed:.2f}s elapsed")
        
        # Update last request time
        self.last_request_time = time.time()
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator that wraps function with rate limiting.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function with rate limiting
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            self.wait()
            return func(*args, **kwargs)
        
        return wrapper
