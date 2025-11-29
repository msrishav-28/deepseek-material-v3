"""
Unit tests for RateLimiter utility.

Tests verify:
1. Rate limiter enforces delays between calls
2. Decorator functionality works correctly
3. No delay on first call
"""

import pytest
import time
from ceramic_discovery.data_pipeline.utils.rate_limiter import RateLimiter


class TestRateLimiterDelays:
    """Test that RateLimiter enforces delays."""
    
    def test_rate_limiter_delays(self):
        """Create RateLimiter and verify delay >= configured value."""
        delay_seconds = 0.1
        limiter = RateLimiter(delay_seconds=delay_seconds)
        
        # First call - should not delay
        start_time = time.time()
        limiter.wait()
        first_call_time = time.time() - start_time
        
        # First call should be immediate (< 0.05s)
        assert first_call_time < 0.05
        
        # Second call - should delay
        start_time = time.time()
        limiter.wait()
        second_call_time = time.time() - start_time
        
        # Second call should wait at least delay_seconds
        assert second_call_time >= delay_seconds
        # Allow some tolerance for timing precision
        assert second_call_time < delay_seconds + 0.05
    
    def test_multiple_calls_enforce_delay(self):
        """Verify delay enforced across multiple calls."""
        delay_seconds = 0.05
        limiter = RateLimiter(delay_seconds=delay_seconds)
        
        call_times = []
        
        # Make 5 calls
        for i in range(5):
            start = time.time()
            limiter.wait()
            elapsed = time.time() - start
            call_times.append(elapsed)
        
        # First call should be immediate
        assert call_times[0] < 0.02
        
        # Subsequent calls should have delays
        for i in range(1, 5):
            assert call_times[i] >= delay_seconds * 0.9  # Allow 10% tolerance


class TestDecoratorFunctionality:
    """Test RateLimiter as a decorator."""
    
    def test_decorator_functionality(self):
        """Create test function decorated with RateLimiter and verify rate limiting."""
        delay_seconds = 0.1
        call_times = []
        
        @RateLimiter(delay_seconds=delay_seconds)
        def test_function():
            call_times.append(time.time())
            return "result"
        
        # Call function multiple times
        for _ in range(3):
            result = test_function()
            assert result == "result"
        
        # Verify delays between calls
        assert len(call_times) == 3
        
        # First to second call should have delay
        delay_1_2 = call_times[1] - call_times[0]
        assert delay_1_2 >= delay_seconds * 0.9
        
        # Second to third call should have delay
        delay_2_3 = call_times[2] - call_times[1]
        assert delay_2_3 >= delay_seconds * 0.9
    
    def test_decorator_preserves_function_metadata(self):
        """Verify decorator preserves function name and docstring."""
        @RateLimiter(delay_seconds=0.1)
        def my_function():
            """This is my function."""
            return 42
        
        # functools.wraps should preserve metadata
        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "This is my function."
    
    def test_decorator_with_arguments(self):
        """Verify decorated function can accept arguments."""
        @RateLimiter(delay_seconds=0.05)
        def add(a, b):
            return a + b
        
        result1 = add(1, 2)
        result2 = add(3, 4)
        
        assert result1 == 3
        assert result2 == 7


class TestNoDelayOnFirstCall:
    """Test that first call has no delay."""
    
    def test_no_delay_on_first_call(self):
        """Create new RateLimiter and verify first wait() returns immediately."""
        limiter = RateLimiter(delay_seconds=1.0)
        
        start_time = time.time()
        limiter.wait()
        elapsed = time.time() - start_time
        
        # First call should be immediate (< 0.05s)
        assert elapsed < 0.05
    
    def test_second_call_has_delay(self):
        """Verify second call enforces delay."""
        limiter = RateLimiter(delay_seconds=0.1)
        
        # First call
        limiter.wait()
        
        # Second call should delay
        start_time = time.time()
        limiter.wait()
        elapsed = time.time() - start_time
        
        assert elapsed >= 0.1


class TestDifferentDelayValues:
    """Test RateLimiter with different delay values."""
    
    def test_short_delay(self):
        """Test with very short delay (0.05s)."""
        limiter = RateLimiter(delay_seconds=0.05)
        
        limiter.wait()  # First call
        
        start = time.time()
        limiter.wait()  # Second call
        elapsed = time.time() - start
        
        assert elapsed >= 0.05
        assert elapsed < 0.1
    
    def test_longer_delay(self):
        """Test with longer delay (0.2s)."""
        limiter = RateLimiter(delay_seconds=0.2)
        
        limiter.wait()  # First call
        
        start = time.time()
        limiter.wait()  # Second call
        elapsed = time.time() - start
        
        assert elapsed >= 0.2
        assert elapsed < 0.3
    
    def test_default_delay(self):
        """Test default delay value (1.0s)."""
        limiter = RateLimiter()  # Default is 1.0s
        
        assert limiter.delay_seconds == 1.0
