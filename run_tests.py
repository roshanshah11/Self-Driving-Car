#!/usr/bin/env python
"""
Test runner for the track detection system
"""
import unittest
import sys
import os
import logging
import time

# Configure logging to file
log_dir = "test_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"tests_{int(time.time())}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestRunner")

# Discover and run tests
def run_tests():
    """Run all tests in the test_*.py files"""
    logger.info("Starting test discovery...")
    start_time = time.time()
    
    # Discover all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(".", pattern="test_*.py")
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Report results
    end_time = time.time()
    logger.info(f"Tests completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Ran {result.testsRun} tests")
    
    if result.wasSuccessful():
        logger.info("All tests passed!")
        return 0
    else:
        logger.error(f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        for failure in result.failures:
            test_case, traceback = failure
            logger.error(f"FAIL: {test_case}")
            logger.error(traceback)
        
        for error in result.errors:
            test_case, traceback = error
            logger.error(f"ERROR: {test_case}")
            logger.error(traceback)
        
        return 1

if __name__ == "__main__":
    sys.exit(run_tests()) 