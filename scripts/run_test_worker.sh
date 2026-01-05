#!/bin/bash
# Helper script to run the test RQ worker with proper environment setup
# Handles OBJC_DISABLE_INITIALIZE_FORK_SAFETY on macOS automatically

set -e

PLATFORM=$(uname -s)
REDIS_TEST_DB=${REDIS_TEST_DB:-1}

echo "════════════════════════════════════════════════════════════════════════"
echo "Test RQ Worker - Starting"
echo "════════════════════════════════════════════════════════════════════════"
echo "Platform: $PLATFORM"
echo "Redis DB: $REDIS_TEST_DB"
echo ""

# Set environment for testing
export REDIS_TEST_DB=${REDIS_TEST_DB}
export QDRANT_TEST_COLLECTION="test_rq_worker_assets"
export WORKER_DEBUG=true
export LOG_LEVEL=DEBUG

# On macOS, set the fork safety environment variable
if [ "$PLATFORM" = "Darwin" ]; then
    echo "Detected macOS - setting OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES"
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    echo ""
fi

echo "Starting RQ worker..."
echo "This worker will process jobs from Redis DB $REDIS_TEST_DB"
echo "Press Ctrl+C to stop"
echo ""

# Run the test worker script
cd "$(dirname "$0")/.."
uv run python -m image_search_service.queue.worker
