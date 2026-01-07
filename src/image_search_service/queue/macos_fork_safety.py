"""macOS fork-safety utilities for RQ work-horse processes.

This module provides utilities to prevent fork-safety crashes when RQ work-horse
subprocesses initialize on macOS. The main issue: urllib's proxy detection code
tries to fork to access system proxy settings, which causes crashes when called
from a multi-threaded process (even though spawn() is used).

Root Cause:
- Python's urllib.request.getproxies() on macOS calls SCDynamicStoreCopyProxiesWithOptions
- This system call forks to access /Library preferences
- When called in a multi-threaded context (after pydantic/sqlalchemy load), crash occurs
- The crash happens on the child side of fork during exec pre-exec initialization

Solution:
- Disable environment variable proxy detection
- Disable system proxy detection
- Set all proxy env vars to empty string
- This prevents urllib from trying to detect proxies via fork()
"""

import os
import platform


def disable_proxy_detection() -> None:
    """Disable system proxy detection on macOS to prevent fork-safety crashes.

    This must be called EARLY in the work-horse subprocess, before any
    urllib/requests imports that might trigger proxy detection.

    Environment Variables Set:
    - HTTP_PROXY, HTTPS_PROXY, FTP_PROXY = "" (disable)
    - NO_PROXY = "*" (match all domains, bypass proxy)
    - REQUEST_METHOD set to prevent CGI-style proxy detection

    This prevents urllib from:
    1. Checking environment for HTTP_PROXY variables
    2. Calling getproxies() which forks on macOS
    3. Attempting system proxy detection via SCDynamicStore

    Why This Works:
    - urllib checks env vars first before system detection
    - Setting empty proxy env vars short-circuits proxy detection
    - No fork = no multi-threaded fork crash
    """
    if platform.system() == "Darwin":  # Only on macOS
        # Disable environment-based proxy detection
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""
        os.environ["FTP_PROXY"] = ""

        # Also set lowercase versions (urllib checks both)
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["ftp_proxy"] = ""

        # Set NO_PROXY to match all (bypass proxy for all hosts)
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

        # Disable CGI-style proxy detection
        os.environ.pop("REQUEST_METHOD", None)

        # Disable all_proxy fallback
        os.environ["all_proxy"] = ""
        os.environ["ALL_PROXY"] = ""
