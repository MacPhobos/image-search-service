#!/usr/bin/env python3
"""Bootstrap OAuth 2.0 refresh token for Google Drive personal account integration.

One-time setup script for obtaining an offline refresh token so that the
GoogleDriveOAuthV3Storage backend can access your personal Google Drive.
Run this script locally (on a machine with a browser), then copy the output
values to your deployment's .env file.

Requirements:
    pip install google-auth-oauthlib
    (or: uv sync from the image-search-service directory)

Setup (before running this script):
    1. Go to https://console.cloud.google.com/apis/credentials
    2. Click "Create Credentials" > "OAuth client ID"
    3. Application type: Desktop app
    4. Download the JSON file ("Download JSON" button)
    5. Enable the Google Drive API:
       https://console.cloud.google.com/apis/library/drive.googleapis.com
    6. Add your email as a test user in the OAuth consent screen:
       https://console.cloud.google.com/apis/credentials/consent

Usage:
    python scripts/gdrive_oauth_bootstrap.py \\
        --client-secrets /path/to/client_secret_XXXXX.json \\
        [--scopes drive] \\
        [--port 8080]

After running:
    1. Copy the printed config values to your .env file.
    2. Set GOOGLE_DRIVE_ROOT_ID to the folder ID where photos should be stored.
       (Open the folder in Google Drive; the ID is the last segment of the URL.)
    3. Restart the application: make dev && make worker

Security notes:
    - The refresh_token grants Drive access — treat it like a password.
    - Never commit .env files containing credentials.
    - You can revoke access at: https://myaccount.google.com/permissions
    - The client_secrets.json file may be deleted after bootstrapping.

OAuth app publishing status:
    If your app is in "testing" mode (default), refresh tokens expire after
    7 days. To avoid weekly re-bootstrapping, either:
    a) Publish the app (requires Google verification for the 'drive' scope), or
    b) Keep re-running this script weekly (simple for self-hosted use).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Scope presets. "drive" grants full access; "drive.file" restricts to
# files and folders created by this application (more secure but cannot
# access pre-existing folders in your Drive).
SCOPE_PRESETS: dict[str, list[str]] = {
    "drive": ["https://www.googleapis.com/auth/drive"],
    "drive.file": ["https://www.googleapis.com/auth/drive.file"],
}

DEFAULT_PORT = 8080


def main() -> int:
    """Run the OAuth bootstrap flow. Returns exit code (0 = success)."""
    parser = argparse.ArgumentParser(
        description="Bootstrap Google Drive OAuth 2.0 refresh token",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--client-secrets",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the OAuth client secrets JSON downloaded from Google Cloud Console",
    )
    parser.add_argument(
        "--scopes",
        default="drive",
        choices=list(SCOPE_PRESETS.keys()),
        help=(
            "Scope preset: 'drive' (full access, default) or "
            "'drive.file' (app-created files only, more restrictive)"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Local port for the OAuth redirect callback (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the config as a JSON file instead of printing to stdout",
    )

    args = parser.parse_args()

    # --- Validate client secrets file ---
    client_secrets_path: Path = args.client_secrets
    if not client_secrets_path.exists():
        print(
            f"ERROR: Client secrets file not found: {client_secrets_path}",
            file=sys.stderr,
        )
        return 1

    try:
        with client_secrets_path.open() as f:
            secrets_data: dict[str, object] = json.load(f)
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: Invalid JSON in client secrets file: {exc}",
            file=sys.stderr,
        )
        return 1

    # Extract client_id for display (never log client_secret).
    installed_or_web = secrets_data.get("installed", secrets_data.get("web", {}))
    client_id = (
        installed_or_web.get("client_id", "unknown")
        if isinstance(installed_or_web, dict)
        else "unknown"
    )

    scopes: list[str] = SCOPE_PRESETS[args.scopes]
    port: int = args.port

    print(f"Client ID  : {client_id}")
    print(f"Scopes     : {scopes}")
    print(f"Local port : {port}")
    print()

    # --- Import google-auth-oauthlib (dev/optional dep) ---
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: google-auth-oauthlib is not installed.\n"
            "Install with one of:\n"
            "  uv sync  (from the image-search-service directory)\n"
            "  pip install google-auth-oauthlib",
            file=sys.stderr,
        )
        return 1

    # --- Run the OAuth consent flow ---
    print("Opening browser for Google OAuth consent...")
    print("(If the browser does not open automatically, copy the URL shown below.)")
    print()

    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_secrets_path),
        scopes=scopes,
    )

    # access_type="offline" requests a refresh token.
    # prompt="consent" forces Google to issue a new refresh token even if the
    # user previously authorized this app (Google only issues refresh tokens on
    # first consent or when re-consent is forced).
    credentials = flow.run_local_server(
        port=port,
        access_type="offline",
        prompt="consent",
        success_message=(
            "Authentication successful! "
            "You may close this browser tab and return to the terminal."
        ),
    )

    # --- Validate that a refresh token was received ---
    if not credentials.refresh_token:
        print(
            "\nERROR: No refresh token was returned by Google.\n"
            "This happens when:\n"
            "  1. You previously authorised this app without prompt=consent\n"
            "  2. The OAuth consent screen is misconfigured\n"
            "\nFix: Revoke previous access at https://myaccount.google.com/permissions\n"
            "     Then re-run this script.",
            file=sys.stderr,
        )
        return 1

    # --- Build config output (never include client_secret in logs) ---
    config = {
        "GOOGLE_DRIVE_AUTH_MODE": "oauth",
        "GOOGLE_DRIVE_CLIENT_ID": credentials.client_id,
        "GOOGLE_DRIVE_CLIENT_SECRET": credentials.client_secret,
        "GOOGLE_DRIVE_REFRESH_TOKEN": credentials.refresh_token,
    }

    if args.output_json:
        output_path: Path = args.output_json
        # Write with restrictive permissions (owner read/write only).
        # os.open with mode 0o600 sets the initial permissions atomically,
        # preventing a window where the file is readable by other users.
        fd = os.open(str(output_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(f"\nConfig written to: {output_path}")
        print("SECURITY: This file has been created with 0o600 (owner read/write only).")
        print("          It contains sensitive credentials — never commit it.")
    else:
        separator = "=" * 60
        print(f"\n{separator}")
        print("SUCCESS! Add these to your .env file:")
        print(separator)
        print()
        for key, value in config.items():
            print(f"{key}={value}")
        print()
        print("# Also set these (if not already in your .env):")
        print("GOOGLE_DRIVE_ENABLED=true")
        print("GOOGLE_DRIVE_ROOT_ID=<your_folder_id>")
        print()
        print("# Find your folder ID:")
        print("# Open the target folder in Google Drive in a browser.")
        print("# The URL ends with the folder ID:")
        print("# https://drive.google.com/drive/folders/<FOLDER_ID_HERE>")
        print()
        print(separator)
        print("SECURITY REMINDERS:")
        print("  - Never commit .env files with these credentials")
        print("  - The refresh token grants Drive access — treat as a password")
        print("  - Revoke access at: https://myaccount.google.com/permissions")
        print(separator)

    return 0


if __name__ == "__main__":
    sys.exit(main())
