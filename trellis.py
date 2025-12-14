#!/usr/bin/env python3
"""
TRELLIS - Interactive Preference Steering for Language Models
==============================================================

A framework for guided growth of model personality through direct preference.
Generate responses, pick the one with the right vibe, update weights, and
observe the shift. Linear undo lets you explore different paths.

Core loop:
    prompt -> sample variants -> vibe-check -> update LoRA -> observe shift

Usage:
    python trellis.py                    # Launch Gradio UI
    python trellis.py --share            # Launch with public URL
    python trellis.py --port 7861        # Custom port
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Trellis: Interactive Preference Steering"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public URL for sharing",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0 for external access)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./trellis_sessions",
        help="Directory for session data (default: ./trellis_sessions)",
    )
    args = parser.parse_args()

    # Import here to defer heavy imports until needed
    import gradio as gr
    from ui.app import TrellisApp, build_ui
    from ui.styles import MOBILE_CSS

    print("=" * 60)
    print("TRELLIS - Interactive Preference Steering")
    print("=" * 60)
    print()
    print(f"Sessions directory: {args.save_dir}")
    print(f"Server: http://{args.host}:{args.port}")
    if args.share:
        print("Public URL will be generated...")
    print()

    # Create app and UI
    app = TrellisApp(base_save_dir=args.save_dir)
    demo = build_ui(app)

    # Launch with theme and CSS (Gradio 6.0+ requires these in launch())
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=MOBILE_CSS,
    )


if __name__ == "__main__":
    main()
