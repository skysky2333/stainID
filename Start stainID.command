#!/bin/bash
# Double-click to start stainID. It opens in your web browser; keep this window open while you work.
cd "$(dirname "$0")"
exec .venv/bin/stainid serve
