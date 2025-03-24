#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Activate the trellis Conda environment
source /opt/conda/bin/activate trellis

# Pass all arguments to the container as commands
python3 /app/image_to_3d.py "$@"
