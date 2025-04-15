#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Activate the trellis Conda environment
source /opt/conda/bin/activate trellis

# Pass all arguments to the container as commands

# If the below script exists run it, otherwise run another
if [ -f "/mnt/scripts/image_to_3d.py" ]; then
    # Run the image_to_3d.py script with all arguments passed to the container
    python3 /mnt/scripts/image_to_3d.py "$@"
else
    python3 /app/image_to_3d.py "$@"
fi

echo "Container has finished running."