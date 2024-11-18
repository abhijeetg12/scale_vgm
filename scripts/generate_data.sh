# scripts/generate_data.sh
#!/bin/bash

# Exit on error
set -e

echo "Starting data generation process..."

# Activate virtual environment if you have one
# source venv/bin/activate

# Run the generation script
python -m src.scripts.generate_data

echo "Data generation complete!"