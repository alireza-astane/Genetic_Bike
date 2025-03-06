FROM python:3.9-slim

WORKDIR /app

# Upgrade pip and install build prerequisites
RUN pip install --upgrade pip && pip install build

# Copy the manifest and source files
COPY setup.py .
COPY src/ ./src

# Install the package (this also installs dependencies from install_requires)
RUN pip install -e .

# Run main.py directly (adjust this path if needed)
CMD ["python", "src/main/main.py"]

# Alternatively, if using the CLI command defined in setup.py:
# CMD ["genetic_bike"]