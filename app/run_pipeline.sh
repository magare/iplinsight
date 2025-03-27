#!/bin/bash

# Set error handling
set -e
set -o pipefail

# Script variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# Create necessary directories
mkdir -p "${LOG_DIR}"
mkdir -p "${SCRIPT_DIR}/data"
mkdir -p "${SCRIPT_DIR}/models"
mkdir -p "${SCRIPT_DIR}/output"

# Logging function
log() {
    local level=$1
    shift
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [${level}] $*" | tee -a "${LOG_FILE}"
}

# Error handler
error_handler() {
    local line_no=$1
    local error_code=$2
    log "ERROR" "Error occurred in line ${line_no} (exit code: ${error_code})"
    exit "${error_code}"
}

# Set error trap
trap 'error_handler ${LINENO} $?' ERR

# Check Python environment
check_environment() {
    log "INFO" "Checking Python environment..."
    
    if ! command -v python3 &> /dev/null; then
        log "ERROR" "Python 3 is not installed"
        exit 1
    fi
    
    if ! command -v pip3 &> /dev/null; then
        log "ERROR" "pip3 is not installed"
        exit 1
    }
    
    # Check for virtual environment
    if [[ -z "${VIRTUAL_ENV}" ]]; then
        log "WARNING" "Not running in a virtual environment"
    fi
    
    log "INFO" "Environment check passed"
}

# Install dependencies
install_dependencies() {
    log "INFO" "Installing dependencies..."
    pip3 install -r "${SCRIPT_DIR}/requirements.txt" | tee -a "${LOG_FILE}"
    log "INFO" "Dependencies installed successfully"
}

# Function to run a pipeline stage
run_stage() {
    local stage=$1
    local script=$2
    
    log "INFO" "Starting stage: ${stage}"
    
    python3 "${script}" 2>&1 | tee -a "${LOG_FILE}"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "INFO" "Stage completed successfully: ${stage}"
    else
        log "ERROR" "Stage failed: ${stage}"
        exit 1
    fi
}

# Parse command line arguments
SKIP_FEATURE_ENGINEERING=0
SKIP_MODEL_TRAINING=0
SKIP_TEAM_OPTIMIZATION=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-feature-engineering)
            SKIP_FEATURE_ENGINEERING=1
            shift
            ;;
        --skip-model-training)
            SKIP_MODEL_TRAINING=1
            shift
            ;;
        --skip-team-optimization)
            SKIP_TEAM_OPTIMIZATION=1
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --skip-feature-engineering    Skip feature engineering stage"
            echo "  --skip-model-training        Skip model training stage"
            echo "  --skip-team-optimization     Skip team optimization stage"
            echo "  --help                       Show this help message"
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log "INFO" "Starting IPL Fantasy Team Predictor pipeline"
    
    # Check environment
    check_environment
    
    # Install dependencies
    install_dependencies
    
    # Feature Engineering
    if [ ${SKIP_FEATURE_ENGINEERING} -eq 0 ]; then
        run_stage "Feature Engineering" "${SCRIPT_DIR}/utils/feature_engineering.py"
    else
        log "INFO" "Skipping Feature Engineering stage"
    fi
    
    # Model Training
    if [ ${SKIP_MODEL_TRAINING} -eq 0 ]; then
        run_stage "Model Training" "${SCRIPT_DIR}/utils/model_training.py"
    else
        log "INFO" "Skipping Model Training stage"
    fi
    
    # Team Optimization
    if [ ${SKIP_TEAM_OPTIMIZATION} -eq 0 ]; then
        run_stage "Team Optimization" "${SCRIPT_DIR}/utils/team_optimizer.py"
    else
        log "INFO" "Skipping Team Optimization stage"
    fi
    
    log "INFO" "Pipeline completed successfully"
}

# Run main function
main

exit 0 