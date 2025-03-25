#!/bin/bash

# IPL Analytics App Build Script
# This script creates a deployment package for Elastic Beanstalk

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Set script to exit on error
set -e

# Start timestamp
START_TIME=$(date +%s)
BUILD_DATE=$(date +"%Y%m%d")
BUILD_TIME=$(date +"%H%M%S")

# Get version from version file or create it
VERSION_FILE="version.txt"
if [ -f "$VERSION_FILE" ]; then
    VERSION=$(cat "$VERSION_FILE")
    # Increment patch version
    MAJOR=$(echo $VERSION | cut -d. -f1)
    MINOR=$(echo $VERSION | cut -d. -f2)
    PATCH=$(echo $VERSION | cut -d. -f3)
    PATCH=$((PATCH + 1))
    VERSION="$MAJOR.$MINOR.$PATCH"
else
    VERSION="1.0.0"
fi

# Save new version
echo "$VERSION" > "$VERSION_FILE"

# Define build directory
BUILD_DIR="builds"
mkdir -p "$BUILD_DIR"

# Define build name
BUILD_NAME="ipl-analytics-v${VERSION}-${BUILD_DATE}-${BUILD_TIME}"
DEPLOY_ZIP="${BUILD_DIR}/${BUILD_NAME}.zip"

echo -e "${GREEN}Starting build process for IPL Analytics App v${VERSION}${NC}"
echo -e "${YELLOW}Build timestamp: $(date)${NC}"

# Validate necessary files
echo -e "\n${YELLOW}Checking required files...${NC}"

required_files=(
    "app/app.py"
    "app/requirements.txt"
    "Procfile"
    "requirements.txt"
    ".ebextensions/01_nginx.config"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Missing required file: $file${NC}"
        missing_files=$((missing_files + 1))
    else
        echo -e "${GREEN}✓ Found: $file${NC}"
    fi
done

if [ $missing_files -gt 0 ]; then
    echo -e "${RED}Build failed: Missing $missing_files required files${NC}"
    exit 1
fi

# Validate Nginx configuration
echo -e "\n${YELLOW}Validating Nginx configuration files...${NC}"

# Check for upstream directives outside http context
nginx_config_files=(
    "app/.platform/nginx/conf.d/elasticbeanstalk/01_streamlit.conf"
    ".ebextensions/01_nginx.config"
    "app/.ebextensions/01_nginx.config"
    "app/.ebextensions/02_nginx.config"
)

nginx_errors=0
for config in "${nginx_config_files[@]}"; do
    if [ -f "$config" ]; then
        if grep -q "upstream.*{" "$config"; then
            if grep -q "server.*{" "$config"; then
                echo -e "${RED}Warning: $config may have upstream directive in wrong context${NC}"
                nginx_errors=$((nginx_errors + 1))
            fi
        fi
        echo -e "${GREEN}✓ Checked: $config${NC}"
    else
        echo -e "${YELLOW}Config file not found: $config (may be acceptable)${NC}"
    fi
done

if [ $nginx_errors -gt 0 ]; then
    echo -e "${RED}Warning: Potential Nginx configuration issues found${NC}"
    read -p "Continue with build? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Build aborted${NC}"
        exit 1
    fi
fi

# Check for Python syntax errors
echo -e "\n${YELLOW}Checking Python syntax...${NC}"
python_files=$(find . -name "*.py" | grep -v "__pycache__" | grep -v "venv")
python_errors=0

for py_file in $python_files; do
    if ! python -m py_compile "$py_file"; then
        echo -e "${RED}Syntax error in $py_file${NC}"
        python_errors=$((python_errors + 1))
    fi
done

if [ $python_errors -gt 0 ]; then
    echo -e "${RED}Build failed: $python_errors Python files contain syntax errors${NC}"
    exit 1
else
    echo -e "${GREEN}✓ All Python files passed syntax check${NC}"
fi

# Create version info file
echo -e "\n${YELLOW}Creating version info...${NC}"
VERSION_INFO="app/version_info.json"
cat > "$VERSION_INFO" << EOF
{
    "version": "$VERSION",
    "build_date": "$BUILD_DATE",
    "build_time": "$BUILD_TIME",
    "timestamp": "$(date)"
}
EOF
echo -e "${GREEN}✓ Created version info at $VERSION_INFO${NC}"

# Create build ZIP file
echo -e "\n${YELLOW}Creating deployment package...${NC}"
if [ -f ".ebignore" ]; then
    zip -r "$DEPLOY_ZIP" . -x@.ebignore
else
    # Default exclusions if .ebignore doesn't exist
    zip -r "$DEPLOY_ZIP" . -x "*.git*" "*.DS_Store" "__pycache__/*" "*.pyc" "venv/*" "builds/*"
fi

# Calculate build size
BUILD_SIZE=$(du -h "$DEPLOY_ZIP" | cut -f1)

# Record build info
BUILD_LOG="${BUILD_DIR}/build_history.log"
echo "[$BUILD_DATE $BUILD_TIME] v$VERSION - Size: $BUILD_SIZE - File: $BUILD_NAME.zip" >> "$BUILD_LOG"

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo -e "\n${GREEN}Build completed successfully!${NC}"
echo -e "Build: ${YELLOW}$BUILD_NAME${NC}"
echo -e "Version: ${YELLOW}$VERSION${NC}"
echo -e "Size: ${YELLOW}$BUILD_SIZE${NC}"
echo -e "Duration: ${YELLOW}$DURATION seconds${NC}"
echo -e "Deployment package: ${YELLOW}$DEPLOY_ZIP${NC}"
echo -e "\nTo deploy to Elastic Beanstalk, upload the zip file through the AWS Console"
echo -e "or use the AWS CLI/EB CLI commands."
