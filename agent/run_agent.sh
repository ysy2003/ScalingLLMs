#!/bin/bash

# This script reads configuration from config.yaml, sets them as environment variables,
# and runs the repair agent Python script.

# Function to parse a simple key: value YAML file.
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'", toupper(vn), toupper($2), $3);
      }
   }'
}

# Check if config.yaml exists
CONFIG_FILE="agent/config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

# Read config.yaml and export variables
echo "📦 Loading configuration from $CONFIG_FILE..."
eval "$(parse_yaml $CONFIG_FILE)"

# Export the variables so the Python script can access them
export GCP_PROJECT_ID
export GCP_REGION
export MODEL_ID
export TEST_FILE_PATH

echo "🚀 Starting the repair agent..."
python3 agent/repair_agent.py