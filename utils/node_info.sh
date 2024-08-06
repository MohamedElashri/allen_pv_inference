#!/bin/bash

# Function to check if a command exists
command_exists() {
  type "$1" &> /dev/null
}

# ANSI Color Codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Retrieve CPU information
cpu_info=$(lscpu | grep "Model name" | awk -F ':' '{print $2}' | sed -e 's/^[ \t]*//')

# Check for GPU information; if no GPU, return "N/A"
gpu_info="N/A"
if command_exists lspci; then
  gpu_info=$(lspci | grep -i vga | cut -d ":" -f3)
  if [ -z "$gpu_info" ]; then
    gpu_info="N/A"
  fi
fi

# Get the number of cores and threads
cores=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
threads=$(lscpu | grep "Thread(s) per core" | awk '{print $4}')
cores_and_threads="$cores cores, $threads threads/core"

# Memory usage; format to be in one line
memory_info=$(free -h | awk '/^Mem|Swap/ {printf "%s: %s total, %s used, %s free; ", $1, $2, $3, $4}')
memory_info=${memory_info%??} # Remove the last '; ' for neatness

# Get current username and root status
username=$(whoami)
root_status="no"
if [ "$username" = "root" ]; then
  root_status="yes"
fi

# Get groups the user is part of
groups_info=$(id -Gn $username | sed 's/ /, /g')

# Get information about which Linux distribution is being used
distro_info=$(cat /etc/*-release | grep "PRETTY_NAME" | awk -F '=' '{print $2}' | sed 's/"//g')


# Print information in a table with borders and colors
echo -e "${BLUE}+------------------+--------------------------------------------------+${NC}"
echo -e "${BLUE}|${NC} ${GREEN}Information${NC}         | ${GREEN}Value${NC}                                            ${BLUE}|${NC}"
echo -e "${BLUE}+------------------+--------------------------------------------------+${NC}"
echo -e "${BLUE}|${NC} CPU Info          | $cpu_info ${BLUE}|${NC}"
echo -e "${BLUE}|${NC} GPU Info          | $gpu_info ${BLUE}|${NC}"
echo -e "${BLUE}|${NC} Cores & Threads   | $cores_and_threads ${BLUE}|${NC}"
echo -e "${BLUE}|${NC} Memory            | $memory_info ${BLUE}|${NC}"
echo -e "${BLUE}|${NC} Username          | $username (root: $root_status) ${BLUE}|${NC}"
echo -e "${BLUE}|${NC} Groups            | $groups_info ${BLUE}|${NC}"
echo -e "${BLUE}|${NC} Distro            | $distro_info ${BLUE}|${NC}"
echo -e "${BLUE}+------------------+--------------------------------------------------+${NC}"
