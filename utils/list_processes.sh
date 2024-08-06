#!/bin/bash

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Header for the table
echo -e "${BOLD}USER\t\tPID\t%CPU\t%MEM\tCOMMAND${NC}"
echo "--------------------------------------------"

# Find GPU process IDs
gpu_pids=$(lsof /dev/nvidia* 2>/dev/null | grep -E "^$USER" | awk '{print $2}')

# Fetch all processes for the current user excluding GPU processes
ps -u $USER -o user,pid,pcpu,pmem,comm --sort=-pcpu | awk -v gpu_pids="$gpu_pids" -v red="$RED" -v green="$GREEN" -v yellow="$YELLOW" -v nc="$NC" 'NR>1 {
  if (index(gpu_pids, $2) == 0) {  # Check if the PID is not a GPU process
    # Color-code rows based on CPU usage
    if ($3>=50.0) color=red; else if ($3>=10.0) color=yellow; else color=green;
  
    # Print each row with colors
    printf "%s%-10s\t%-8s\t%-8s\t%-8s\t%-10s%s\n", color, $1, $2, $3, $4, $5, nc;
  }
}'

# Special case: Highlight processes running on the GPU
echo -e "\n${BOLD}GPU Processes:${NC}"
echo "--------------------------------------------"

# Check if there are any GPU processes
if [ -z "$gpu_pids" ]; then
  echo "No GPU processes are running."
else
  # Print GPU processes
  for pid in $gpu_pids; do
    ps -p $pid -o user,pid,comm | awk -v red="$RED" -v nc="$NC" 'NR>1 {
      printf "%s%-10s\t%-8s\t%-10s%s\n", red, $1, $2, $3, nc;
    }'
  done
fi
