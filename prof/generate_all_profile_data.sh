#!/bin/bash

echo "========== Generating All Profiling Datasets =========="
echo ""

# 100K
echo "--- Compiling 100K generator ---"
g++ generate_profile_data.cpp -o gen_data_100k -lm -O2
if [ $? -ne 0 ]; then echo "Failed to compile 100K generator"; exit 1; fi

echo "--- Running 100K generator ---"
./gen_data_100k
if [ $? -ne 0 ]; then echo "Failed to run 100K generator"; exit 1; fi
echo ""

# 1M
echo "--- Compiling 1M generator ---"
g++ generate_profile_data_1m.cpp -o gen_data_1m -lm -O2
if [ $? -ne 0 ]; then echo "Failed to compile 1M generator"; exit 1; fi

echo "--- Running 1M generator ---"
./gen_data_1m
if [ $? -ne 0 ]; then echo "Failed to run 1M generator"; exit 1; fi
echo ""

# 10M
echo "--- Compiling 10M generator ---"
g++ generate_profile_data_10m.cpp -o gen_data_10m -lm -O2
if [ $? -ne 0 ]; then echo "Failed to compile 10M generator"; exit 1; fi

echo "--- Running 10M generator (this will take a minute or two) ---"
./gen_data_10m
if [ $? -ne 0 ]; then echo "Failed to run 10M generator"; exit 1; fi
echo ""

# 100M
echo "--- Compiling 100M generator ---"
g++ generate_profile_data_100m.cpp -o gen_data_100m -lm -O2
if [ $? -ne 0 ]; then echo "Failed to compile 100M generator"; exit 1; fi

echo "--- Running 100M generator (this will take several minutes) ---"
./gen_data_100m
if [ $? -ne 0 ]; then echo "Failed to run 100M generator"; exit 1; fi
echo ""

echo "========== Generation Complete =========="
echo ""
echo "Generated files:"
ls -lh profile_logits_*.bin
