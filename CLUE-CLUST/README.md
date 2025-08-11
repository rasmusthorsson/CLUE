# LSHDBSCAN

LSHDBSCAN is a high-performance, concurrent implementation of DBSCAN clustering that uses Locality-Sensitive Hashing (LSH) to significantly improve performance on large, high-dimensional datasets.

## Algorithms Included

1. **ConcurrentLSHDBSCAN**: The main algorithm that uses LSH and multi-threading
2. **VanillaDBSCAN**: Traditional DBSCAN implementation (for comparison)
3. **VanillaDBSCANLSH**: Single-threaded DBSCAN with LSH (for comparison)

## Usage

```bash
java -jar lshdbscan.jar -c <command> -f <inputfile> [options]
```

### Commands

- `lshdbscan`: Run the concurrent LSH-DBSCAN algorithm (default)
- `vanilla`: Run traditional DBSCAN
- `vanillalsh`: Run DBSCAN with LSH (single-threaded)
- `optimize`: Run parameter optimization to find the best epsilon and minPts values

### Options

- `-a`: Use angular distance metric (default is Euclidean)
- `-dtw`: Use Dynamic Time Warping distance metric
- `-window <size>`: Window size for DTW calculation (default: 10)
- `-nolb`: Disable LB_Keogh lower bound for DTW
- `-m <minPts>`: Minimum points for core cluster (default: 500)
- `-e <epsilon>`: Epsilon value for neighborhood (default: 0.001)
- `-t <threads>`: Number of threads (default: available processors)
- `-L <hashTables>`: Number of hash tables (default: 10)
- `-M <hyperplanes>`: Number of hyperplanes per hash table (default: 10)
- `-b <baselineFile>`: Baseline clustering file for accuracy comparison

### Examples

```bash
# Run LSH-DBSCAN with default parameters
java -jar lshdbscan.jar -c lshdbscan -f dataset.csv

# Run with custom parameters
java -jar lshdbscan.jar -c lshdbscan -f dataset.csv -e 0.1 -m 20 -L 15 -M 12 -t 8

# Run with angular distance metric
java -jar lshdbscan.jar -c lshdbscan -f dataset.csv -a

# Find optimal parameters
java -jar lshdbscan.jar -c optimize -f dataset.csv

# Compare with traditional DBSCAN
java -jar lshdbscan.jar -c vanilla -f dataset.csv
```

## Input Data Format

The application accepts CSV files with the following format:

- First column: Point ID
- Remaining columns: Feature values
- Optional header row (enabled by default)

Example:
```
ID,Dimension1,Dimension2,Dimension3
point1,1.234,5.678,9.012
point2,2.345,6.789,0.123
...
```

## Output

The application generates several output files:

- `<inputfile>_<hashtables>_<hyperplanes>_<threads>.idx_concurrentlshdbscan`: Clustering results with point IDs and assigned cluster labels
- `<inputfile>_<hashtables>_<hyperplanes>_<threads>.metadata`: Cluster metadata including centroids, bounding boxes, and statistics
- `<inputfile>_k_distances.csv`: K-distance plot data (when using parameter optimization)
- `<inputfile>_k_distances.png`: K-distance plot visualization
- `<inputfile>_parameter_results.csv`: Parameter exploration results

## Performance Considerations

- **Number of Hash Tables (L)**: More tables improve accuracy but increase memory usage and preprocessing time
- **Number of Hyperplanes per Table (M)**: More hyperplanes improve hash function precision but may reduce recall
- **Number of Threads**: Should be set according to available CPU cores
- **Epsilon and MinPts**: Critical parameters that control cluster density; use the optimization tool to find good values

## Algorithm Details

LSHDBSCAN works in the following stages:

1. **LSH Index Construction**: Builds hash tables with random hyperplanes
2. **Bucket Population**: Hashes all points into appropriate buckets
3. **Core Bucket Identification**: Identifies dense buckets that can form cluster cores
4. **Merge Task Identification**: Determines which core buckets should be merged
5. **Relabeling**: Assigns final cluster labels to all points

The algorithm uses a Union-Find data structure for efficient cluster merging and path compression for optimal performance.