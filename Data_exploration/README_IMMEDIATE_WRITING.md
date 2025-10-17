# Immediate Result Writing for Memory Efficiency

## What Changed

The `polynomial_regression.py` script now writes results to disk **immediately** as each worker process completes, rather than holding all results in memory until the end.

## Benefits

### 1. **Reduced Memory Usage**
- Results are written to disk and freed from memory immediately
- No memory spike at the end when writing all results
- Main process doesn't accumulate large result objects

### 2. **Progress Tracking**
- Each dataset gets its own output files as it completes
- You can monitor progress by checking the output directory
- Progress counter shows `(completed/total)` in real-time

### 3. **Fault Tolerance**
- If the script crashes, completed results are already saved
- You don't lose hours of computation
- Can resume by skipping already-processed datasets (future enhancement)

### 4. **Better for HPC Clusters**
- Reduces risk of out-of-memory errors
- Works better with job time limits
- Individual results available before job completes

## Output Files

For each dataset, three files are created immediately:

1. **`{dataset_name}_poly_result.pickle`** - Full statsmodels result object
2. **`{dataset_name}_summary.txt`** - Human-readable summary with parameters and fit statistics
3. **`all_results_piecewise_{args}_separate_{args}_degree_{args}.txt`** - Final combined summary (created at end)

### Example Output Structure:
```
Data_exploration/results/regression/poly/
├── dnrp1-1-a_poly_result.pickle
├── dnrp1-1-a_summary.txt
├── dnrp1-1-b_poly_result.pickle
├── dnrp1-1-b_summary.txt
├── FD7_1_poly_result.pickle
├── FD7_1_summary.txt
├── combined_poly_result.pickle
└── all_results_piecewise_True_separate_True_degree_3.txt
```

## Memory Flow

### Before (Holding All Results):
```
Worker 1 → Result 1 ┐
Worker 2 → Result 2 ├─→ Accumulate in memory → Write all at end
Worker 3 → Result 3 ┘                          ↑ MEMORY SPIKE
```

### After (Immediate Writing):
```
Worker 1 → Result 1 → Write to disk → Free memory
Worker 2 → Result 2 → Write to disk → Free memory  
Worker 3 → Result 3 → Write to disk → Free memory
                      ↑ CONSTANT LOW MEMORY
```

## New Parameters

The `perform_regression_on_datasets()` function has two new parameters:

- **`output_dir`**: Directory where results are saved (default: `"Data_exploration/results/regression/poly"`)
- **`write_immediately`**: Whether to write results as they complete (default: `True`)

## Console Output

You'll see progress like this:

```
Found 45 datasets to process
✓ Completed (1/45): dnrp1-1-a
✓ Completed (2/45): FD7_1
✓ Completed (3/45): dnrp1-1-b
...
✓ Completed (45/45): yWT03a_5
```

## Individual Summary File Format

Each `{dataset_name}_summary.txt` contains:

```
Dataset: dnrp1-1-a
============================================================

Parameters:
inflate_const        -2.456
inflate_Nuc_c         0.123
...

Log-Likelihood: -12345.67
AIC: 24789.34
BIC: 24912.56
```

## Usage

No changes needed to how you run the script:

```bash
python Data_exploration/polynomial_regression.py --piecewise --separate --degree 3
```

The immediate writing happens automatically!

## Performance Impact

- **Disk I/O**: Minimal - writing happens while other workers are still computing
- **Wall time**: Nearly identical (writing happens in parallel with computation)
- **Memory**: Significantly reduced - no accumulation of results

## For HPC Users

This is especially helpful when:
- Running many datasets (40+ samples)
- Limited memory per node
- Long-running jobs (can check intermediate results)
- Job might hit time limit (partial results saved)

You can even check results while the job is running:
```bash
# Count completed datasets
ls Data_exploration/results/regression/poly/*_summary.txt | wc -l

# View a specific result
cat Data_exploration/results/regression/poly/dnrp1-1-a_summary.txt
```
