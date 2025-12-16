# ZINB Plot Scaling - Detailed Fixes

## The Problem

The ZINB plots had ambiguous scaling that could lead to misinterpretation of results. The issue was that multiple parameters were being plotted on the same axes or referenced without clear indication of what scale they were in.

## Parameters in ZINB Models

### Data Scales:
1. **Normalized log counts** - log-transformed and normalized values (-∞ to +∞ range typically -5 to +5)
2. **Raw counts** - original count values (0, 1, 2, 3, ... integer values)
3. **ZINB parameters** - θ, π, μ (each has their own scale)

### Variables in Input:
- `all_originals` = Normalized log counts
- `all_reconstructions_mu` = Model's predicted μ (in normalized log space)
- `all_raw_counts` = Original raw counts (used only for zero detection)
- `all_theta` = Dispersion parameters (typically 0.1 to 10.0)
- `all_pi` = Zero-inflation probabilities (0.0 to 1.0)

## Specific Fixes

### Fix 1: Parameter Distribution Plots

**Before:**
```
Axes labels:
- "Mean (μ)"  ← Unclear if normalized or raw
- Distribution of Zero-inflation π  ← No context on range
```

**After:**
```
Axes labels:
- "Mean (μ) [Normalized Log Space]"
- "Zero-inflation Probability (π)" 
  with note: "(Note: 0 < π < 1, represents P(zero))"
```

Plot titles now include informative notes:
```python
axes[0, 0].set_title(f'{model_type}: Distribution of Dispersion Parameter θ\n(Note: θ > 0, represents overdispersion)')
axes[0, 1].set_title(f'{model_type}: Distribution of Zero-inflation π\n(Note: 0 < π < 1, represents P(zero))')
```

### Fix 2: Actual vs Predicted Plot

**Before:**
```
axes[0].set_ylabel('Predicted Mean (μ)')
```

**After:**
```
axes[0].set_ylabel('Predicted Mean (μ) [Normalized Log Space]')
```

And consistent labeling:
```python
axes[0].set_xlabel('Actual (Normalized Log Counts)')
axes[0].set_ylabel('Predicted Mean (μ) [Normalized Log Space]')
```

### Fix 3: Zero-Inflation Analysis

**Before:**
```
axes[0, 0].set_title(f'{model_type}: π Distribution by Actual Zeros')
# Unclear: actual zeros in what scale?
```

**After:**
```
axes[0, 0].set_title(f'{model_type}: π Distribution by Actual Zeros\n(Note: Uses raw counts for zero detection)')
# Now clearly indicates raw counts are used
```

And axis labels clarify:
```python
axes[1, 1].set(ylabel='Actual (from raw counts)', ...)
```

### Fix 4: Example Reconstructions

**Before:**
```
ax.set_ylabel('Value')
# Doesn't indicate scale
```

**After:**
```
ax.set_ylabel('Value [Normalized Log Space]')
# Clearly shows what scale is being used
```

Plot title clarifies:
```python
ax.set_title(f'{model_type}: Example {i+1} - Reconstruction')
# Legend shows what each line represents
ax.plot(positions, all_originals[i], label='Actual (Normalized)', ...)
ax.plot(positions, all_reconstructions_mu[i], label='Predicted μ (Normalized)', ...)
```

### Fix 5: JSON Metadata

**Before:**
```json
{
  "summary_statistics": {
    "original": {...},
    "predicted_mu": {...}
  }
}
```
No indication of scale.

**After:**
```json
{
  "scaling_notes": {
    "all_originals": "Normalized log counts",
    "all_reconstructions_mu": "Predicted mean parameter μ in normalized log space",
    "all_raw_counts": "Raw count data before normalization (used for zero detection)",
    "all_theta": "Dispersion parameter (θ > 0)",
    "all_pi": "Zero-inflation probability (0 < π < 1, represents P(zero))"
  },
  "summary_statistics": {
    "original": {...},
    "predicted_mu": {...}
  }
}
```

Now includes complete scaling information!

## Practical Impact

### Before Fixes:
❌ User sees plot with "Predicted Mean" axis label  
❌ Doesn't know if values are in normalized or raw scale  
❌ May incorrectly compare to raw count predictions  
❌ JSON file doesn't document scale  
❌ Confusion about what "zero" detection uses  

### After Fixes:
✓ Plot axes clearly labeled with scale: "[Normalized Log Space]"  
✓ JSON file includes "scaling_notes" explaining everything  
✓ Plot titles include clarifying notes about parameters  
✓ Zero-inflation analysis explicitly shows it uses raw counts  
✓ Example plots labeled with what each line represents  
✓ Consistent documentation across all ZINB functions  

## Code Examples

### Example 1: Clearer Axis Labels
```python
# Before
axes[0].set_ylabel('Predicted Mean (μ)')

# After
axes[0].set_ylabel('Predicted Mean (μ) [Normalized Log Space]')
```

### Example 2: Informative Titles
```python
# Before
axes[0, 0].set_title('Distribution of Dispersion Parameter θ')

# After
axes[0, 0].set_title('Distribution of Dispersion Parameter θ\n(Note: θ > 0, represents overdispersion)')
```

### Example 3: Clear Legend in Reconstructions
```python
# Before
ax.plot(positions, all_originals[i], label='Actual', ...)

# After
ax.plot(positions, all_originals[i], label='Actual (Normalized)', ...)
ax.plot(positions, all_reconstructions_mu[i], label='Predicted μ (Normalized)', ...)
```

### Example 4: JSON Metadata
```python
# Added to metrics_to_save:
'scaling_notes': {
    'all_originals': 'Normalized log counts',
    'all_reconstructions_mu': 'Predicted mean parameter μ in normalized log space',
    'all_raw_counts': 'Raw count data before normalization (used for zero detection)',
    'all_theta': 'Dispersion parameter (θ > 0)',
    'all_pi': 'Zero-inflation probability (0 < π < 1, represents P(zero))'
}
```

## Validation Checklist

✓ All ZINB plot functions use clear scale labels  
✓ Docstrings include "IMPORTANT: SCALING NOTES" section  
✓ JSON output includes scaling_notes metadata  
✓ Example plots clearly label what each line is  
✓ Zero-inflation analysis notes it uses raw counts  
✓ Parameter distributions explain what each value range means  
✓ Axes labels are consistent across all plots  
✓ Plot titles include clarifying information  

## Summary

The ZINB plots now have **correct and clear scaling** throughout:
- Axis labels explicitly show data scale
- Docstrings explain scaling for each parameter  
- JSON output includes scaling metadata
- Plots have informative titles with notes
- Zero-inflation analysis clearly indicates use of raw counts
- Legends show what each line represents
- Consistent labeling across all functions
