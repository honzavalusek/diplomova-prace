# Sakoe-Chiba Window Constraint for DTW

## Overview

The Sakoe-Chiba window constraint has been implemented in the `SubsequenceDTWMatcher` to:
- **Reduce computational cost** by limiting the search space
- **Prevent pathological warping paths** where distant frames are incorrectly matched
- **Reflect physical speech constraints** by maintaining reasonable temporal relationships

## Usage

### 1. In Code (SubsequenceDTWMatcher)

```python
from src.matching import SubsequenceDTWMatcher

# Initialize with default window for all matches
matcher = SubsequenceDTWMatcher(window=25)

# Or set no window constraint (slower, less reliable)
matcher = SubsequenceDTWMatcher(window=None)

# Override window for specific match
result = matcher.match(query_emb, ref_emb, window=50)

# Use with top-k matching
results = matcher.match_top_k(query_emb, ref_emb, k=3, window=30)
```

### 2. In Search Script

```bash
# Use window constraint of 25 frames
python scripts/01_search.py \
    --query query.wav \
    --corpus corpus.wav \
    --window 25

# No window constraint (default if --window is not specified)
python scripts/01_search.py \
    --query query.wav \
    --corpus corpus.wav
```

## Recommended Values

For speech recognition tasks:

- **No constraint** (`window=None`): Slowest, allows extreme time warping (not recommended)
- **Loose constraint** (`window=50-100`): Flexible alignment, slower computation
- **Moderate constraint** (`window=20-40`): Good balance for most speech tasks ✓ **Recommended**
- **Tight constraint** (`window=5-15`): Fast, but may miss valid alignments if speech rate varies

### Rule of Thumb

A good starting point is:
```
window = 0.5 * query_length
```

For a 2-second query (~100 frames at 50 FPS):
```
window = 50 frames ≈ 1 second of temporal deviation
```

## Technical Details

### Implementation

The window constraint is implemented using the `dtaidistance` library, which is specifically designed for subsequence DTW:

```python
from dtaidistance.subsequence.dtw import subsequence_alignment

# Perform subsequence alignment
sa = subsequence_alignment(query, reference, use_c=True)

# Apply Sakoe-Chiba window constraint
sa.settings.window = window_size

# Get best match
best = sa.best_match()
```

### What It Does

The Sakoe-Chiba window restricts the DTW alignment path to stay within a diagonal band of width `2*window + 1` around the main diagonal of the DTW cost matrix.

```
Reference frames
│ ▓▓▓░░░░░░░░
│ ░▓▓▓░░░░░░░
│ ░░▓▓▓░░░░░░  ← window=2 band
│ ░░░▓▓▓░░░░░
│ ░░░░▓▓▓░░░░
│ ░░░░░▓▓▓░░░
└─────────────
  Query frames
```

This prevents mapping query frame 0 to reference frame 50, for example, which would be physically implausible for continuous speech.

## Performance Impact

| Window Size | Speed | Alignment Quality |
|------------|-------|------------------|
| None       | 1.0x  | Unreliable (pathological paths) |
| 50         | ~2x   | Good (flexible) |
| 25         | ~4x   | Good (recommended) |
| 10         | ~8x   | Fair (may be too strict) |

*Speed multipliers are approximate and depend on sequence lengths.*

## References

- Sakoe, H., & Chiba, S. (1978). "Dynamic programming algorithm optimization for spoken word recognition"
- dtaidistance documentation: https://dtaidistance.readthedocs.io/
- Meert, W., & Wannes, M. (2020). "DTAIDistance: Dynamic Time Warping"
