# StainID Feature Reference

This document is the exhaustive reference for the features emitted by the current `stainID` pipeline.

It covers:

- every per-site column written to `features/<TYPE>/<channel>/<image>_sites.csv`
- every image-level summary feature family written to `summary/<TYPE>_image_metrics.csv`
- the naming rules that generate all scale-specific and cross-channel feature combinations
- a practical interpretation guide for each feature family

## Scope And Naming Rules

### Channels

The supported channels are:

- `dab`
- `hematoxylin`
- `eosin`

For image-level summaries, each row is for one `(image, type, channel)` combination and also includes:

- `image`
- `type`
- `channel`
- `threshold_used`

### Regions

Several features use these region labels:

- `tissue`: all pixels in the valid tissue mask (`valid_mask`)
- `positive`: pixels positive for the current channel, restricted to tissue

### Intensity Spaces

- `norm`: robustly normalized oriented channel intensity in `[0, 1]`
- `raw`: raw oriented channel intensity on the OD-like HED scale

Interpretation:

- normalized features are easier to compare within an image or within a processing setting
- raw features preserve more absolute channel scale information across images when configs are fixed

### Scale Labels

Texture and granularity features use physical scale labels derived from:

- `mean_pixel_um = sqrt(pixel_width_um * pixel_height_um)`

Current defaults are:

- texture scales: `1`, `2`, `4` pixels
- granularity scales: `1`, `2`, `4`, `8` pixels

At the default pixel size `0.2738 x 0.2738 um`, these become:

- texture scales: `s0p274um`, `s0p548um`, `s1p1um`
- granularity scales: `s0p274um`, `s0p548um`, `s1p1um`, `s2p19um`

If you change pixel size, the scale labels in the output column names will change accordingly.

### Missing Values

Features can be `NaN` when:

- there are too few objects
- there are too few valid pixels
- a mask is empty
- a correlation/shape statistic is undefined for a near-constant or too-small sample

## Per-Site CSV Columns

Each connected component after segmentation cleanup and optional watershed splitting becomes one site.

The current per-site CSV has 42 columns.

### Identifiers And Geometry

| Column | Meaning | Interpretation |
|---|---|---|
| `label` | Connected-component label id within the current image/channel | Identifier only |
| `centroid_row_um` | Row centroid in microns | Physical vertical location |
| `centroid_col_um` | Column centroid in microns | Physical horizontal location |
| `area_um2` | Site area in square microns | Overall object size |
| `perimeter_um` | Boundary length in microns | Boundary extent; larger values often indicate larger or more irregular objects |
| `equivalent_diameter_um` | Diameter of a circle with the same area | Size as a single comparable diameter |
| `major_axis_length_um` | Major ellipse axis length | Long-axis extent |
| `minor_axis_length_um` | Minor ellipse axis length | Short-axis extent |
| `eccentricity` | Ellipse eccentricity | Near 0 is rounder; near 1 is more elongated |
| `solidity` | Area divided by convex hull area | Lower values suggest concavity or irregular outline |
| `extent` | Area divided by bounding-box area | Lower values suggest empty bbox space or irregularity |
| `orientation_deg` | Ellipse orientation in degrees | Preferred site orientation |
| `circularity` | `4*pi*area / perimeter^2` | Near 1 is round; lower is less circular |
| `roughness` | `(perimeter^2) / (4*pi*area)` | Reciprocal-style irregularity score; higher is rougher |
| `feret_diameter_um` | Maximum caliper diameter | Longest span across the object |
| `bbox_area_um2` | Bounding-box area | Size of enclosing rectangle |
| `bbox_fill_ratio` | Area divided by bbox area | Higher means tighter bbox occupancy |
| `bbox_aspect_ratio` | Bounding-box width / height | Shape anisotropy from bbox geometry |
| `elongation` | Major axis / minor axis | Higher means more elongated |
| `filled_area_um2` | Area after filling holes | Size ignoring internal holes |
| `convex_area_um2` | Convex-hull area | Size of convex envelope |
| `hole_area_um2` | Filled area minus original area | Internal void burden |
| `hole_fraction` | Hole area divided by filled area | Relative hole burden |
| `euler_number` | Components minus holes for the object | Lower values indicate more holes |

### Normalized Intensity Features

These are computed on the normalized oriented signal inside each object.

| Column | Meaning | Interpretation |
|---|---|---|
| `intensity_mean` | Mean normalized intensity | Average stain strength within the site |
| `intensity_std` | Standard deviation of normalized intensity | Within-object intensity heterogeneity |
| `intensity_median` | Median normalized intensity | Robust central stain strength |
| `intensity_iqr` | Interquartile range of normalized intensity | Robust spread of stain strength |
| `intensity_integrated` | Sum of normalized intensities | Approximate total normalized signal burden |
| `intensity_min` | Minimum normalized intensity | Weakest site pixel |
| `intensity_max` | Maximum normalized intensity | Strongest site pixel |
| `intensity_skew` | Skewness of normalized intensity distribution | Positive values indicate a long bright tail |
| `intensity_kurtosis` | Kurtosis of normalized intensity distribution | High values indicate heavier tails or peakedness |

### Raw Intensity Features

These are computed on the raw oriented HED/OD-like signal inside each object.

| Column | Meaning | Interpretation |
|---|---|---|
| `intensity_raw_mean` | Mean raw oriented intensity | Average absolute stain signal |
| `intensity_raw_std` | Standard deviation of raw intensity | Within-object raw heterogeneity |
| `intensity_raw_median` | Median raw intensity | Robust absolute stain strength |
| `intensity_raw_iqr` | Interquartile range of raw intensity | Robust spread on the raw scale |
| `intensity_raw_integrated` | Sum of raw intensities | Approximate total absolute signal burden |
| `intensity_raw_min` | Minimum raw intensity | Weakest raw pixel |
| `intensity_raw_max` | Maximum raw intensity | Strongest raw pixel |
| `intensity_raw_skew` | Skewness of raw intensity distribution | Tail asymmetry on the raw scale |
| `intensity_raw_kurtosis` | Kurtosis of raw intensity distribution | Tail heaviness or peakedness on the raw scale |

## Image-Level Summary Rows

The image-level summary contains:

- the 4 metadata columns added by `segment_stain.py`: `image`, `type`, `channel`, `threshold_used`
- the feature columns generated by `compute_channel_summary()`

The feature count produced by `compute_channel_summary()` is:

- `295 + 15 * (number_of_other_output_channels)`

Examples:

- if 1 channel is output: `295`
- if 2 channels are output: `310`
- if 3 channels are output: `325`

## Summary Family 1: Aggregated Site Features

The following site-level columns are summarized into two image-level columns each:

- `<site_feature>_median`
- `<site_feature>_iqr`

The site features included in this aggregation are:

- `area_um2`
- `perimeter_um`
- `equivalent_diameter_um`
- `major_axis_length_um`
- `minor_axis_length_um`
- `eccentricity`
- `solidity`
- `extent`
- `circularity`
- `roughness`
- `feret_diameter_um`
- `bbox_area_um2`
- `bbox_fill_ratio`
- `bbox_aspect_ratio`
- `elongation`
- `filled_area_um2`
- `convex_area_um2`
- `hole_area_um2`
- `hole_fraction`
- `euler_number`
- `intensity_mean`
- `intensity_raw_mean`
- `intensity_raw_median`
- `intensity_raw_iqr`
- `intensity_raw_integrated`

Interpretation:

- `_median` is the typical object value for the image
- `_iqr` is the object-to-object heterogeneity for that quantity

Important omission:

- `orientation_deg`, `intensity_std`, `intensity_median`, `intensity_integrated`, `intensity_min`, `intensity_max`, `intensity_skew`, `intensity_kurtosis`, and their raw counterparts are present per-site but are not currently aggregated at image level by this function

## Summary Family 2: Burden And Pixel-Distribution Features

### Always Present

| Feature | Meaning | Interpretation |
|---|---|---|
| `num_sites` | Number of connected components/sites | Object count for the current channel |
| `percent_area_stained` | Positive pixels divided by all pixels in the image | Gross stained fraction, including non-tissue background if present |
| `percent_tissue_area` | Tissue pixels divided by all pixels in the image | How much of the image is valid tissue |
| `positive_fraction_in_tissue` | Positive pixels divided by tissue pixels | Tissue-normalized stained burden |
| `tissue_area_um2` | Tissue area in square microns | Physical tissue area analyzed |
| `positive_area_um2` | Positive area in square microns | Physical stained area |
| `num_sites_per_mm2_tissue` | Site count divided by tissue area in mm² | Tissue-normalized object density |

### Pixel Distribution Feature Prefixes

These prefixes are used:

- `pixel_norm_tissue`
- `pixel_norm_positive`
- `pixel_raw_tissue`
- `pixel_raw_positive`

For each prefix, the following features exist:

- `<prefix>_mean`
- `<prefix>_sd`
- `<prefix>_iqr`
- `<prefix>_skew`
- `<prefix>_kurtosis`

Interpretation:

- `mean`: average signal level
- `sd`: overall spread
- `iqr`: robust spread
- `skew`: asymmetry of the intensity distribution
- `kurtosis`: peakedness / heavy-tail behavior

### Pixel H-Score Style Features

These are generated only for normalized pixel intensities:

- `pixel_norm_tissue_frac_0`
- `pixel_norm_tissue_frac_1plus`
- `pixel_norm_tissue_frac_2plus`
- `pixel_norm_tissue_frac_3plus`
- `pixel_norm_tissue_hscore`
- `pixel_norm_positive_frac_0`
- `pixel_norm_positive_frac_1plus`
- `pixel_norm_positive_frac_2plus`
- `pixel_norm_positive_frac_3plus`
- `pixel_norm_positive_hscore`

Bin definitions:

- `0`: `[0.00, 0.25)`
- `1plus`: `[0.25, 0.50)`
- `2plus`: `[0.50, 0.75)`
- `3plus`: `[0.75, 1.00]`

H-score formula:

- `100 * (1*frac_1plus + 2*frac_2plus + 3*frac_3plus)`

Interpretation:

- higher H-score means a larger share of pixels occupy stronger normalized intensity bins

## Summary Family 3: Object-Score Features

These are H-score style summaries over object-level intensities.

Generated features:

- `object_norm_mean_frac_0`
- `object_norm_mean_frac_1plus`
- `object_norm_mean_frac_2plus`
- `object_norm_mean_frac_3plus`
- `object_norm_mean_hscore`
- `object_norm_integrated_frac_0`
- `object_norm_integrated_frac_1plus`
- `object_norm_integrated_frac_2plus`
- `object_norm_integrated_frac_3plus`
- `object_norm_integrated_hscore`

Definitions:

- `object_norm_mean_*`: bins object-level `intensity_mean`
- `object_norm_integrated_*`: bins object-level `intensity_integrated` after dividing by the image-wise maximum finite integrated value

Interpretation:

- these measure how many objects are weak, intermediate, or strong rather than how many pixels are
- they emphasize lesion/site severity at the object level

## Summary Family 4: Topology Features

Exact features:

- `topology_component_count`
- `topology_component_density_per_mm2`
- `topology_largest_component_frac`
- `topology_component_area_median_um2`
- `topology_component_area_iqr_um2`
- `topology_euler_number`
- `topology_hole_area_frac`
- `topology_hole_count`
- `topology_boundary_length_um`
- `topology_fractal_dimension`

Interpretation:

- `component_count`: number of connected positive regions
- `component_density_per_mm2`: component count normalized by tissue area
- `largest_component_frac`: dominance of the biggest connected region; high values suggest consolidation into one large patch
- `component_area_median_um2`: typical component size
- `component_area_iqr_um2`: component-size heterogeneity
- `euler_number`: components minus holes; lower values indicate more holes/voids
- `hole_area_frac`: fraction of positive area occupied by internal holes
- `hole_count`: approximate number of holes across the mask
- `boundary_length_um`: total contour length; larger values often mean more fragmented or more extensive patterns
- `fractal_dimension`: boundary/area complexity across scales; higher values suggest more space-filling or more intricate morphology

## Summary Family 5: Tile Heterogeneity Features

Tile heterogeneity is computed on `4x4` and `8x8` grids.

For each `g in {4, 8}`, the following columns exist:

- `tile_g<g>_mass_entropy`
- `tile_g<g>_mass_gini`
- `tile_g<g>_positive_frac_mean`
- `tile_g<g>_positive_frac_sd`
- `tile_g<g>_positive_frac_cv`
- `tile_g<g>_positive_frac_iqr`
- `tile_g<g>_intensity_mean_sd`
- `tile_g<g>_intensity_mean_iqr`

Interpretation:

- `mass_entropy`: how evenly positive pixels are distributed across tiles; higher means more spatial spread
- `mass_gini`: inequality of positive mass across tiles; higher means more concentrated in a few tiles
- `positive_frac_*`: distribution of tile-level stained fraction
- `intensity_mean_*`: distribution of tile-level mean normalized intensity

Rule of thumb:

- high entropy + low gini suggests diffuse spread
- low entropy + high gini suggests focal concentration

## Summary Family 6: Spatial Organization Features

Exact features:

- `density_per_mm2`
- `nn_mean_um`
- `nn_sd_um`
- `clark_evans_R`
- `clark_evans_z`
- `grid_vmr`
- `grid_cv`
- `knn_k1_mean_um`
- `knn_k3_mean_um`
- `knn_k5_mean_um`
- `delaunay_edge_mean_um`
- `delaunay_edge_sd_um`
- `delaunay_edge_iqr_um`
- `delaunay_triangle_area_mean_um2`
- `delaunay_triangle_area_sd_um2`
- `mst_edge_mean_um`
- `mst_edge_sd_um`
- `mst_edge_iqr_um`
- `graph_radius_um`
- `graph_mean_degree`
- `graph_sd_degree`
- `graph_components`
- `graph_largest_component_frac`
- `graph_singleton_frac`

Interpretation:

- `density_per_mm2`: object density per full image area in physical units
- `nn_mean_um`: typical nearest-neighbor distance; smaller values suggest tighter packing
- `nn_sd_um`: heterogeneity in nearest-neighbor distance
- `clark_evans_R`: clustering/dispersion score; `<1` clustered, `~1` random, `>1` dispersed
- `clark_evans_z`: z-score for the Clark-Evans deviation
- `grid_vmr`: variance-to-mean ratio of counts over a grid; `>1` suggests clustering
- `grid_cv`: coefficient of variation of grid counts; higher means less even spatial distribution
- `knn_k1_mean_um`, `knn_k3_mean_um`, `knn_k5_mean_um`: mean distances to the 1st, 3rd, and 5th neighbors
- `delaunay_edge_*`: spacing variation in the Delaunay triangulation
- `delaunay_triangle_area_*`: characteristic local triangle area; larger values suggest sparser layouts
- `mst_edge_*`: spacing along the minimum spanning tree; useful for network-like spacing structure
- `graph_radius_um`: radius used for the proximity graph, defined as `1.5 * mean nearest-neighbor distance` with a minimum of `1.0 um`
- `graph_mean_degree`: average number of neighbors in the proximity graph
- `graph_sd_degree`: heterogeneity in graph degree
- `graph_components`: number of connected components in the proximity graph
- `graph_largest_component_frac`: fraction of sites in the largest connected graph component
- `graph_singleton_frac`: fraction of sites with no neighbors in the graph

## Summary Family 7: GLCM Texture Features

### What GLCM Means

GLCM stands for gray-level co-occurrence matrix.

For a chosen pixel offset, it counts how often intensity level `i` occurs next to intensity level `j`.
The resulting matrix summarizes local texture, not just intensity magnitude.

In this pipeline:

- the image is quantized to `32` gray levels inside the chosen mask
- GLCMs are computed at distances `1`, `2`, and `4` pixels
- four angles are used: `0`, `45`, `90`, and `135` degrees

### Texture Prefixes

The exact prefixes are:

- `texture_tissue_norm`
- `texture_tissue_raw`
- `texture_positive_norm`
- `texture_positive_raw`

### Texture Properties

The exact texture properties are:

- `contrast`
- `dissimilarity`
- `homogeneity`
- `asm`
- `energy`
- `correlation`
- `entropy`

### Exact Texture Naming Rules

For every combination of:

- `prefix in {texture_tissue_norm, texture_tissue_raw, texture_positive_norm, texture_positive_raw}`
- `scale in {s0p274um, s0p548um, s1p1um}` at default pixel size
- `property in {contrast, dissimilarity, homogeneity, asm, energy, correlation, entropy}`

the pipeline emits:

- `<prefix>_<scale>_<property>_mean`

For every combination of:

- `prefix`
- `property`

the pipeline also emits:

- `<prefix>_<property>_mean`
- `<prefix>_<property>_sd`

This is exhaustive for the texture family.

### Texture Interpretation

| Property | Meaning | Typical interpretation |
|---|---|---|
| `contrast` | Emphasizes large gray-level differences between neighbors | Higher means harsher local variation / coarser texture |
| `dissimilarity` | Mean absolute gray-level difference between neighbors | Higher means more local variation |
| `homogeneity` | Weights similar neighbors more strongly | Higher means smoother, more locally uniform texture |
| `asm` | Angular second moment | Higher means more orderly or repetitive texture |
| `energy` | Square root of `asm` | Higher means stronger regularity / lower randomness |
| `correlation` | Linear dependency of neighboring gray levels | Higher magnitude means more structured neighboring relationships |
| `entropy` | Randomness / unpredictability of co-occurrence patterns | Higher means more complex or disordered texture |

Interpretation of scales:

- smaller scales reflect fine-grained local texture
- larger scales reflect coarser spatial texture

Interpretation of region/intensity combinations:

- `tissue` vs `positive`: all valid tissue vs only positive pixels
- `norm` vs `raw`: normalized relative contrast vs raw absolute signal contrast

## Summary Family 8: Granularity Features

Granularity features measure how much image mass disappears after grayscale openings with increasing radii.

Exact prefixes:

- `granularity_tissue_norm`
- `granularity_positive_norm`

Exact scale labels at the default pixel size:

- `s0p274um`
- `s0p548um`
- `s1p1um`
- `s2p19um`

Exact naming rule:

For every combination of:

- `prefix in {granularity_tissue_norm, granularity_positive_norm}`
- `scale in {s0p274um, s0p548um, s1p1um, s2p19um}`

the pipeline emits:

- `<prefix>_<scale>`

This is exhaustive for the granularity family.

Interpretation:

- small-scale granularity features respond to fine speckled structure
- large-scale granularity features respond to broader clumps or coarse blobs
- higher values at a given scale mean more of the signal lives at that spatial size

## Summary Family 9: Cross-Channel Features

Cross-channel features are emitted for each `other_channel` that is present in `channels_to_output`, excluding the current summary row's own channel.

If the current row is `channel = dab` and the other channels are `hematoxylin` and `eosin`, the pipeline emits the same suffix set twice:

- `cross_with_hematoxylin_<suffix>`
- `cross_with_eosin_<suffix>`

The exact suffixes are:

- `jaccard`
- `dice`
- `frac_self_overlap`
- `frac_other_overlap`
- `positive_area_ratio`
- `touching_object_fraction`
- `overlapping_object_fraction`
- `nn_other_mean_um`
- `corr_raw_tissue`
- `corr_raw_union`
- `corr_norm_tissue`
- `annulus_0_5um_norm_mean`
- `annulus_5_15um_norm_mean`
- `annulus_15_30um_norm_mean`
- `annulus_near_far_ratio`

This is exhaustive for the cross-channel family.

Interpretation:

| Suffix | Meaning | Typical interpretation |
|---|---|---|
| `jaccard` | Intersection / union of positive masks | Overall overlap similarity |
| `dice` | `2*intersection / (area_a + area_b)` | Overlap similarity with a different normalization |
| `frac_self_overlap` | Fraction of current channel area overlapping the other channel | How much of the current stain sits on the other stain |
| `frac_other_overlap` | Fraction of the other channel area overlapping the current channel | Reverse overlap burden |
| `positive_area_ratio` | Current positive area / other positive area | Relative abundance of the two channels |
| `touching_object_fraction` | Fraction of current objects touching the other channel after a 2-pixel dilation | Near-contact rate between stains |
| `overlapping_object_fraction` | Fraction of current objects with true overlap | Strict co-localization at the object level |
| `nn_other_mean_um` | Mean nearest distance from current object centroids to other-channel centroids | Physical proximity between object populations |
| `corr_raw_tissue` | Correlation of raw signals over tissue pixels | Whether raw channel intensities covary in tissue |
| `corr_raw_union` | Correlation of raw signals over the union of the two masks | Covariation focused on positive areas |
| `corr_norm_tissue` | Correlation of normalized signals over tissue pixels | Relative covariation after normalization |
| `annulus_0_5um_norm_mean` | Mean current normalized intensity within 0-5 um of the other mask | Immediate peri-other-channel intensity |
| `annulus_5_15um_norm_mean` | Mean current normalized intensity within 5-15 um | Near-neighborhood intensity |
| `annulus_15_30um_norm_mean` | Mean current normalized intensity within 15-30 um | Farther-neighborhood intensity |
| `annulus_near_far_ratio` | `(0-5 um mean) / (15-30 um mean)` | Enrichment near the other channel relative to farther away |

## Quick Interpretation Guide By Biological Question

If you want to ask:

- "How much stain is there?"
  - use `positive_fraction_in_tissue`, `positive_area_um2`, `percent_area_stained`, pixel H-scores
- "How many objects are there?"
  - use `num_sites`, `num_sites_per_mm2_tissue`, `density_per_mm2`
- "Are the objects large, elongated, hollow, or irregular?"
  - use site-shape medians/IQRs and the topology family
- "Is the staining smooth or patchy?"
  - use GLCM texture features and tile heterogeneity
- "Is the signal fine-grained or coarse?"
  - use granularity features
- "Are the objects clustered or dispersed?"
  - use nearest-neighbor, Clark-Evans, grid, Delaunay, MST, and graph features
- "Do two stains co-localize or sit near each other?"
  - use cross-channel overlap, touching, distance, correlation, and annulus features

## Code Pointers

The feature definitions come from:

- [`object_features.py`](/Volumes/sky_4t/alzh_brain_TMA/stainID/scripts/features/object_features.py)
- [`mask_features.py`](/Volumes/sky_4t/alzh_brain_TMA/stainID/scripts/features/mask_features.py)
- [`spatial_features.py`](/Volumes/sky_4t/alzh_brain_TMA/stainID/scripts/features/spatial_features.py)
- [`texture_features.py`](/Volumes/sky_4t/alzh_brain_TMA/stainID/scripts/features/texture_features.py)
- [`cross_channel_features.py`](/Volumes/sky_4t/alzh_brain_TMA/stainID/scripts/features/cross_channel_features.py)
- [`pipeline.py`](/Volumes/sky_4t/alzh_brain_TMA/stainID/scripts/features/pipeline.py)
