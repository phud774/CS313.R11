# Figure 8

Run the three encoders on Roman, Squirrel and Actor. The runner uses one
undirected edge per pair of differently labeled nodes and evaluates the
same pairs for all representations. Labels enter only this analysis. No
linear probe is trained. Outputs are a PNG, a PDF, a compressed NumPy file
per dataset with ordered node pairs and raw cosine values, and metadata
with the seed, model settings, source revisions and input hashes.

The paper does not specify the visualization seed, pair sampling, KDE
bandwidth or exact baseline settings. This run uses seed 0, all qualifying
edges, and SciPy's default KDE bandwidth. DGMAE uses the team's
reimplementation and node config with the one-hop discrepancy objective.
GraphMAE uses the official encoder, decoder and objective with the
corresponding DGMAE training budget and encoder settings. GREET uses its
official Actor and Squirrel scripts' settings and its generic defaults on
Roman, with sparse adjacency and row-blocked full-batch InfoNCE for memory.
GREET uses its original feature normalization for Squirrel. All models
use the same undirected graph and preserve its node order.

The Kaggle notebook runs one epoch per encoder and dataset in smoke mode.
Full mode uses the documented epoch budgets and writes its outputs to a
separate directory. Compare the curve shapes with the published figure
under the assumptions above.
