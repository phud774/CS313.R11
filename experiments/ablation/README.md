# DGMAE Table 4 ablation

The experiment covers Roman-empire, Actor, Citeseer and Amazon Photo. It uses the GAT encoder, GAT decoder, projector and SCE implementation from the authors' pinned DGMAE repository. A frozen encoder supplies features to an Adam-trained logistic regression classifier. Validation accuracy selects the probe checkpoint, and the test set is evaluated after selection.

| Table 4 row | `--variant` | Active losses and target |
| --- | --- | --- |
| DGMAE | `full` | Feature and discrepancy, with adaptive edge selection |
| Variant 1 | `no-selection` | Feature and discrepancy, using the unfiltered graph |
| Variant 2 | `no-feature` | Discrepancy only, with adaptive edge selection |
| Variant 3 | `no-discrepancy` | Feature only |

## Protocol choice

`paper` implements Eqs. 1–13 and the four dataset settings in Table 8. It uses zeroed Bernoulli masks, attention before dropout, and the mean SCE on visible nodes for Eq. 12. The displayed normalization in Eq. 12 is ambiguous. Each retained loss keeps its Eq. 13 coefficient. The upstream config supplies architecture and optimizer settings omitted from Table 8.

The paper protocol defaults to `--discrepancy-neighborhood one-hop`, as in Kaggle v12. Use `--discrepancy-neighborhood one-hop-two-hop` for the neighborhood comparison. Given the one-hop operator $T$ from Eq. 5 or Eq. 9, the runner computes $T X$ and $T(T X)$ using the same edges. It averages their two SCE losses on visible nodes and keeps the original discrepancy weight. Appendix D.2 reports a combined result without specifying the combination operator. The equal-weight loss average is the choice for this comparison. Combined runs write under `paper/{smoke,full}/one-hop-two-hop/` so they can be compared with the original one-hop outputs.

`upstream` follows the public transductive implementation for fixed-size masking, masking noise, remasking at the decoder, post-dropout attention, normalized sigmoid selection scores, graph Laplacian powers from `configs.yml`, and 0.9/0.1 loss weights with discrepancy SCE exponent 4. The upstream repository does not ship Table 4 variant scripts. Here, the three variants independently disable the corresponding term or edge selection. Both protocols share a validation-only linear probe because the public Citeseer evaluator stores test accuracy in its best-validation variable. The raw artifacts also include final-epoch test accuracy.

## Splits and aggregation

- **Roman** uses the pinned original NPZ with 22,662 nodes and 32,927 raw edges. The public masks are 50/25/25. Table 7 says 48/32/20 and its Roman node and edge counts differ from the source NPZ. A full `paper` run requires an explicit choice between the public masks and fresh seeded 48/32/20 masks.
- **Actor** and **Citeseer** use DGL's public masks and the authors' preprocessing. Actor supplies 10 splits and Citeseer uses the fixed 120/500/1,000 split.
- **Amazon Photo** uses DGL's loader and ten seeded 10/10/80 splits. The published release has Photo settings but omits the dataset from its loader. The exact indices are unspecified.
- A full run defaults to 10 pretraining repetitions using seeds 0 through 9. The paper does not identify the seeds for Table 4. The `paper` protocol pairs each Actor or public Roman repetition with one predefined split. The `upstream` protocol probes all predefined splits per seed and averages them per seed, following the structure of the public evaluator. Both report the population standard deviation across the seed-level accuracy percentages, as in the public code.

## Kaggle notebook

The notebook starts in `configure` mode. Select `smoke` to run one pretraining epoch and two linear-probe epochs per selected dataset and variant. Select `full` to use the full epoch budgets and explicitly choose the protocol. The notebook accepts subsets of datasets, variants and seeds for splitting the runs across GPU sessions. Each completed seed gets a JSON artifact, and each dataset and variant gets an aggregate with the corresponding Table 4 reference for full runs. Keep the output paths or carry the result directories to a later session when extending a partial run.

The Kaggle environment uses Python 3.11, PyTorch 2.2.1 CUDA 12.1, and DGL 2.1.0. The notebook needs internet for the pinned checkouts, packages, and data. It reports the allocated GPUs and uses one CUDA device. Smoke-test results check execution and memory for the selected configuration. Full-run duration and accuracy need a GPU execution to verify.
