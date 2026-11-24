"""Settings for the DGMAE Table 4 ablation and the public code comparison."""

DATASETS = {"roman": "roman", "actor": "act", "citeseer": "citeseer", "photo": "photo"}
VARIANTS = ("full", "no-selection", "no-feature", "no-discrepancy")

# Appendix C, Table 8. Remaining settings come from the pinned configs.yml.
PAPER_SETTINGS = {
    "roman": {"discrepancy_weight": 0.4, "gamma2": 3, "mask_rate": 0.75},
    "actor": {"discrepancy_weight": 0.7, "gamma2": 3, "mask_rate": 0.9},
    "citeseer": {"discrepancy_weight": 0.1, "gamma2": 4, "mask_rate": 0.5},
    "photo": {"discrepancy_weight": 0.1, "gamma2": 5, "mask_rate": 0.5},
}

# Published accuracy percentages in Table 4, in DATASETS order.
PAPER_ACCURACY = {
    "full": [(76.62, 0.41), (36.61, 0.74), (73.82, 0.64), (93.96, 0.41)],
    "no-selection": [(75.35, 0.30), (33.41, 1.07), (73.03, 0.52), (93.40, 0.42)],
    "no-feature": [(36.71, 0.50), (28.14, 1.12), (73.05, 0.23), (93.15, 0.41)],
    "no-discrepancy": [(75.04, 0.44), (35.16, 0.75), (58.02, 1.33), (92.88, 0.45)],
}

UPSTREAM_URL = "https://github.com/zhengziyu77/DGMAE.git"
UPSTREAM_COMMIT = "0e770864b5283b01e2597d6f092e8a812aa5031a"
ROMAN_SOURCE_COMMIT = "a431395582e929d88271309716bea4fe24ce6318"
ROMAN_SHA256 = "a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428"
CITESEER_SOURCE_COMMIT = "af6abf468a772cc055bec88c52e4d6f51b7c37e3"
DATA_SOURCES = {
    "roman": f"yandex-research/heterophilous-graphs@{ROMAN_SOURCE_COMMIT}",
    "actor": "DGL 2.1.0 ActorDataset (dataset/actor.zip)",
    "citeseer": f"kimiyoung/planetoid@{CITESEER_SOURCE_COMMIT}",
    "photo": "DGL 2.1.0 AmazonCoBuyPhotoDataset (dataset/amazon_co_buy_photo.zip)",
}
