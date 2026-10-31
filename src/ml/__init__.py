"""Learning-to-rank layer (recsys-style), fully offline.

Retrieval (cheap, deterministic) -> ranking (LightGBM LambdaRank).

Nothing here calls external services: every feature comes from the local
VCF + VEP annotation + HPO config. The labeled training DF is yours
(ClinVar-derived or curated); the trained model file is local.
"""
