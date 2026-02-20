"""Artifact contracts for code and visualization generation."""

code_artifact_contract = {
    "uncertainty_estimator": "bootstrap",
    "n_draws": 1000,
    "confidence_level": 0.95,
}

vis_artifact_contract = {
    "output_format": "altair",
    "encodings": ["x", "y", "color", "row", "column"],
    "continuous_markings": ["point", "area"],
}
