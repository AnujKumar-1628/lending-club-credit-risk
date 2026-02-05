"""
this is the Repository scaffold for lending-club-credit-risk project.

"""

from pathlib import Path

PROJECT_STRUCTURE = {
    "docs": [
        "00_overview.md",
        "01_business_problem.md",
        "02_target_definition.md",
        "03_pipeline_and_features.md",
        "04_models.md",
        "05_evaluation.md",
        "06_artifacts_and_versioning.md",
        "07_inference_api.md",
        "08_docker_and_environment.md",
        "09_testing.md",
        "10_limitations_and_future_work.md",
    ],
    "data": {
        "raw": [
            ".gitkeep",
        ],
        "processed": [
            ".gitkeep",
        ],
        "samples": [],
    },
    "notebooks": {
        "exploration": [
            "eda.ipynb",
            "test.ipynb",
        ],
        "feature_analysis": [
            "basic_feature.ipynb",
            "data_cleaning.ipynb",
            "feature_engineering.ipynb",
        ],
        "modeling": [
            "logistic_experiments.ipynb",
            "xgboost_experiments.ipynb",
        ],
        "validation": [
            "model_validation.ipynb",
        ],
    },
    "api": {
        "__init__.py": None,
        "app.py": None,
        "dependencies.py": None,
        "schemas.py": None,
        "routes": [
            "__init__.py",
            "predict.py",
        ],
    },
    "src": {
        "credit_risk": {
            "__init__.py": None,
            "data": [
                "__init__.py",
                "load_data.py",
                "clean_data.py",
                "split_data.py",
            ],
            "features": [
                "__init__.py",
                "build_features.py",
            ],
            "models": [
                "__init__.py",
                "base.py",
                "logistic_model.py",
                "xgboost_model.py",
                "train.py",
                "predict.py",
            ],
            "evaluation": [
                "metrics.py",
                "calibration.py",
                "model_comparison.py",
            ],
            "utils": [
                "__init__.py",
                "config.py",
                "logging.py",
                "paths.py",
            ],
        }
    },
    "scripts": [
        "run_split.py",
        "train_logistic.py",
        "train_xgboost.py",
        "compare_models.py",
    ],
    "models": {
        "logistic": [
            "model.pkl",
            "metrics.json",
            "feature_builder.pkl",
            "model_card.md",
        ],
        "xgboost": [
            "model.pkl",
            "metrics.json",
            "feature_importance.csv",
            "feature_builder.pkl",
            "model_card.md",
        ],
    },
    "tests": [
        "__init__.py",
        "test_api.py",
        "test_split.py",
        "test_features.py",
        "test_logistic_model.py",
        "test_xgboost_model.py",
    ],
    ".github": {
        "workflows": [
            "ci.yml",
        ]
    },
}


ROOT_FILES = [
    ".dockerignore",
    "README.md",
    "LICENSE",
    ".gitignore",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "Dockerfile",
    "LCDataDictionary.xlsx",
    "template.py",
]


def create_structure(base: Path, structure):
    if isinstance(structure, list):
        for item in structure:
            path = base / item
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)

    elif isinstance(structure, dict):
        for name, content in structure.items():
            path = base / name
            if content is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)
                create_structure(path, content)


def main():
    root = Path.cwd()

    for file in ROOT_FILES:
        (root / file).touch(exist_ok=True)

    create_structure(root, PROJECT_STRUCTURE)

    print("OK: Credit Risk Modeling repository scaffold created successfully.")


if __name__ == "__main__":
    main()
