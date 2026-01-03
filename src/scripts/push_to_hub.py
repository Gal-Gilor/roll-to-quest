"""Upload anchor-positive pairs dataset to HuggingFace Hub.

This script reads a JSONL file containing anchor-positive pairs and uploads
it to the HuggingFace Hub as a Dataset with a single 'train' split.

The dataset README is automatically set using the DATASET_CARD.md file from
the project root directory.

Dependencies:
    - HuggingFace token (set HF_TOKEN in .env or environment)
    - datasets library for Dataset creation
    - huggingface_hub for README upload

Example Usage:
    # Upload with auto-derived repo name (pushes to your namespace)
    poetry run python -m src.scripts.push_to_hub SRD_CC_v5.2.1_pairs.jsonl

    # Upload to specific repository
    poetry run python -m src.scripts.push_to_hub SRD_CC_v5.2.1_pairs.jsonl \
        --repo-id gal-gilor/dnd-srd-anchor-positive-pairs

    # Upload as private dataset
    poetry run python -m src.scripts.push_to_hub SRD_CC_v5.2.1_pairs.jsonl \
        --repo-id my-private-dataset --private

Output:
    - Dataset uploaded to HuggingFace Hub
    - Returns URL of the uploaded dataset
"""

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

from datasets import Dataset
from huggingface_hub import HfApi

from src.settings import config
from src.settings import logger


def derive_repo_name(filename: str) -> str:
    """Derive a HuggingFace repo name from a filename.

    Converts filename to a URL-friendly repo name:
    - Removes .jsonl extension
    - Replaces underscores and dots with hyphens
    - Converts to lowercase

    Args:
        filename: The JSONL filename (e.g., 'SRD_CC_v5.2.1_pairs.jsonl')

    Returns:
        str: URL-friendly repo name (e.g., 'srd-cc-v5-2-1-pairs')
    """
    name = filename.replace(".jsonl", "")
    name = name.replace("_", "-").replace(".", "-")
    return name.lower()


def validate_jsonl_file(file_path: Path, sample_lines: int = 5) -> bool:
    """Validate that the JSONL file exists and has the expected format.

    Checks that the file:
    1. Exists and is readable
    2. Contains valid JSON lines
    3. Has 'anchor' and 'positive' keys in each line

    Args:
        file_path: Path to the JSONL file to validate.
        sample_lines: Number of lines to validate. Defaults to 5.

    Returns:
        bool: True if file is valid, False otherwise.
    """
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return False

    if not file_path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break

                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                if "anchor" not in data:
                    logger.error(f"Line {i + 1}: Missing 'anchor' key")
                    return False

                if "positive" not in data:
                    logger.error(f"Line {i + 1}: Missing 'positive' key")
                    return False

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON at line {i + 1}: {e}")
        return False
    except (IOError, OSError, PermissionError) as e:
        logger.error(f"Error reading file: {e}")
        return False

    logger.info(f"Validated JSONL file format: {file_path}")
    return True


def load_dataset_from_jsonl(file_path: Path) -> Dataset:
    """Load a HuggingFace Dataset from a JSONL file.

    Uses Dataset.from_json() which efficiently loads JSONL format.
    The resulting dataset will have 'anchor' and 'positive' columns.

    Args:
        file_path: Path to the JSONL file containing anchor-positive pairs.

    Returns:
        Dataset: HuggingFace Dataset with 'anchor' and 'positive' columns.
    """
    logger.info(f"Loading dataset from: {file_path}")

    dataset = Dataset.from_json(str(file_path))

    logger.info(f"Loaded {len(dataset)} pairs from JSONL file")
    logger.info(f"Dataset features: {dataset.features}")

    return dataset


def push_to_hub(
    dataset: Dataset,
    repo_id: str,
    readme_path: Path,
    private: bool = False,
    token: str | None = None,
) -> str:
    """Push dataset and README to HuggingFace Hub.

    Uploads the dataset as a single 'train' split. The dataset is created
    automatically if it doesn't exist. Then updates the README.md with the
    content from DATASET_CARD.md.

    Args:
        dataset: HuggingFace Dataset to upload.
        repo_id: Dataset name ('my-dataset') or full path ('user/my-dataset').
        readme_path: Path to the DATASET_CARD.md file for README.
        private: Whether to create a private dataset. Defaults to False.
        token: HuggingFace API token. If None, uses HF_TOKEN from environment.

    Returns:
        str: URL of the uploaded dataset.
    """
    api = HfApi(token=token)

    # If repo_id doesn't have a namespace, prepend the current user's namespace
    if "/" not in repo_id:
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{repo_id}"
        logger.info(f"Resolved full repo path: {repo_id}")

    commit_message = f"Upload dataset with {len(dataset)} anchor-positive pairs"

    logger.info(f"Pushing dataset to HuggingFace Hub: {repo_id}")
    logger.info(f"Private: {private}, Pairs: {len(dataset)}")

    # Push dataset to hub - creates repo if it doesn't exist
    dataset.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
        commit_message=commit_message,
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(f"Dataset pushed successfully: {url}")

    # Update README with dataset card
    if readme_path.exists():
        logger.info(f"Updating README.md from: {readme_path}")

        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        # Replace local image path with GitHub raw URL
        readme_content = readme_content.replace(
            "../../assets/generation_pipeline_diagram.png",
            "https://raw.githubusercontent.com/Gal-Gilor/roll-to-quest/main/assets/generation_pipeline_diagram.png",
        )

        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update README with dataset documentation",
        )
        logger.info("README.md updated successfully")
    else:
        logger.warning(f"Dataset card not found: {readme_path}")

    return url


def main() -> int:
    """Upload anchor-positive pairs dataset to HuggingFace Hub.

    This function orchestrates the entire upload process:
    1. Parses CLI arguments
    2. Validates the HuggingFace token is configured
    3. Locates the JSONL file in the data/pairs/ directory
    4. Validates the JSONL file format
    5. Loads the dataset using Dataset.from_json()
    6. Reads the DATASET_CARD.md for the dataset README
    7. Pushes the dataset to HuggingFace Hub
    8. Updates the dataset README with DATASET_CARD.md content

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    args = parse_arguments()

    # Validate filename to prevent path traversal attacks
    if "/" in args.filename or "\\" in args.filename or ".." in args.filename:
        logger.error(
            f"Invalid filename: {args.filename}. "
            "Filename must not contain path separators or '..'."
        )
        return 1

    # Get HuggingFace token from config or environment
    token = config.HF_TOKEN
    if not token:
        logger.error(
            "HuggingFace token not configured. Set HF_TOKEN in .env file or environment."
        )
        return 1

    # Construct paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "pairs"
    file_path = data_dir / args.filename
    readme_path = project_root / "src" / "pair_generation" / "DATASET_CARD.md"

    # Use provided repo_id or derive from filename
    repo_id = args.repo_id
    if repo_id is None:
        repo_id = derive_repo_name(args.filename)
        logger.info(f"No --repo-id provided, using derived name: {repo_id}")

    logger.info("Starting HuggingFace Hub upload process")
    logger.info(f"Input file: {file_path}")
    logger.info(f"Target repository: {repo_id}")

    # Validate input file
    if not validate_jsonl_file(file_path):
        return 1

    try:
        # Load dataset from JSONL
        dataset = load_dataset_from_jsonl(file_path)

        # Push to HuggingFace Hub with README
        url = push_to_hub(
            dataset=dataset,
            repo_id=repo_id,
            readme_path=readme_path,
            private=args.private,
            token=token,
        )

        logger.info(f"Upload complete! Dataset available at: {url}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return 1


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - filename: Name of JSONL file to upload
            - repo_id: HuggingFace repository ID
            - private: Whether to create private repository
    """
    parser = argparse.ArgumentParser(
        description="Upload anchor-positive pairs dataset to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
            Examples:
              # Upload with auto-derived repo name (pushes to your namespace)
              poetry run python -m src.scripts.push_to_hub SRD_CC_v5.2.1_pairs.jsonl

              # Upload to specific repository
              poetry run python -m src.scripts.push_to_hub SRD_CC_v5.2.1_pairs.jsonl \\
                  --repo-id gal-gilor/dnd-srd-pairs

              # Upload as private dataset
              poetry run python -m src.scripts.push_to_hub SRD_CC_v5.2.1_pairs.jsonl \\
                  --repo-id my-private-dataset --private
        """),
    )

    parser.add_argument(
        "filename",
        type=str,
        help="Name of the JSONL file in data/pairs/ directory (e.g., pairs.jsonl)",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "Dataset name on HuggingFace Hub. Can be 'dataset-name' (uses your "
            "namespace) or 'org/dataset-name' (for organizations). If omitted, "
            "derives from filename. The dataset is created automatically."
        ),
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private dataset repository. Defaults to public.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
