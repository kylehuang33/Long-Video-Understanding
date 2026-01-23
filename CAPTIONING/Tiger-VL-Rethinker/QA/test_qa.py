#!/usr/bin/env python3
"""
Test script to verify QA evaluation setup.
"""

import os
import pandas as pd


def test_question_file(question_path: str):
    """Test loading and parsing question file."""
    print("Testing question file loading...")

    if not os.path.exists(question_path):
        print(f"✗ Question file not found: {question_path}")
        return False

    try:
        df = pd.read_parquet(question_path)
        print(f"✓ Successfully loaded {len(df)} questions")

        # Check columns
        required_cols = ['question', 'answer', 'category', 'qid', 'image']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"✗ Missing columns: {missing_cols}")
            return False

        print(f"✓ All required columns present: {list(df.columns)}")

        # Show sample
        print("\nSample question:")
        sample = df.iloc[0].to_dict()
        print(f"  QID: {sample['qid']}")
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Answer: {sample['answer']}")
        print(f"  Category: {sample['category']}")
        print(f"  Image: {sample['image']}")

        return True

    except Exception as e:
        print(f"✗ Error loading question file: {e}")
        return False


def test_dataset_path(dataset_path: str, sample_image: str = None):
    """Test dataset path and image access."""
    print("\nTesting dataset path...")

    if not os.path.exists(dataset_path):
        print(f"✗ Dataset path not found: {dataset_path}")
        return False

    print(f"✓ Dataset path exists: {dataset_path}")

    if sample_image:
        image_path = os.path.join(dataset_path, sample_image)
        if os.path.exists(image_path):
            print(f"✓ Sample image found: {image_path}")
            return True
        else:
            print(f"✗ Sample image not found: {image_path}")
            return False

    return True


def test_model_import():
    """Test if required packages can be imported."""
    print("\nTesting package imports...")

    try:
        import torch
        print(f"✓ torch imported (version: {torch.__version__})")
    except ImportError:
        print("✗ torch not found")
        return False

    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        print("✓ transformers imported")
    except ImportError:
        print("✗ transformers not found or Qwen3VL not available")
        return False

    try:
        from qwen_vl_utils import process_vision_info
        print("✓ qwen_vl_utils imported")
    except ImportError:
        print("✗ qwen_vl_utils not found")
        return False

    try:
        import pandas as pd
        print(f"✓ pandas imported (version: {pd.__version__})")
    except ImportError:
        print("✗ pandas not found")
        return False

    return True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")

    try:
        import torch
        if torch.cuda.is_available():
            print("✓ CUDA is available")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  Current GPU: {torch.cuda.current_device()}")
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA is not available (CPU mode)")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def main():
    print("="*60)
    print("Qwen3VL QA Evaluation Setup Test")
    print("="*60)

    # Test 1: Package imports
    test1 = test_model_import()

    # Test 2: CUDA
    test2 = test_cuda()

    # Test 3: Question file (update path as needed)
    question_path = "/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/39Krelease.parquet"
    test3 = test_question_file(question_path) if os.path.exists(question_path) else None

    # Test 4: Dataset path (update as needed)
    dataset_path = "/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K"
    test4 = test_dataset_path(dataset_path) if os.path.exists(dataset_path) else None

    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    print(f"Package imports: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"CUDA available: {'✓ PASS' if test2 else '⚠ CPU mode'}")
    if test3 is not None:
        print(f"Question file: {'✓ PASS' if test3 else '✗ FAIL'}")
    if test4 is not None:
        print(f"Dataset path: {'✓ PASS' if test4 else '✗ FAIL'}")
    print("="*60)

    if test1:
        print("\n✓ Setup is ready for QA evaluation!")
    else:
        print("\n✗ Setup incomplete. Please install missing packages.")


if __name__ == "__main__":
    main()
