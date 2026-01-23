#!/usr/bin/env python3
"""
Test script to verify Qwen3VL captioning setup.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def test_model_loading():
    """Test if the model loads correctly."""
    print("Testing model loading...")
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        print("✓ Model loaded successfully!")

        # Check device
        device = next(model.parameters()).device
        print(f"✓ Model is on device: {device}")

        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  Current GPU: {torch.cuda.current_device()}")
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA is not available (CPU mode)")

        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def test_caption_generation(image_path: str = None):
    """Test caption generation on a sample image."""
    if image_path is None:
        print("⚠ No test image provided, skipping caption generation test")
        return True

    print(f"\nTesting caption generation on: {image_path}")
    try:
        # Load model
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }
        ]

        # Process
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)

        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)

        print("✓ Caption generated successfully!")
        print(f"  Caption: {output_text[0]}")
        return True
    except Exception as e:
        print(f"✗ Error generating caption: {e}")
        return False

def main():
    print("="*60)
    print("Qwen3VL Captioning Setup Test")
    print("="*60)

    # Test 1: Model loading
    test1 = test_model_loading()

    # Test 2: Caption generation (optional, requires image)
    # Uncomment and provide a test image path if you want to test this
    # test2 = test_caption_generation("/path/to/test/image.jpg")

    print("\n" + "="*60)
    if test1:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*60)

if __name__ == "__main__":
    main()
