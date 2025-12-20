"""
Design2Code-18B-v0 Inference Script

This script runs inference on the Design2Code-18B model (CogAgent-based).
Requires: SwissArmyTransformer, the CogVLM code, and the model checkpoint.

Usage:
    python run_design2code_18b.py --model-path /path/to/design2code-18b-v0 \
                                   --input-dir /path/to/images \
                                   --output-dir /path/to/predictions
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Run Design2Code-18B inference")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to design2code-18b-v0 model checkpoint")
    parser.add_argument("--cogvlm-path", type=str, default="/root/Design2Code_official/CogVLM",
                        help="Path to CogVLM code directory")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing input images (PNG files)")
    parser.add_argument("--output-dir", type=str, default="results_Design2Code18B/predictions",
                        help="Directory to save HTML predictions")
    parser.add_argument("--samples", type=int, default=484,
                        help="Number of samples to process")
    parser.add_argument("--use-dataset", action="store_true",
                        help="Load images from HuggingFace dataset instead of local directory")
    args = parser.parse_args()

    # Add CogVLM to path
    sys.path.insert(1, args.cogvlm_path)

    try:
        from sat.model import AutoModel
        from sat.model.mixins import CachedAutoregressiveMixin
        from utils.models import FineTuneTestCogAgentModel
        from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
    except ImportError as e:
        print(f"Error importing SAT/CogVLM modules: {e}")
        print("\nPlease ensure you have:")
        print("1. Installed SwissArmyTransformer: pip install SwissArmyTransformer")
        print("2. Cloned the Design2Code repo with CogVLM code")
        print(f"3. Set --cogvlm-path to the CogVLM directory (current: {args.cogvlm_path})")
        sys.exit(1)

    import torch

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DESIGN2CODE-18B INFERENCE")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Samples: {args.samples}")

    # Load model
    print("\nLoading model...")
    model_args_namespace = argparse.Namespace(
        deepspeed=None,
        local_rank=0,
        rank=0,
        world_size=1,
        model_parallel_size=1,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True,
        device='cuda',
        bf16=True,
        fp16=None,
        stream_chat=False  # Required for chat function
    )

    # Load from local path - home_path should be parent directory
    import os
    model_dir = os.path.dirname(args.model_path)
    model_name = os.path.basename(args.model_path)
    model, model_args = FineTuneTestCogAgentModel.from_pretrained(
        model_name,
        args=model_args_namespace,
        home_path=model_dir,
        overwrite_args={'model_parallel_size': 1}
    )
    model = model.eval()
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    print("Model loaded successfully!")

    # Initialize processors
    print("Initializing processors...")
    language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else "chat"
    tokenizer = llama2_tokenizer("lmsys/vicuna-7b-v1.5", signal_type=language_processor_version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
    text_processor_infer = llama2_text_processor_inference(tokenizer, 2048, model.image_length)

    # Load images
    if args.use_dataset:
        print("\nLoading images from HuggingFace dataset...")
        from datasets import load_dataset
        from PIL import Image
        import tempfile

        dataset = load_dataset("SALT-NLP/Design2Code-hf", split="train")
        sample_indices = list(range(min(args.samples, len(dataset))))

        def get_image_path(idx):
            """Save dataset image to temp file and return path"""
            img = dataset[idx]['image']
            temp_path = f"/tmp/design2code_img_{idx}.png"
            img.save(temp_path)
            return temp_path, idx

        image_sources = [(get_image_path(idx), idx) for idx in sample_indices]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory {input_dir} does not exist")
            sys.exit(1)

        # Get PNG files
        png_files = sorted(input_dir.glob("*.png"))[:args.samples]
        image_sources = [(str(f), f.stem) for f in png_files]

    # Inference function
    def get_html(image_path):
        with torch.no_grad():
            history = None
            cache_image = None
            query = ''  # Empty query for Design2Code

            response, history, cache_image = chat(
                image_path,
                model,
                text_processor_infer,
                image_processor,
                query,
                history=history,
                cross_img_processor=cross_image_processor,
                image=cache_image,
                max_length=4096,
                top_p=1.0,
                temperature=0.5,
                top_k=1,
                invalid_slices=text_processor_infer.invalid_slices,
                repetition_penalty=1.1,
                args=model_args_namespace
            )
        return response

    # Run inference
    print(f"\nRunning inference on {len(image_sources)} images...")

    for item in tqdm(image_sources, desc="Inference"):
        if args.use_dataset:
            (image_path, idx), _ = item
            output_filename = f"{idx}.html"
        else:
            image_path, stem = item
            idx = stem
            output_filename = f"{stem}.html"

        output_path = output_dir / output_filename

        # Skip if already exists
        if output_path.exists():
            continue

        try:
            html_output = get_html(image_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_output)
        except Exception as e:
            print(f"\nError processing {idx}: {e}")
            continue

        # Clean up temp file if using dataset
        if args.use_dataset and os.path.exists(image_path):
            os.remove(image_path)

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE!")
    print("=" * 60)
    print(f"Predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
