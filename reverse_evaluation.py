import argparse
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms

from models.reverse_prediction import Basic_Triplet

from utils.dataset import create_dataloaders
import config as cfg


def load_model_from_checkpoint(checkpoint_path, device):
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    
    model = Basic_Triplet().to(device)
    
    if 'ema_state_dict' in checkpoint:
        print("Loading EMA weights")
        model.load_state_dict(checkpoint['ema_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
 
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Successfully loaded model from epoch {epoch}")
    
    return model


def evaluate_test_set(
    model,
    device,
    batch_size=8,
    num_workers=2
):

    print("\n" + "="*60)
    print("EVALUATING MODEL ON TEST DATASET")
    print("="*60)

    _, _, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        distributed=False,
    )
    
    if test_loader is None:
        raise ValueError("Test dataloader is None. Check if test CSV exists.")
    
    print(f"Test set size: {len(test_loader.dataset)} samples")

    l1_loss = nn.L1Loss()
    total_l1 = 0.0
    total_count = 0
    
    with torch.no_grad():
        start_time = time.time()
        
        for batch_idx, (input_feat, input_image, target_image, target) in enumerate(test_loader):
            input_image = input_image.to(device)
            target_image = target_image.to(device)
            input_feat = input_feat.to(device)
            target = target.to(device)
            batch_size_local = target_image.size(0)


            l1 = l1_loss(model, target_image).item()

            
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches...")
        
        elapsed_time = time.time() - start_time

    avg_l1 = total_l1 / total_count
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Total samples evaluated: {total_count}")
    print(f"Evaluation time: {elapsed_time:.2f}s")
    print(f"Time per sample: {elapsed_time/total_count:.2f}s")
    print(f"\nMetrics:")
    print(f"  L1 Loss:       {avg_l1:.4f}")
    print("="*60)
    
    
    # Save results to file
    results = {
        'total_samples': total_count,
        'l1_loss': avg_l1,
        'evaluation_time': elapsed_time,
        'time_per_sample': elapsed_time / total_count,
    }
    
    results_file = os.path.join(cfg.OUTPUT_DIR, "test_results.txt")
    with open(results_file, 'w') as f:
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Evaluation")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    args = parser.parse_args()

    device_str = f"cuda:{cfg.DEVICE_IDS[0]}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    print(f"Using device: {device}")
    print(f"Batch size:   {cfg.BATCH_SIZE_PER_GPU} (from config)")
    
    pipeline, _, _ = load_model_from_checkpoint(
        args.checkpoint,
        device
    )

    evaluate_test_set(
        pipeline,
        device,
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS
    )

if __name__ == '__main__':
    main()