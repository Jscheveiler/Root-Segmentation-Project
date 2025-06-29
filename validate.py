import torch
from torch.utils.data import DataLoader
from dataset import RootSegmentationDataset
from models.unet import UNet
from utils.metrics import dice_coefficient, iou_score, pixel_accuracy, recall, precision, f1_score, specificity, balanced_accuracy
from utils.helpers import load_checkpoint
import argparse

def validate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and loader
    dataset = RootSegmentationDataset(root_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model
    model = UNet(in_channels=3, out_channels=1)
    model.to(device)

    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}...")
        load_checkpoint(args.ckpt, model)

    model.eval()

    # Initialize accumulator list
    # Dice, IoU, Accuracy, Recall, Precision, F1, Specificity, Balanced Accuracy
    accs = [0 for i in range(8)]

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # Updating accumulators
            accs[0] += dice_coefficient(outputs, masks)
            accs[1] += iou_score(outputs, masks)
            accs[2] += pixel_accuracy(outputs, masks)
            accs[3] += recall(outputs, masks)
            accs[4] += precision(outputs, masks)
            accs[5] += f1_score(outputs, masks)
            accs[6] += specificity(outputs, masks)
            accs[7] += balanced_accuracy(outputs, masks)

    num_samples = len(dataloader)
    print(f"\nValidation Results:")
    print(f"---------------------------------")
    print(f"| Dice Coefficient     | {accs[0] / num_samples:.4f} |")
    print(f"| IoU Score            | {accs[1] / num_samples:.4f} |")
    print(f"| Pixel Accuracy       | {accs[2] / num_samples:.4f} |")
    print(f"| Recall (Sensitivity) | {accs[3] / num_samples:.4f} |")
    print(f"| Precision            | {accs[4] / num_samples:.4f} |")
    print(f"| F1 Score             | {accs[5] / num_samples:.4f} |")
    print(f"| Specificity          | {accs[6] / num_samples:.4f} |")
    print(f"| Balanced Accuracy    | {accs[7] / num_samples:.4f} |")
    print(f"---------------------------------")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/1_mesocosm", help="Path to dataset")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    validate(args)
