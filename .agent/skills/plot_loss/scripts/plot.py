#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Plot training loss from a console log file.')
    parser.add_argument('--log-file', type=str, required=True, help='Path to the console log file')
    parser.add_argument('--out-file', type=str, default='training_loss.png', help='Path to save the output PNG file')
    args = parser.parse_args()

    steps = []
    losses = []

    # Example log line: Step 02989/6075 (49.20%) | Loss: 6.6599 | LR: 1.2e-3 | Speed: 48090 tok/s | Seen: 179.1M / 7.3M | ETA: 6.4m
    pattern = re.compile(r"Step\s+(\d+)/\d+.*?Loss:\s+([\d\.]+)")

    try:
        with open(args.log_file, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    steps.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
    except FileNotFoundError:
        print(f"Error: Log file not found at {args.log_file}")
        return

    if not steps:
        print("No loss data found in the log file.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss', alpha=0.3, color='blue')

    # Calculate moving average
    if len(losses) > 0:
        window = max(1, len(losses) // 50)
        moving_avg = [sum(losses[max(0, i-window+1):i+1]) / min(i+1, window) for i in range(len(losses))]
        plt.plot(steps, moving_avg, label=f'Moving Average (window={window})', color='red')

    plt.xlabel('Global Step')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training Loss over time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(args.out_file, dpi=300)
    print(f"Plot saved to {args.out_file}")

if __name__ == "__main__":
    main()
