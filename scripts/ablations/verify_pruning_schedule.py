#!/usr/bin/env python3
"""
Pruning Schedule Verification Tool

Usage:
    python verify_pruning_schedule.py

This tool helps you verify that your custom pruning schedules
maintain the target average token count (default: 128).
"""

def verify_schedule(name, schedule, total_layers=32, stage1_tokens=256, target_avg=128):
    """
    Verify a pruning schedule maintains the target average.

    Args:
        name: Display name for the schedule
        schedule: List of (layer_idx, token_count) tuples
        total_layers: Total number of layers (32 for 7B, 40 for 13B)
        stage1_tokens: Initial tokens from Stage 1 (256 for T=128)
        target_avg: Target average tokens across all layers

    Returns:
        Average token count
    """
    current_tokens = stage1_tokens
    total = 0

    prev_layer = 0
    details = []
    for layer_idx, token_count in schedule:
        layers_span = layer_idx - prev_layer
        total += layers_span * current_tokens
        details.append(f"    Layers {prev_layer:2d}-{layer_idx-1:2d}: {current_tokens:3d} tokens ({layers_span:2d} layers)")
        current_tokens = token_count
        prev_layer = layer_idx

    # Remaining layers
    layers_span = total_layers - prev_layer
    total += layers_span * current_tokens
    details.append(f"    Layers {prev_layer:2d}-{total_layers-1:2d}: {current_tokens:3d} tokens ({layers_span:2d} layers)")

    average = total / total_layers
    status = '✓' if abs(average - target_avg) < 0.01 else '✗'

    print(f"\n{status} {name}")
    print(f"  Schedule: {schedule}")
    for detail in details:
        print(detail)
    print(f"  Average: {average:.2f} tokens (target: {target_avg})")

    return average


def main():
    print("=" * 90)
    print("Pruning Schedule Verification Tool")
    print("=" * 90)

    # ===== 在这里定义你的配置 =====
    schedules = [
        # 示例配置 (请根据需要修改)
        ("Single-stage Example", [(16, 32)]),
        ("Two-stage [STAR, Ours]", [(12, 64), (24, 32)]),
        ("Three-stage Example", [(8, 128), (18, 72), (26, 32)]),
    ]

    # 验证所有配置
    for name, schedule in schedules:
        verify_schedule(name, schedule)

    print("\n" + "=" * 90)
    print("Verification Complete!")
    print("=" * 90)
    print("\nTo use these schedules in the ablation script:")
    print("1. Open scripts/ablations/run_stage2_pruning_schedule.sh")
    print("2. Add your schedules to SCHEDULE_CONFIGS array")
    print("3. Format: \"Name|custom|[[layer1, tokens1], [layer2, tokens2], ...]\"")
    print("\nExample:")
    print("  SCHEDULE_CONFIGS=(")
    for name, schedule in schedules:
        print(f"      \"{name}|custom|{schedule}\"")
    print("  )")


if __name__ == "__main__":
    main()
