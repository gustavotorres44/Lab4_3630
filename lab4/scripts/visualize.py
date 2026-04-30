"""
Visualization Script for Lab4: Behavior Cloning

This script reads TensorBoard event files and creates matplotlib plots for:
1. Training Loss over training progress
2. Evaluation Average Return over training progress
3. Comparison with Expert performance

Usage:
    python lab4/scripts/visualize.py                        # Visualize most recent run
    python lab4/scripts/visualize.py --logdir data/run_name # Specific run
    python lab4/scripts/visualize.py --all                  # All runs in data/
    python lab4/scripts/visualize.py --save                 # Save plots as PNG
    python lab4/scripts/visualize.py --logdir data/run_name --export-video
"""

import os
import sys
import argparse
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
sys.path.insert(0, PROJECT_ROOT)

# Try to import tensorboard for reading event files
try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not installed. Using backup CSV logging.")


def get_metric(metrics, *names):
    """Return the first matching metric entry from a list of possible names."""
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def find_latest_policy_file(logdir):
    """Find the most recent checkpoint inside a single run directory."""
    policy_files = glob.glob(os.path.join(logdir, 'policy_itr_*.pt'))
    if not policy_files:
        return None
    return max(policy_files, key=os.path.getmtime)


def infer_env_name(logdir):
    """Infer the environment name from the run directory name."""
    match = re.search(r'([A-Za-z0-9]+-v\d+)', os.path.basename(logdir))
    if match:
        return match.group(1)
    return None


def export_policy_video(
        logdir,
        policy_file=None,
        env_name=None,
        n_layers=2,
        size=64,
        episodes=1,
        seed=0,
        output_dir=None,
):
    """Export rollout videos for a saved checkpoint as mp4 files."""
    os.environ.setdefault('MUJOCO_GL', 'egl')

    import torch
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo
    from lab4.policies.MLP_policy import MLPPolicySL

    policy_file = policy_file or find_latest_policy_file(logdir)
    if policy_file is None:
        raise FileNotFoundError(f'No policy checkpoint found in {logdir}')

    env_name = env_name or infer_env_name(logdir)
    if env_name is None:
        raise ValueError(
            'Could not infer env name from logdir. Please pass --env-name explicitly.'
        )

    output_dir = output_dir or os.path.join(logdir, 'exported_videos')
    os.makedirs(output_dir, exist_ok=True)

    env_kwargs = {'render_mode': 'rgb_array'}
    if 'Ant' in env_name:
        env_kwargs['use_contact_forces'] = True

    env = gym.make(env_name, **env_kwargs)
    env = RecordVideo(
        env,
        video_folder=output_dir,
        episode_trigger=lambda episode_id: True,
        name_prefix=os.path.splitext(os.path.basename(policy_file))[0],
    )

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    policy = MLPPolicySL(
        ac_dim=ac_dim,
        ob_dim=ob_dim,
        n_layers=n_layers,
        size=size,
        discrete=False,
        learning_rate=1e-3,
    )
    policy.load_state_dict(torch.load(policy_file, map_location='cpu'))
    policy.eval()

    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0

        while not done:
            action = policy.get_action(obs)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        returns.append(episode_return)

    env.close()

    print(f"Saved rollout video(s) to: {output_dir}")
    print(f"Policy checkpoint used: {policy_file}")
    print(f"Video episode return(s): {[float(round(ret, 2)) for ret in returns]}")


def get_x_data(metrics, metric_entry):
    """
    Prefer environment steps on the x-axis when available.
    Fall back to the metric's logged scalar step otherwise.
    """
    envsteps_metric = get_metric(metrics, 'Train_EnvstepsSoFar')
    if envsteps_metric is not None and len(envsteps_metric['values']) == len(metric_entry['values']):
        return envsteps_metric['values'], 'Environment Steps'
    return metric_entry['steps'], 'Step'


def plot_metric(ax, x, y, fmt, label, linewidth=2, markersize=6, alpha=1.0, color=None):
    """Plot a metric and ensure single-point runs are still visible."""
    plot_kwargs = {
        'label': label,
        'linewidth': linewidth,
        'markersize': markersize,
        'alpha': alpha,
    }
    if color is not None:
        plot_kwargs['color'] = color
    if len(x) <= 1:
        plot_kwargs['linestyle'] = 'None'
        plot_kwargs['marker'] = 'o'
        if color is None and fmt:
            plot_kwargs['color'] = fmt[0]
        return ax.plot(x, y, **plot_kwargs)
    if color is not None:
        plot_kwargs['marker'] = 'o'
        return ax.plot(x, y, **plot_kwargs)
    return ax.plot(x, y, fmt, marker='o', **plot_kwargs)


def read_tensorboard_logs(logdir):
    """Read metrics from TensorBoard event files"""
    if not HAS_TENSORBOARD:
        return None
    
    # Find event files
    event_files = glob.glob(os.path.join(logdir, 'events.out.tfevents.*'))
    if not event_files:
        return None
    
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    
    metrics = {}
    tags = ea.Tags().get('scalars', [])
    
    for tag in tags:
        events = ea.Scalars(tag)
        metrics[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events]
        }
    
    return metrics


def read_csv_logs(logdir):
    """Fallback: read from CSV files if they exist"""
    csv_file = os.path.join(logdir, 'progress.csv')
    if not os.path.exists(csv_file):
        return None
    
    import csv
    metrics = defaultdict(lambda: {'steps': [], 'values': []})
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            for key, value in row.items():
                try:
                    metrics[key]['steps'].append(i)
                    metrics[key]['values'].append(float(value))
                except (ValueError, TypeError):
                    pass
    
    return dict(metrics) if metrics else None


def find_log_directories(base_dir='data'):
    """Find all log directories in the data folder"""
    if not os.path.exists(base_dir):
        return []
    
    log_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains event files or CSV
            has_events = bool(glob.glob(os.path.join(item_path, 'events.out.tfevents.*')))
            has_csv = os.path.exists(os.path.join(item_path, 'progress.csv'))
            if has_events or has_csv:
                log_dirs.append(item_path)
    
    # Sort by modification time (most recent first)
    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return log_dirs


def plot_training_curves(metrics, title="Training Curves", save_path=None):
    """Plot training loss and evaluation returns"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    ax1 = axes[0]
    loss_metric = get_metric(metrics, 'Training Loss', 'Training_Loss')
    if loss_metric is not None:
        steps, x_label = get_x_data(metrics, loss_metric)
        values = loss_metric['values']
        plot_metric(ax1, steps, values, 'b-', label='Training Loss')
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No Training Loss data found', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Training Loss', fontsize=14)
    
    # Plot 2: Evaluation Return
    ax2 = axes[1]
    has_data = False
    x_label = 'Step'
    
    if 'Eval_AverageReturn' in metrics:
        eval_metric = metrics['Eval_AverageReturn']
        steps, x_label = get_x_data(metrics, eval_metric)
        values = eval_metric['values']
        plot_metric(ax2, steps, values, 'g-', label='Eval Return')
        has_data = True
    
    if 'Train_AverageReturn' in metrics:
        train_metric = metrics['Train_AverageReturn']
        steps, x_label = get_x_data(metrics, train_metric)
        values = train_metric['values']
        plot_metric(ax2, steps, values, 'b--', label='Train Return', alpha=0.7)
        has_data = True
    
    if 'Initial_DataCollection_AverageReturn' in metrics:
        initial = metrics['Initial_DataCollection_AverageReturn']['values'][0]
        ax2.axhline(y=initial, color='r', linestyle=':', linewidth=2, 
                   label=f'Expert Return ({initial:.0f})')
        
        # Add threshold line (2/3 of expert)
        threshold = initial * (2/3)
        ax2.axhline(y=threshold, color='orange', linestyle='--', linewidth=2,
                   label=f'Threshold (2/3 = {threshold:.0f})')
    
    if has_data:
        ax2.set_xlabel(x_label, fontsize=12)
        ax2.set_ylabel('Average Return', fontsize=12)
        ax2.set_title('Evaluation Returns', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right')
    else:
        ax2.text(0.5, 0.5, 'No Return data found',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Evaluation Returns', fontsize=14)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def plot_comparison(all_metrics, save_path=None):
    """Plot comparison of multiple runs"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))
    
    # Plot 1: Training Loss comparison
    ax1 = axes[0]
    loss_x_label = 'Step'
    for (name, metrics), color in zip(all_metrics.items(), colors):
        loss_metric = get_metric(metrics, 'Training Loss', 'Training_Loss')
        if loss_metric is not None:
            steps, loss_x_label = get_x_data(metrics, loss_metric)
            values = loss_metric['values']
            label = os.path.basename(name)[:30]
            plot_metric(ax1, steps, values, '', label=label, color=color, markersize=4)
    
    ax1.set_xlabel(loss_x_label, fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # Plot 2: Eval Return comparison
    ax2 = axes[1]
    return_x_label = 'Step'
    for (name, metrics), color in zip(all_metrics.items(), colors):
        if 'Eval_AverageReturn' in metrics:
            eval_metric = metrics['Eval_AverageReturn']
            steps, return_x_label = get_x_data(metrics, eval_metric)
            values = eval_metric['values']
            label = os.path.basename(name)[:30]
            plot_metric(ax2, steps, values, '', label=label, color=color, markersize=4)
    
    # Add expert reference line if available
    for name, metrics in all_metrics.items():
        if 'Initial_DataCollection_AverageReturn' in metrics:
            initial = metrics['Initial_DataCollection_AverageReturn']['values'][0]
            ax2.axhline(y=initial, color='red', linestyle=':', linewidth=2,
                       label=f'Expert ({initial:.0f})')
            break
    
    ax2.set_xlabel(return_x_label, fontsize=12)
    ax2.set_ylabel('Average Return', fontsize=12)
    ax2.set_title('Evaluation Return Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='lower right')
    
    plt.suptitle('Multi-Run Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    
    return fig


def print_summary(metrics, name=""):
    """Print a summary of the training run"""
    print("\n" + "=" * 50)
    print(f"Summary: {name}")
    print("=" * 50)
    
    loss_metric = get_metric(metrics, 'Training Loss', 'Training_Loss')
    if loss_metric is not None:
        values = loss_metric['values']
        print(f"Training Loss: {values[0]:.4f} -> {values[-1]:.4f}")
    
    if 'Eval_AverageReturn' in metrics:
        values = metrics['Eval_AverageReturn']['values']
        print(f"Eval Return: {values[0]:.2f} -> {values[-1]:.2f}")
    
    if 'Initial_DataCollection_AverageReturn' in metrics:
        expert = metrics['Initial_DataCollection_AverageReturn']['values'][0]
        print(f"Expert Return: {expert:.2f}")
        
        if 'Eval_AverageReturn' in metrics:
            final_eval = metrics['Eval_AverageReturn']['values'][-1]
            pct = (final_eval / expert) * 100
            threshold_pct = 66.67
            status = "PASS" if pct >= threshold_pct else "FAIL"
            print(f"Final Performance: {pct:.1f}% of expert [{status}]")
    
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Visualize Lab4 Training Results')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Path to specific log directory')
    parser.add_argument('--all', action='store_true',
                        help='Plot all runs in data/')
    parser.add_argument('--save', action='store_true',
                        help='Save plots as PNG files')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (useful with --save)')
    parser.add_argument('--export-video', action='store_true',
                        help='Export rollout video(s) from the selected run')
    parser.add_argument('--policy-file', type=str, default=None,
                        help='Optional path to a specific policy checkpoint to render')
    parser.add_argument('--env-name', type=str, default=None,
                        help='Optional environment name override for video export')
    parser.add_argument('--video-episodes', type=int, default=1,
                        help='Number of episodes to render when exporting video')
    parser.add_argument('--video-output-dir', type=str, default=None,
                        help='Directory where exported mp4 files should be saved')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='Policy hidden-layer count used when loading checkpoints for video export')
    parser.add_argument('--size', type=int, default=64,
                        help='Policy hidden-layer width used when loading checkpoints for video export')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed used when exporting rollout video(s)')
    args = parser.parse_args()
    
    # Determine which directories to process
    if args.logdir:
        log_dirs = [args.logdir]
    elif args.all:
        log_dirs = find_log_directories()
    else:
        # Default: most recent run
        log_dirs = find_log_directories()
        if log_dirs:
            log_dirs = [log_dirs[0]]
    
    if not log_dirs:
        print("No log directories found in data/")
        print("Run training first with: python lab4/scripts/run_lab4.py ...")
        return
    
    print(f"Found {len(log_dirs)} log directories")
    
    # Read metrics from each directory
    all_metrics = {}
    for logdir in log_dirs:
        print(f"\nReading: {logdir}")
        metrics = read_tensorboard_logs(logdir)
        if metrics is None:
            metrics = read_csv_logs(logdir)
        
        if metrics:
            all_metrics[logdir] = metrics
            print_summary(metrics, os.path.basename(logdir))
        else:
            print(f"  Warning: No metrics found in {logdir}")
    
    if not all_metrics:
        print("No metrics data found")
        return
    
    # Create plots
    if len(all_metrics) == 1:
        # Single run: detailed plot
        logdir, metrics = list(all_metrics.items())[0]
        name = os.path.basename(logdir)
        save_path = os.path.join(logdir, 'training_curves.png') if args.save else None
        plot_training_curves(metrics, title=f"Training Results: {name}", save_path=save_path)
    else:
        # Multiple runs: comparison plot
        save_path = 'data/comparison_plot.png' if args.save else None
        plot_comparison(all_metrics, save_path=save_path)
        
        # Also save individual plots if requested
        if args.save:
            for logdir, metrics in all_metrics.items():
                name = os.path.basename(logdir)
                save_path = os.path.join(logdir, 'training_curves.png')
                plot_training_curves(metrics, title=f"Training Results: {name}", save_path=save_path)
                plt.close()

    if args.export_video:
        if len(log_dirs) != 1:
            print("Video export currently expects exactly one run. Pass --logdir for a single run.")
        else:
            export_policy_video(
                logdir=log_dirs[0],
                policy_file=args.policy_file,
                env_name=args.env_name,
                n_layers=args.n_layers,
                size=args.size,
                episodes=args.video_episodes,
                seed=args.seed,
                output_dir=args.video_output_dir,
            )

    if not args.no_show:
        plt.show()
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
