#!/usr/bin/env python3
"""
🧹 CLEAN AND RESET TOOL
Completely removes all checkpoints, wandb data, and cached files
Interactive Python version with better control
"""

import os
import shutil
import glob
from pathlib import Path
import argparse


class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


def get_dir_size(path):
    """Calculate directory size"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(entry.path)
    except PermissionError:
        pass
    return total


def format_size(size):
    """Format size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def confirm(message, default=False):
    """Ask for user confirmation"""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        response = input(f"{Colors.YELLOW}{message}{suffix}{Colors.NC}").strip().lower()
        if response == "":
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print(f"{Colors.RED}Please answer 'y' or 'n'{Colors.NC}")


def remove_directory(path, name, force=False):
    """Remove a directory with confirmation"""
    if not os.path.exists(path):
        print(f"{Colors.YELLOW}⚠️  Not found: {path}{Colors.NC}")
        return False
    
    print(f"\n{Colors.BLUE}📁 Found: {path}{Colors.NC}")
    
    # Calculate size and file count
    size = get_dir_size(path)
    file_count = sum(1 for _ in Path(path).rglob('*') if _.is_file())
    
    print(f"   Size: {Colors.YELLOW}{format_size(size)}{Colors.NC}")
    print(f"   Files: {Colors.YELLOW}{file_count:,}{Colors.NC}")
    
    if force or confirm(f"   Delete {name}?"):
        try:
            shutil.rmtree(path)
            print(f"   {Colors.GREEN}✅ Deleted!{Colors.NC}")
            return True
        except Exception as e:
            print(f"   {Colors.RED}❌ Error: {e}{Colors.NC}")
            return False
    else:
        print(f"   {Colors.YELLOW}⏭️  Skipped{Colors.NC}")
        return False


def remove_pattern(pattern, name, force=False):
    """Remove files matching pattern"""
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print(f"{Colors.YELLOW}⚠️  No {name} found{Colors.NC}")
        return 0
    
    print(f"\n{Colors.BLUE}🔍 Found {len(files)} {name}{Colors.NC}")
    
    if force or confirm(f"   Delete all {name}?"):
        deleted = 0
        for file in files:
            try:
                if os.path.isfile(file):
                    os.remove(file)
                elif os.path.isdir(file):
                    shutil.rmtree(file)
                deleted += 1
            except Exception as e:
                print(f"   {Colors.RED}❌ Error deleting {file}: {e}{Colors.NC}")
        
        print(f"   {Colors.GREEN}✅ Deleted {deleted}/{len(files)} items{Colors.NC}")
        return deleted
    else:
        print(f"   {Colors.YELLOW}⏭️  Skipped{Colors.NC}")
        return 0


def clean_all(force=False, dry_run=False):
    """Clean everything"""
    
    if dry_run:
        print(f"\n{Colors.CYAN}🔍 DRY RUN MODE - No files will be deleted{Colors.NC}\n")
        force = False
    
    print("\n" + "="*80)
    print("🧹 CLEAN AND RESET - COMPLETE FRESH START")
    print("="*80)
    
    stats = {
        'directories_removed': 0,
        'files_removed': 0,
        'space_freed': 0
    }
    
    # Step 1: Checkpoints
    print("\n" + "="*80)
    print("🎯 STEP 1: CHECKPOINTS")
    print("="*80)
    
    if not dry_run and remove_directory("checkpoints", "all checkpoints", force):
        stats['directories_removed'] += 1
    elif dry_run:
        remove_directory("checkpoints", "all checkpoints", force)
    
    # Step 2: W&B Data
    print("\n" + "="*80)
    print("🎯 STEP 2: WANDB DATA")
    print("="*80)
    
    if not dry_run and remove_directory("wandb", "W&B local logs", force):
        stats['directories_removed'] += 1
    elif dry_run:
        remove_directory("wandb", "W&B local logs", force)
    
    if not dry_run and remove_directory(".wandb", "W&B cache", force):
        stats['directories_removed'] += 1
    elif dry_run:
        remove_directory(".wandb", "W&B cache", force)
    
    # Step 3: Hydra Outputs
    print("\n" + "="*80)
    print("🎯 STEP 3: HYDRA OUTPUTS")
    print("="*80)
    
    if not dry_run and remove_directory("outputs", "Hydra outputs", force):
        stats['directories_removed'] += 1
    elif dry_run:
        remove_directory("outputs", "Hydra outputs", force)
    
    if not dry_run and remove_directory("multirun", "Hydra multirun outputs", force):
        stats['directories_removed'] += 1
    elif dry_run:
        remove_directory("multirun", "Hydra multirun outputs", force)
    
    # Step 4: Python Cache
    print("\n" + "="*80)
    print("🎯 STEP 4: PYTHON CACHE")
    print("="*80)
    
    count = remove_pattern("**/__pycache__", "__pycache__ directories", force)
    stats['files_removed'] += count
    
    count = remove_pattern("**/*.pyc", ".pyc files", force)
    stats['files_removed'] += count
    
    # Step 5: Torch Cache
    print("\n" + "="*80)
    print("🎯 STEP 5: TORCH CACHE")
    print("="*80)
    
    if not dry_run and remove_directory(".torch_compile_cache", "Torch compile cache", force):
        stats['directories_removed'] += 1
    elif dry_run:
        remove_directory(".torch_compile_cache", "Torch compile cache", force)
    
    # Step 6: Log Files
    print("\n" + "="*80)
    print("🎯 STEP 6: LOG FILES")
    print("="*80)
    
    count = remove_pattern("**/*.log", "log files", force)
    stats['files_removed'] += count
    
    # Step 7: Verification
    print("\n" + "="*80)
    print("🎯 STEP 7: VERIFICATION")
    print("="*80)
    print()
    
    print(f"{Colors.BLUE}📊 Current directory status:{Colors.NC}\n")
    
    checks = [
        ("checkpoints/", "checkpoints"),
        ("wandb/", "wandb local logs"),
        ("outputs/", "Hydra outputs"),
        (".torch_compile_cache/", "Torch cache"),
    ]
    
    for path, name in checks:
        if os.path.exists(path):
            print(f"{Colors.YELLOW}⚠️  {name} still exists{Colors.NC}")
        else:
            print(f"{Colors.GREEN}✅ {name} removed{Colors.NC}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 CLEANUP SUMMARY")
    print("="*80)
    print()
    print(f"Directories removed: {Colors.GREEN}{stats['directories_removed']}{Colors.NC}")
    print(f"Files removed: {Colors.GREEN}{stats['files_removed']}{Colors.NC}")
    
    if not dry_run:
        print("\n" + "="*80)
        print(f"{Colors.GREEN}✅ CLEANUP COMPLETE!{Colors.NC}")
        print("="*80)
        print()
        print(f"{Colors.GREEN}🎉 You're ready for a fresh start!{Colors.NC}")
        print()
        print("To start new training:")
        print("  python unified_training.py --mode pretrain --epochs 10")
        print()
    else:
        print("\n" + "="*80)
        print(f"{Colors.CYAN}🔍 DRY RUN COMPLETE - No files were deleted{Colors.NC}")
        print("="*80)
        print()
        print("Run without --dry-run to actually delete files:")
        print("  python clean_and_reset.py")
        print()
    
    # W&B Cloud cleanup reminder
    print("="*80)
    print("☁️  WANDB CLOUD CLEANUP (OPTIONAL)")
    print("="*80)
    print()
    print(f"{Colors.YELLOW}Note: This script only removes LOCAL wandb data.{Colors.NC}")
    print(f"{Colors.YELLOW}To delete runs from wandb.ai cloud:{Colors.NC}")
    print()
    print("  1. Go to: https://wandb.ai/your-username/your-project")
    print("  2. Select runs to delete")
    print("  3. Click 'Delete' button")
    print()
    print("Or use wandb CLI:")
    print("  wandb run delete <run-id>")
    print()


def selective_clean():
    """Selective cleaning - ask for each category"""
    
    print("\n" + "="*80)
    print("🎯 SELECTIVE CLEANUP MODE")
    print("="*80)
    print()
    print("This will ask you about each category separately.")
    print()
    
    categories = [
        ("checkpoints", "All training checkpoints", "checkpoints/"),
        ("wandb", "W&B local logs and cache", ["wandb/", ".wandb/"]),
        ("hydra", "Hydra outputs", ["outputs/", "multirun/"]),
        ("python_cache", "Python cache (__pycache__, .pyc)", ["**/__pycache__", "**/*.pyc"]),
        ("torch_cache", "Torch compile cache", ".torch_compile_cache/"),
        ("logs", "Log files", "**/*.log"),
    ]
    
    for cat_id, cat_name, paths in categories:
        print(f"\n{Colors.BLUE}{'='*80}{Colors.NC}")
        print(f"{Colors.CYAN}{cat_name}{Colors.NC}")
        print(f"{Colors.BLUE}{'='*80}{Colors.NC}")
        
        if not confirm(f"\nClean {cat_name.lower()}?", default=False):
            print(f"{Colors.YELLOW}⏭️  Skipped {cat_name.lower()}{Colors.NC}")
            continue
        
        if isinstance(paths, list):
            for path in paths:
                if '*' in path:
                    remove_pattern(path, f"{cat_name} files", force=True)
                else:
                    remove_directory(path, cat_name, force=True)
        else:
            if '*' in paths:
                remove_pattern(paths, cat_name, force=True)
            else:
                remove_directory(paths, cat_name, force=True)


def main():
    parser = argparse.ArgumentParser(
        description="🧹 Clean and reset training environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (asks for confirmation)
  python clean_and_reset.py
  
  # Clean everything without asking
  python clean_and_reset.py --force
  
  # Dry run (show what would be deleted)
  python clean_and_reset.py --dry-run
  
  # Selective cleaning
  python clean_and_reset.py --selective
  
  # Just checkpoints
  python clean_and_reset.py --checkpoints-only
  
  # Just W&B
  python clean_and_reset.py --wandb-only
        """
    )
    
    parser.add_argument("--force", "-f", action="store_true",
                       help="Delete without asking for confirmation")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--selective", "-s", action="store_true",
                       help="Ask about each category separately")
    parser.add_argument("--checkpoints-only", action="store_true",
                       help="Only delete checkpoints")
    parser.add_argument("--wandb-only", action="store_true",
                       help="Only delete W&B data")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Clean everything (same as no arguments)")
    
    args = parser.parse_args()
    
    # Quick modes
    if args.checkpoints_only:
        print("\n🎯 Cleaning checkpoints only...")
        remove_directory("checkpoints", "all checkpoints", args.force)
        return
    
    if args.wandb_only:
        print("\n🎯 Cleaning W&B data only...")
        remove_directory("wandb", "W&B local logs", args.force)
        remove_directory(".wandb", "W&B cache", args.force)
        return
    
    # Selective mode
    if args.selective:
        selective_clean()
        return
    
    # Full clean (default)
    clean_all(force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Cleanup cancelled by user{Colors.NC}")
        exit(1)