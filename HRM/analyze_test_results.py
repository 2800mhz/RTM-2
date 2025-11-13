#!/usr/bin/env python3
"""
Analyze test results from comprehensive configuration testing
Usage: python analyze_test_results.py test_results/20251112_123456/
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_metrics_from_log(log_file: str) -> Dict:
    """Extract key metrics from training log"""
    metrics = {
        'completed': False,
        'num_params': None,
        'final_loss': None,
        'final_accuracy': None,
        'meta_loss': None,
        'query_accuracy': None,
        'training_time': None,
        'errors': []
    }
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Check completion
        metrics['completed'] = 'TRAINING COMPLETE' in content
        
        # Extract model size
        match = re.search(r'Model created: ([\d.]+)M parameters', content)
        if match:
            metrics['num_params'] = float(match.group(1))
        
        # Extract training time
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'Training:' in line and '100%' in line:
                # Extract time from progress bar
                time_match = re.search(r'\[(\d+:\d+)<', line)
                if time_match:
                    metrics['training_time'] = time_match.group(1)
        
        # Extract final metrics
        # Standard training
        matches = re.findall(r"train/loss[\"']:\s*([\d.]+)", content)
        if matches:
            metrics['final_loss'] = float(matches[-1])
        
        # Meta pretraining
        matches = re.findall(r'meta/total_loss:\s*([\d.]+)', content)
        if matches:
            metrics['meta_loss'] = float(matches[-1])
        
        # Meta-learning
        matches = re.findall(r'Query Acc:\s*([\d.]+)%', content)
        if matches:
            metrics['query_accuracy'] = float(matches[-1])
        
        # Extract errors
        error_lines = [line for line in lines if 'Error' in line or 'Exception' in line]
        metrics['errors'] = error_lines[:5]  # First 5 errors
        
    except Exception as e:
        metrics['errors'].append(f"Failed to parse log: {e}")
    
    return metrics


def analyze_test_directory(test_dir: str):
    """Analyze all tests in a directory"""
    test_dir = Path(test_dir)
    
    if not test_dir.exists():
        print(f"❌ Directory not found: {test_dir}")
        return
    
    print("="*80)
    print("📊 HRM-FREE-META TEST ANALYSIS")
    print("="*80)
    print(f"Directory: {test_dir}")
    print()
    
    # Find all log files
    log_files = sorted(test_dir.glob("test*/*.log")) + sorted(test_dir.glob("test*.log"))
    
    if not log_files:
        print("❌ No log files found")
        return
    
    results = []
    
    for log_file in log_files:
        test_name = log_file.stem.replace('_', ' ').title()
        print(f"\n{'='*80}")
        print(f"📝 {test_name}")
        print('='*80)
        
        metrics = extract_metrics_from_log(str(log_file))
        results.append((test_name, metrics))
        
        # Status
        if metrics['completed']:
            print("✅ Status: COMPLETED")
        else:
            print("❌ Status: FAILED or INCOMPLETE")
        
        # Model info
        if metrics['num_params']:
            print(f"🔧 Model Size: {metrics['num_params']:.2f}M parameters")
        
        # Training time
        if metrics['training_time']:
            print(f"⏱️  Training Time: {metrics['training_time']}")
        
        # Performance metrics
        if metrics['final_loss'] is not None:
            print(f"📉 Final Loss: {metrics['final_loss']:.4f}")
        
        if metrics['meta_loss'] is not None:
            print(f"📉 Meta Loss: {metrics['meta_loss']:.4f}")
        
        if metrics['query_accuracy'] is not None:
            print(f"🎯 Query Accuracy: {metrics['query_accuracy']:.2f}%")
        
        # Errors
        if metrics['errors']:
            print(f"\n⚠️  Errors/Warnings:")
            for error in metrics['errors'][:3]:
                print(f"   - {error[:100]}...")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, m in results if m['completed'])
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    # Model comparison
    print("\n" + "-"*80)
    print("Model Sizes:")
    for name, metrics in results:
        if metrics['num_params']:
            print(f"  {name:40} {metrics['num_params']:>8.2f}M params")
    
    # Performance comparison
    print("\n" + "-"*80)
    print("Learning Performance:")
    for name, metrics in results:
        perf_str = ""
        if metrics['final_loss']:
            perf_str = f"Loss: {metrics['final_loss']:.4f}"
        elif metrics['meta_loss']:
            perf_str = f"Meta Loss: {metrics['meta_loss']:.4f}"
        elif metrics['query_accuracy']:
            perf_str = f"Query Acc: {metrics['query_accuracy']:.2f}%"
        
        if perf_str:
            status = "✅" if metrics['completed'] else "❌"
            print(f"  {status} {name:40} {perf_str}")
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_test_results.py <test_results_directory>")
        print("\nExample:")
        print("  python analyze_test_results.py test_results/20251112_204500/")
        sys.exit(1)
    
    analyze_test_directory(sys.argv[1])