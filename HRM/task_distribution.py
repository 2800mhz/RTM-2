"""
Task Distribution Handler for Meta-Learning

🧠 MANIFESTO PRINCIPLE: "Learn across diverse task variants"

Organizes puzzle dataset into task batches for meta-learning:
- Groups puzzles by type/difficulty
- Samples task batches for meta-training
- Handles support/query splits
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Iterator
from dataclasses import dataclass
from collections import defaultdict

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


@dataclass
class Task:
    """Single task definition"""
    task_id: int                    # Unique task identifier
    task_type: str                  # 'sudoku', 'arc', etc.
    difficulty: int                 # 0-4 (easy to hard)
    support_set: Dict[str, torch.Tensor]  # Training examples
    query_set: Dict[str, torch.Tensor]    # Test examples
    metadata: Dict                  # Additional info


class TaskDistribution:
    """
    Manages task sampling for meta-learning
    
    🧠 KEY FEATURES:
    1. Group puzzles by similarity (using puzzle_identifiers)
    2. Sample task batches for meta-training
    3. Split each task into support/query sets
    4. Track task statistics (difficulty, success rate)
    """
    
    def __init__(
        self,
        dataset: PuzzleDataset,
        n_support: int = 5,
        n_query: int = 10,
        task_batch_size: int = 4,
        seed: int = 42
    ):
        self.dataset = dataset
        self.n_support = n_support
        self.n_query = n_query
        self.task_batch_size = task_batch_size
        self.rng = np.random.RandomState(seed)
        
        # Load and organize data
        self._load_data()
        self._organize_tasks()
    
    def _load_data(self):
        """Load all data from dataset"""
        self.dataset._lazy_load_dataset()
        
        # Assuming single set for simplicity (extend for multi-set)
        set_name = self.dataset.metadata.sets[0]
        self.data = self.dataset._data[set_name]
        
        print(f"📂 Loaded {len(self.data['inputs'])} examples")
        print(f"   Puzzle IDs: {len(np.unique(self.data['puzzle_identifiers']))}")
        print(f"   Groups: {len(self.data['group_indices']) - 1}")
    
    def _organize_tasks(self):
        """
        Organize puzzles into tasks
        
        Each "group" in dataset = one task variant
        Each task has multiple examples (different puzzles of same type)
        """
        self.tasks = []
        
        num_groups = len(self.data['group_indices']) - 1
        
        for group_id in range(num_groups):
            # Get puzzles in this group
            puzzle_start = self.data['group_indices'][group_id]
            puzzle_end = self.data['group_indices'][group_id + 1]
            
            # Get example indices for all puzzles in group
            example_indices = []
            for puzzle_id in range(puzzle_start, puzzle_end):
                ex_start = self.data['puzzle_indices'][puzzle_id]
                ex_end = self.data['puzzle_indices'][puzzle_id + 1]
                example_indices.extend(range(ex_start, ex_end))
            
            # Skip if not enough examples
            if len(example_indices) < self.n_support + self.n_query:
                continue
            
            # Create task
            task_id = group_id
            puzzle_id = self.data['puzzle_identifiers'][puzzle_start]
            
            self.tasks.append({
                'task_id': task_id,
                'puzzle_id': int(puzzle_id),
                'example_indices': example_indices,
                'n_examples': len(example_indices)
            })
        
        print(f"\n✅ Organized {len(self.tasks)} tasks")
        print(f"   Avg examples per task: {np.mean([t['n_examples'] for t in self.tasks]):.1f}")
    
# task_distribution.py içinde sample_task metodunu TAM DEĞİŞTİRİN:

    def sample_task(self, task_info: Dict) -> Task:
        """
        Sample support and query sets for a single task
        """
        indices = task_info['example_indices']
        
        # Randomly sample support and query
        n_needed = self.n_support + self.n_query
        if len(indices) > n_needed:
            sampled = self.rng.choice(indices, size=n_needed, replace=False)
        else:
            sampled = indices
        
        support_idx = sampled[:self.n_support]
        query_idx = sampled[self.n_support:self.n_support + self.n_query]
        
        # Get device
        device = getattr(self, 'device', 'cpu')
        
        # ✅ CRITICAL FIX: Use actual puzzle_identifier from data, not task_id!
        # Get puzzle_identifier from the first example
        first_example_idx = indices[0]
        actual_puzzle_id = int(self.data['puzzle_identifiers'][first_example_idx])
        
        print(f"🔍 Task {task_info['task_id']}: Using puzzle_id={actual_puzzle_id}")
        
        # Extract data
        # task_distribution.py, extract_batch fonksiyonuna ekleyin:

        def extract_batch(idx_list):
            inputs_data = self.data['inputs'][idx_list]
            labels_data = self.data['labels'][idx_list]
            
            # ✅ FIX 1: Sudoku verileri 1-10 arasında, 0-9'a map et
            # 10 = boş hücre, 1-9 = rakamlar → 0 = boş, 1-9 = rakamlar
            inputs_data = inputs_data - 1  # 1-10 → 0-9
            labels_data = labels_data - 1  # 1-10 → 0-9
            
            # Güvenlik için kırp
            inputs_data = np.clip(inputs_data, 0, 9)
            labels_data = np.clip(labels_data, 0, 9)
            
            print(f"  ✅ After mapping: Inputs={inputs_data.min()}-{inputs_data.max()}, Labels={labels_data.min()}-{labels_data.max()}")
            
            return {
                'inputs': torch.from_numpy(inputs_data).long().to(device),
                'labels': torch.from_numpy(labels_data).long().to(device),
                'puzzle_identifiers': torch.full(
                    (len(idx_list),),
                    actual_puzzle_id,
                    dtype=torch.long,
                    device=device
                )
            }
        
        support_set = extract_batch(support_idx)
        query_set = extract_batch(query_idx)
        
        return Task(
            task_id=task_info['task_id'],
            task_type='sudoku',
            difficulty=0,
            support_set=support_set,
            query_set=query_set,
            metadata={
                'puzzle_id': actual_puzzle_id,
                'n_examples': task_info['n_examples']
            }
        )

    def sample_task_batch(self) -> List[Task]:
        """
        Sample a batch of tasks for meta-training
        
        Returns:
            List of Task objects
        """
        # Randomly sample tasks
        sampled_task_info = self.rng.choice(
            self.tasks, 
            size=min(self.task_batch_size, len(self.tasks)),
            replace=False
        )
        
        # Create Task objects
        task_batch = [self.sample_task(info) for info in sampled_task_info]
        
        return task_batch
    
    def __iter__(self) -> Iterator[List[Task]]:
        """
        Iterate over task batches
        
        Usage:
            for task_batch in task_distribution:
                # task_batch is List[Task]
                ...
        """
        while True:
            yield self.sample_task_batch()
    
    def get_task_by_id(self, task_id: int) -> Task:
        """Get specific task by ID"""
        task_info = next(t for t in self.tasks if t['task_id'] == task_id)
        return self.sample_task(task_info)


class BalancedTaskSampler:
    """
    Advanced sampler that balances task difficulty
    
    🧠 MANIFESTO: "Don't overtrain on easy tasks"
    """
    
    def __init__(
        self,
        task_distribution: TaskDistribution,
        difficulty_bins: int = 3
    ):
        self.dist = task_distribution
        self.difficulty_bins = difficulty_bins
        
        # Organize tasks by difficulty
        self._compute_difficulties()
        self._create_bins()
    
    def _compute_difficulties(self):
        """
        Estimate task difficulty
        
        Heuristic: More examples = easier (more patterns to learn)
        """
        for task in self.dist.tasks:
            # Simple heuristic: inverse of example count
            task['difficulty_score'] = 1.0 / (task['n_examples'] + 1)
    
    def _create_bins(self):
        """Group tasks by difficulty"""
        scores = [t['difficulty_score'] for t in self.dist.tasks]
        percentiles = np.linspace(0, 100, self.difficulty_bins + 1)
        thresholds = np.percentile(scores, percentiles)
        
        self.bins = [[] for _ in range(self.difficulty_bins)]
        
        for task in self.dist.tasks:
            score = task['difficulty_score']
            for i in range(self.difficulty_bins):
                if thresholds[i] <= score < thresholds[i + 1]:
                    self.bins[i].append(task)
                    break
        
        print(f"\n📊 Difficulty bins created:")
        for i, bin_tasks in enumerate(self.bins):
            print(f"   Bin {i}: {len(bin_tasks)} tasks")
    
    def sample_balanced_batch(self) -> List[Task]:
        """
        Sample tasks with balanced difficulty
        
        Returns:
            Mixed difficulty task batch
        """
        task_batch = []
        
        # Sample from each bin
        tasks_per_bin = self.dist.task_batch_size // self.difficulty_bins
        
        for bin_tasks in self.bins:
            if len(bin_tasks) == 0:
                continue
            
            sampled = self.dist.rng.choice(
                bin_tasks,
                size=min(tasks_per_bin, len(bin_tasks)),
                replace=False
            )
            
            for task_info in sampled:
                task_batch.append(self.dist.sample_task(task_info))
        
        return task_batch


# ============================================================================
# INTEGRATION WITH META-TRAINING
# ============================================================================

def create_meta_training_loader(
    data_path: str,
    n_support: int = 5,
    n_query: int = 10,
    task_batch_size: int = 4,
    num_meta_batches: int = 100
) -> List[List[Task]]:
    """
    Create task distribution for meta-training
    
    Args:
        data_path: Path to puzzle dataset
        n_support: Support set size per task
        n_query: Query set size per task
        task_batch_size: Number of tasks per meta-batch
        num_meta_batches: Total meta-batches to generate
    
    Returns:
        List of meta-batches (each is List[Task])
    """
    # Create dataset
    dataset_config = PuzzleDatasetConfig(
        seed=42,
        dataset_path=data_path,
        global_batch_size=32,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    
    dataset = PuzzleDataset(dataset_config, split='train')
    
    # Create task distribution
    task_dist = TaskDistribution(
        dataset=dataset,
        n_support=n_support,
        n_query=n_query,
        task_batch_size=task_batch_size
    )
    
    # Optional: Use balanced sampler
    # sampler = BalancedTaskSampler(task_dist)
    
    # Generate meta-batches
    meta_batches = []
    for _ in range(num_meta_batches):
        task_batch = task_dist.sample_task_batch()
        # Or: task_batch = sampler.sample_balanced_batch()
        meta_batches.append(task_batch)
    
    return meta_batches


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Create task distribution and sample tasks
    """
    
    # 1. Create dataset config
    config = PuzzleDatasetConfig(
        seed=42,
        dataset_path='data/sudoku-extreme-1k-aug-1000',
        global_batch_size=32,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    
    dataset = PuzzleDataset(config, split='train')
    
    # 2. Create task distribution
    print("\n" + "="*60)
    print("Creating Task Distribution")
    print("="*60)
    
    task_dist = TaskDistribution(
        dataset=dataset,
        n_support=5,
        n_query=10,
        task_batch_size=4
    )
    
    # 3. Sample a task batch
    print("\n" + "="*60)
    print("Sampling Task Batch")
    print("="*60)
    
    task_batch = task_dist.sample_task_batch()
    
    for i, task in enumerate(task_batch):
        print(f"\nTask {i+1}:")
        print(f"  ID: {task.task_id}")
        print(f"  Puzzle ID: {task.metadata['puzzle_id']}")
        print(f"  Support: {task.support_set['inputs'].shape}")
        print(f"  Query: {task.query_set['inputs'].shape}")
    
    # 4. Balanced sampling
    print("\n" + "="*60)
    print("Balanced Sampling")
    print("="*60)
    
    sampler = BalancedTaskSampler(task_dist, difficulty_bins=3)
    balanced_batch = sampler.sample_balanced_batch()
    
    print(f"\nBalanced batch size: {len(balanced_batch)}")
    
    print("\n✅ Task distribution demo complete!")