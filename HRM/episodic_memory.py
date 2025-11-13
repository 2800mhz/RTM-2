"""
Episodic Memory for HRM-Free-Meta

🧠 MANIFESTO PRINCIPLE: "Remember what worked for similar tasks"

Stores task experiences (z_context, gates, success) and retrieves
similar past experiences to guide future adaptation.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle


@dataclass
class TaskEpisode:
    """Single task experience"""
    z_context: torch.Tensor      # [z_dim] - Task representation
    gates: torch.Tensor          # [2] - H/L weights used
    task_id: int                 # Puzzle identifier
    accuracy: float              # Success rate
    loss: float                  # Final loss
    timestamp: int               # When this was stored
    
    def to_dict(self):
        return {
            'z_context': self.z_context.float().cpu().numpy(),  # 🔥 BFloat16 → Float32
            'gates': self.gates.float().cpu().numpy(),          # 🔥 BFloat16 → Float32
            'task_id': self.task_id,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'timestamp': self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(
            z_context=torch.from_numpy(d['z_context']),
            gates=torch.from_numpy(d['gates']),
            task_id=d['task_id'],
            accuracy=d['accuracy'],
            loss=d['loss'],
            timestamp=d['timestamp'],
        )


class EpisodicMemory:
    """
    Memory buffer for task experiences
    
    🧠 KEY OPERATIONS:
    1. add() - Store new task experience
    2. retrieve_similar() - Find k most similar past tasks
    3. get_task_statistics() - Aggregate stats for task type
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        z_dim: int = 128,
        device: str = 'cuda'
    ):
        self.capacity = capacity
        self.z_dim = z_dim
        self.device = device
        
        # Storage
        self.episodes: List[TaskEpisode] = []
        self.step_counter = 0
        
        # Fast similarity search (using numpy for CPU efficiency)
        self.context_matrix = None  # [N, z_dim]
        self.dirty = True  # Rebuild index when True
    
    def add(
        self,
        z_context: torch.Tensor,
        gates: torch.Tensor,
        task_id: int,
        accuracy: float,
        loss: float
    ):
        """
        Add new task experience to memory
        
        Args:
            z_context: Task latent representation [z_dim]
            gates: Module weights used [2]
            task_id: Puzzle identifier
            accuracy: Success rate on this task
            loss: Final loss value
        """
        episode = TaskEpisode(
            z_context=z_context.detach().cpu(),
            gates=gates.detach().cpu(),
            task_id=task_id,
            accuracy=accuracy,
            loss=loss,
            timestamp=self.step_counter
        )
        
        self.episodes.append(episode)
        self.step_counter += 1
        self.dirty = True
        
        # Evict oldest if over capacity
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
    
    def _rebuild_index(self):
        """Rebuild search index for fast similarity queries"""
        if not self.dirty or len(self.episodes) == 0:
            return
        
        # Stack all contexts into matrix
        # 🔥 FIX: Convert BFloat16 to Float32 before numpy conversion
        contexts = [ep.z_context for ep in self.episodes]
        self.context_matrix = torch.stack(contexts).float().cpu().numpy()  # [N, z_dim]
        self.dirty = False
    
    def retrieve_similar(
        self,
        query_z: torch.Tensor,  # ✅ Correct parameter name
        k: int = 5,
        similarity_metric: str = 'cosine'
    ) -> List[Tuple[TaskEpisode, float]]:
        """
        Retrieve k most similar past task experiences
        
        Args:
            query_z: Query context [z_dim]
            k: Number of neighbors to return
            similarity_metric: 'cosine' or 'euclidean'
        
        Returns:
            List of (episode, similarity_score) tuples
        """
        if len(self.episodes) == 0:
            return []
        
        self._rebuild_index()
        
        # 🔥 FIX: Use correct parameter name and convert BFloat16 to Float32
        query = query_z.float().cpu().numpy().reshape(1, -1)
        
        if similarity_metric == 'cosine':
            # Cosine similarity
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            context_norms = self.context_matrix / (
                np.linalg.norm(self.context_matrix, axis=1, keepdims=True) + 1e-8
            )
            similarities = (context_norms @ query_norm.T).flatten()  # [N]
        
        elif similarity_metric == 'euclidean':
            # Negative L2 distance (higher = more similar)
            distances = np.linalg.norm(
                self.context_matrix - query, axis=1
            )
            similarities = -distances  # Negate so higher = better
        
        else:
            raise ValueError(f"Unknown metric: {similarity_metric}")
        
        # Get top-k indices
        k = min(k, len(self.episodes))
        topk_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return episodes with scores
        results = [
            (self.episodes[idx], float(similarities[idx]))
            for idx in topk_indices
        ]
        
        return results
    
    def get_task_statistics(self, task_id: int) -> Dict:
        """
        Get aggregate statistics for a specific task type
        
        Args:
            task_id: Puzzle identifier
        
        Returns:
            Dict with mean accuracy, success rate, preferred gates
        """
        # Filter episodes for this task
        task_episodes = [ep for ep in self.episodes if ep.task_id == task_id]
        
        if len(task_episodes) == 0:
            return {
                'count': 0,
                'mean_accuracy': 0.0,
                'best_accuracy': 0.0,
                'mean_H_weight': 0.5,
                'mean_L_weight': 0.5,
            }
        
        accuracies = [ep.accuracy for ep in task_episodes]
        gate_H = [ep.gates[0].item() for ep in task_episodes]
        gate_L = [ep.gates[1].item() for ep in task_episodes]
        
        return {
            'count': len(task_episodes),
            'mean_accuracy': np.mean(accuracies),
            'best_accuracy': np.max(accuracies),
            'worst_accuracy': np.min(accuracies),
            'mean_H_weight': np.mean(gate_H),
            'mean_L_weight': np.mean(gate_L),
            'std_H_weight': np.std(gate_H),
        }
    
    def get_best_gates_for_task(
        self, 
        task_id: int, 
        top_k: int = 3
    ) -> Optional[torch.Tensor]:
        """
        Get averaged gates from top-k best performing episodes for this task
        
        Args:
            task_id: Puzzle identifier
            top_k: Average over top-k best episodes
        
        Returns:
            gates: [2] averaged gates, or None if no history
        """
        task_episodes = [ep for ep in self.episodes if ep.task_id == task_id]
        
        if len(task_episodes) == 0:
            return None
        
        # Sort by accuracy
        task_episodes.sort(key=lambda ep: ep.accuracy, reverse=True)
        
        # Average top-k gates
        top_episodes = task_episodes[:top_k]
        gates_list = [ep.gates for ep in top_episodes]
        avg_gates = torch.stack(gates_list).mean(dim=0)
        
        return avg_gates
    
    def save(self, path: str):
        """Save memory to disk"""
        data = {
            'episodes': [ep.to_dict() for ep in self.episodes],
            'step_counter': self.step_counter,
            'capacity': self.capacity,
            'z_dim': self.z_dim,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Episodic memory saved to {path}")
    
    def load(self, path: str):
        """Load memory from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.episodes = [TaskEpisode.from_dict(d) for d in data['episodes']]
        self.step_counter = data['step_counter']
        self.capacity = data['capacity']
        self.z_dim = data['z_dim']
        self.dirty = True
        
        print(f"📂 Loaded {len(self.episodes)} episodes from {path}")


class MemoryAugmentedMetaLearning:
    """
    Meta-learning enhanced with episodic memory
    
    🧠 USAGE:
    1. Before adapting to new task: retrieve similar past tasks
    2. Use their successful gates as initialization hint
    3. After task: store experience to memory
    """
    
    def __init__(
        self,
        memory: EpisodicMemory,
        use_memory_initialization: bool = True,
        memory_weight: float = 0.3
    ):
        self.memory = memory
        self.use_memory_initialization = use_memory_initialization
        self.memory_weight = memory_weight
    
    def get_initialization_hint(
        self,
        z_context: torch.Tensor,
        task_id: Optional[int] = None
    ) -> Dict:
        """
        Get initialization hints from memory
        
        Args:
            z_context: Current task context
            task_id: If known, use task-specific history
        
        Returns:
            Dict with suggested gates, similar tasks info
        """
        if len(self.memory.episodes) == 0:
            return {'suggested_gates': None, 'similar_tasks': []}
        
        # Retrieve similar tasks (pass z_context correctly)
        similar = self.memory.retrieve_similar(z_context, k=5)
        
        # If task_id known, also get best gates for this specific task
        task_specific_gates = None
        if task_id is not None:
            task_specific_gates = self.memory.get_best_gates_for_task(task_id)
        
        # Compute weighted average of similar task gates
        if similar:
            gates_list = [ep.gates for ep, score in similar]
            similarities = torch.tensor([score for _, score in similar])
            
            # Weighted average by similarity
            weights = torch.softmax(similarities, dim=0)
            suggested_gates = sum(w * g for w, g in zip(weights, gates_list))
            
            # Blend with task-specific if available
            if task_specific_gates is not None:
                suggested_gates = (
                    self.memory_weight * task_specific_gates +
                    (1 - self.memory_weight) * suggested_gates
                )
        else:
            suggested_gates = task_specific_gates
        
        return {
            'suggested_gates': suggested_gates,
            'similar_tasks': [(ep.task_id, ep.accuracy, score) 
                             for ep, score in similar],
            'confidence': float(similarities.max()) if similar else 0.0
        }
    
    def store_experience(
        self,
        z_context: torch.Tensor,
        gates: torch.Tensor,
        task_id: int,
        accuracy: float,
        loss: float
    ):
        """Store task experience to memory"""
        self.memory.add(z_context, gates, task_id, accuracy, loss)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Using episodic memory in meta-learning
    """
    
    # 1. Create memory
    memory = EpisodicMemory(capacity=1000, z_dim=64)
    
    # 2. Simulate storing experiences
    print("📝 Storing task experiences...\n")
    
    for i in range(20):
        # Simulate task execution
        z_context = torch.randn(64)
        gates = torch.softmax(torch.randn(2), dim=0)
        task_id = i % 5  # 5 different task types
        accuracy = 0.5 + 0.3 * torch.rand(1).item()
        loss = 2.0 - accuracy
        
        memory.add(z_context, gates, task_id, accuracy, loss)
        
        if i % 5 == 0:
            print(f"  Stored episode {i}: task={task_id}, acc={accuracy:.2f}")
    
    print(f"\n✅ Memory contains {len(memory.episodes)} episodes")
    
    # 3. Retrieve similar tasks
    print("\n🔍 Retrieving similar tasks...\n")
    
    query_z = torch.randn(64)
    similar_tasks = memory.retrieve_similar(query_z, k=3)
    
    for i, (episode, similarity) in enumerate(similar_tasks):
        print(f"  {i+1}. Task {episode.task_id} (similarity={similarity:.3f})")
        print(f"     Accuracy: {episode.accuracy:.2f}")
        print(f"     Gates: H={episode.gates[0]:.3f}, L={episode.gates[1]:.3f}")
    
    # 4. Get task statistics
    print("\n📊 Task statistics for task_id=2:\n")
    
    stats = memory.get_task_statistics(task_id=2)
    print(f"  Count: {stats['count']}")
    print(f"  Mean accuracy: {stats['mean_accuracy']:.2f}")
    print(f"  Preferred H-weight: {stats['mean_H_weight']:.3f}")
    print(f"  Preferred L-weight: {stats['mean_L_weight']:.3f}")
    
    # 5. Save/load
    print("\n💾 Saving memory...")
    memory.save('episodic_memory.pkl')
    
    print("\n✅ Episodic memory demo complete!")