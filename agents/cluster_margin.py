from typing import Union, Callable
import numpy as np
import torch
import random
from collections import Counter
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent
from torch.utils.data import TensorDataset, DataLoader
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

class ClusterMargin(BaseAgent):

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget: int, added_images: int,
                      initial_test_acc: float, current_test_acc: float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size: int = 5000,
                      instances_multiplier: float = 1.25,
                      random_query: bool = True) -> list[int]:
        
        with torch.no_grad():

            # Get model predictions
            pred = self._predict(x_unlabeled, classifier)
            pred = torch.softmax(pred, dim=1)
            
            # Calculate margin (difference between top two probabilities)
            sorted_pred, _ = torch.sort(pred, dim=1, descending=True)
            max_probas = sorted_pred[:, 0]
            second_max_probas = sorted_pred[:, 1]
            uncertainty_estimates = 1 + second_max_probas - max_probas
            
            # Select top uncertain samples (multiplied by instances_multiplier)
            n_top_samples = int(instances_multiplier * self.query_size)
            top_indices = torch.topk(uncertainty_estimates, n_top_samples).indices.cpu().numpy()
            
            # Get embeddings for the uncertain samples
            embeddings = self._embed(x_unlabeled[top_indices], classifier).cpu().numpy()
            
            # Perform HAC with average linkage
            if len(embeddings) > 1:
                pairwise_dist = pdist(embeddings, metric='euclidean')
                Z = linkage(pairwise_dist, method='average')
                cluster_labels = fcluster(Z, t=self.query_size, criterion='maxclust')
            else:
                cluster_labels = np.array([1])
            
            # Implement the choose_cm_samples logic directly here
            cluster_sizes = Counter(cluster_labels)
            query_idx_by_clusters = {
                idx: list(top_indices[np.where(cluster_labels == idx)[0]])
                for idx in cluster_sizes.keys()
            }
            
            new_query_idx = []
            sorted_cluster = [el[1] for el in sorted([(v, k) for k, v in cluster_sizes.items()])]
            curr_idx = 0
            
            while sum(cluster_sizes.values()) > 0 and len(new_query_idx) < self.query_size:
                curr_cluster = sorted_cluster[curr_idx]
                if cluster_sizes[curr_cluster] == 0:
                    curr_idx = (curr_idx + 1) % len(sorted_cluster)
                    continue
                
                if random_query:
                    sample_idx = random.choice(range(len(query_idx_by_clusters[curr_cluster])))
                else:
                    sample_idx = 0
                
                new_query_idx.append(query_idx_by_clusters[curr_cluster][sample_idx])
                query_idx_by_clusters[curr_cluster] = np.delete(
                    query_idx_by_clusters[curr_cluster], sample_idx
                )
                cluster_sizes[curr_cluster] -= 1
                curr_idx = (curr_idx + 1) % len(sorted_cluster)
            
            chosen = new_query_idx

        return chosen