# Code Taken from here with adaptions to be usable:
# https://github.com/automl/auto-sklearn/blob/master/autosklearn/ensembles/ensemble_selection.py

import os
from collections import Counter
from typing import List, Optional, Union, Callable, Dict, Tuple
import numpy as np
from sklearn.utils import check_random_state
from assembled_ensembles.wrapper.abstract_weighted_ensemble import AbstractWeightedEnsemble
from assembled_ensembles.util.metrics import AbstractMetric
from concurrent.futures import ProcessPoolExecutor
from abc import ABC, abstractmethod
from enum import Enum


class SelectionStrategy(Enum):
    """Enumeration of different selection strategies"""
    GREEDY = "greedy"
    RANK_BASED = "rank_based"
    DIVERSITY_AWARE = "diversity_aware"
    EXPERT_WEIGHTED = "expert_weighted"


class ExpertModelSelection(ABC):
    """Abstract base class for expert model selection criteria"""
    
    @abstractmethod
    def evaluate_model(self, model_idx: int, predictions: List[np.ndarray], 
                      labels: np.ndarray, current_ensemble: List[int]) -> float:
        """Evaluate a single model according to expert criteria"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this expert criterion"""
        pass


class AccuracyExpert(ExpertModelSelection):
    """Expert that prioritizes model accuracy"""
    
    def __init__(self, metric: AbstractMetric, weight: float = 1.0):
        self.metric = metric
        self.weight = weight
    
    def evaluate_model(self, model_idx: int, predictions: List[np.ndarray], 
                      labels: np.ndarray, current_ensemble: List[int]) -> float:
        # Calculate accuracy of this model alone
        return self.metric(labels, predictions[model_idx], to_loss=True) * self.weight
    
    def get_name(self) -> str:
        return "AccuracyExpert"


class DiversityExpert(ExpertModelSelection):
    """Expert that prioritizes model diversity"""
    
    def __init__(self, weight: float = 1.0, diversity_metric: str = "disagreement"):
        self.weight = weight
        self.diversity_metric = diversity_metric
    
    def evaluate_model(self, model_idx: int, predictions: List[np.ndarray], 
                      labels: np.ndarray, current_ensemble: List[int]) -> float:
        if not current_ensemble:
            return 0.0  # No diversity to measure for first model
        
        # Calculate diversity with current ensemble
        ensemble_pred = np.mean([predictions[i] for i in current_ensemble], axis=0)
        model_pred = predictions[model_idx]
        
        if self.diversity_metric == "disagreement":
            # Measure disagreement rate
            ensemble_classes = np.argmax(ensemble_pred, axis=1)
            model_classes = np.argmax(model_pred, axis=1)
            disagreement = np.mean(ensemble_classes != model_classes)
            return disagreement * self.weight
        else:
            # Default: correlation-based diversity
            correlation = np.corrcoef(ensemble_pred.flatten(), model_pred.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0
            return (1 - abs(correlation)) * self.weight
    
    def get_name(self) -> str:
        return "DiversityExpert"


class ComplexityExpert(ExpertModelSelection):
    """Expert that considers model complexity"""
    
    def __init__(self, model_complexities: List[float], weight: float = 1.0):
        self.model_complexities = model_complexities
        self.weight = weight
    
    def evaluate_model(self, model_idx: int, predictions: List[np.ndarray], 
                      labels: np.ndarray, current_ensemble: List[int]) -> float:
        complexity = self.model_complexities[model_idx]
        # Prefer simpler models (lower complexity)
        return -complexity * self.weight
    
    def get_name(self) -> str:
        return "ComplexityExpert"


class MultiCriteriaExpertSelector:
    """Multi-criteria expert model selection system"""
    
    def __init__(self, experts: List[ExpertModelSelection], 
                 strategy: SelectionStrategy = SelectionStrategy.EXPERT_WEIGHTED,
                 expert_weights: Optional[List[float]] = None):
        self.experts = experts
        self.strategy = strategy
        
        if expert_weights is None:
            # Default: equal weights for all experts
            self.expert_weights = [1.0 / len(experts) for _ in experts]
        else:
            self.expert_weights = expert_weights
    
    def select_model(self, candidate_indices: List[int], predictions: List[np.ndarray],
                    labels: np.ndarray, current_ensemble: List[int]) -> int:
        """Select the best model using multi-criteria expert selection"""
        
        if self.strategy == SelectionStrategy.GREEDY:
            return self._greedy_selection(candidate_indices, predictions, labels, current_ensemble)
        elif self.strategy == SelectionStrategy.RANK_BASED:
            return self._rank_based_selection(candidate_indices, predictions, labels, current_ensemble)
        elif self.strategy == SelectionStrategy.DIVERSITY_AWARE:
            return self._diversity_aware_selection(candidate_indices, predictions, labels, current_ensemble)
        elif self.strategy == SelectionStrategy.EXPERT_WEIGHTED:
            return self._expert_weighted_selection(candidate_indices, predictions, labels, current_ensemble)
        else:
            raise ValueError(f"Unknown selection strategy: {self.strategy}")
    
    def _greedy_selection(self, candidate_indices: List[int], predictions: List[np.ndarray],
                         labels: np.ndarray, current_ensemble: List[int]) -> int:
        """Traditional greedy selection based on primary expert (accuracy)"""
        best_score = float('inf')
        best_model = candidate_indices[0]
        
        for model_idx in candidate_indices:
            score = self.experts[0].evaluate_model(model_idx, predictions, labels, current_ensemble)
            if score < best_score:
                best_score = score
                best_model = model_idx
        
        return best_model
    
    def _rank_based_selection(self, candidate_indices: List[int], predictions: List[np.ndarray],
                            labels: np.ndarray, current_ensemble: List[int]) -> int:
        """Rank-based selection combining multiple criteria"""
        ranks = {idx: [] for idx in candidate_indices}
        
        # Get ranks from each expert
        for expert in self.experts:
            scores = []
            for model_idx in candidate_indices:
                score = expert.evaluate_model(model_idx, predictions, labels, current_ensemble)
                scores.append((model_idx, score))
            
            # Sort by score (lower is better)
            scores.sort(key=lambda x: x[1])
            
            # Assign ranks
            for rank, (model_idx, _) in enumerate(scores):
                ranks[model_idx].append(rank)
        
        # Calculate combined rank (lower is better)
        best_combined_rank = float('inf')
        best_model = candidate_indices[0]
        
        for model_idx, rank_list in ranks.items():
            combined_rank = sum(rank_list)
            if combined_rank < best_combined_rank:
                best_combined_rank = combined_rank
                best_model = model_idx
        
        return best_model
    
    def _diversity_aware_selection(self, candidate_indices: List[int], predictions: List[np.ndarray],
                                 labels: np.ndarray, current_ensemble: List[int]) -> int:
        """Diversity-aware selection that balances accuracy and diversity"""
        if len(current_ensemble) < 2:
            # Early stages: prioritize accuracy
            accuracy_expert = next((e for e in self.experts if e.get_name() == "AccuracyExpert"), None)
            if accuracy_expert:
                return self._greedy_selection(candidate_indices, predictions, labels, current_ensemble)
        
        # Calculate weighted scores
        best_score = float('inf')
        best_model = candidate_indices[0]
        
        for model_idx in candidate_indices:
            weighted_score = 0
            for expert, weight in zip(self.experts, self.expert_weights):
                score = expert.evaluate_model(model_idx, predictions, labels, current_ensemble)
                weighted_score += score * weight
            
            if weighted_score < best_score:
                best_score = weighted_score
                best_model = model_idx
        
        return best_model
    
    def _expert_weighted_selection(self, candidate_indices: List[int], predictions: List[np.ndarray],
                                 labels: np.ndarray, current_ensemble: List[int]) -> int:
        """Expert-weighted selection using all criteria with configurable weights"""
        best_score = float('inf')
        best_model = candidate_indices[0]
        
        for model_idx in candidate_indices:
            total_score = 0
            for expert, weight in zip(self.experts, self.expert_weights):
                expert_score = expert.evaluate_model(model_idx, predictions, labels, current_ensemble)
                total_score += expert_score * weight
            
            if total_score < best_score:
                best_score = total_score
                best_model = model_idx
        
        return best_model


class EnsembleSelection(AbstractWeightedEnsemble):
    """An ensemble of selected algorithms with multi-criteria expert selection

    Fitting an EnsembleSelection generates an ensemble from the models
    generated during the search process. Can be further used for prediction.

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    metric: AbstractMetric
        The metric used to evaluate the models
    n_jobs: int, default=-1
        Number of processes to use for parallelization. -1 means all available.
    random_state: Optional[int | RandomState] = None
        The random_state used for ensemble selection.
        *   None - Uses numpy's default RandomState object
        *   int - Successive calls to fit will produce the same results
        *   RandomState - Truely random, each call to fit will produce
                          different results, even with the same object.
    use_best: bool = True
        After finishing all iterations, use the best found ensemble instead of the last found ensemble.
    multi_criteria_config: Optional[Dict] = None
        Configuration for multi-criteria expert selection. If None, uses traditional greedy selection.
        Example: {
            'strategy': 'expert_weighted',
            'experts': [
                {'type': 'accuracy', 'weight': 0.6},
                {'type': 'diversity', 'weight': 0.3},
                {'type': 'complexity', 'weight': 0.1, 'complexities': [1.0, 2.0, ...]}
            ]
        }
    """

    def __init__(self, base_models: List[Callable], n_iterations: int, metric: AbstractMetric, n_jobs: int = -1,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 use_best: bool = False, multi_criteria_config: Optional[Dict] = None) -> None:

        super().__init__(base_models, "predict_proba")
        self.ensemble_size = n_iterations
        self.metric = metric
        self.use_best = use_best
        self.multi_criteria_config = multi_criteria_config
        self.expert_selector = None

        # Initialize multi-criteria expert selector if configured
        if multi_criteria_config:
            self._initialize_expert_selector(multi_criteria_config)

        # -- Code for multiprocessing
        if (n_jobs == 1) or (os.name == "nt"):
            self._use_mp = False
            if os.name == "nt":
                print("WARNING: n_jobs != 1 is not supported on Windows. Setting n_jobs=1.")
        else:
            if n_jobs == -1:
                n_jobs = len(os.sched_getaffinity(0))
            self._n_jobs = n_jobs
            self._use_mp = True

        # Behaviour similar to sklearn
        self.random_state = random_state

    def _initialize_expert_selector(self, config: Dict) -> None:
        """Initialize the multi-criteria expert selector based on configuration"""
        experts = []
        expert_weights = []
        
        strategy = SelectionStrategy(config.get('strategy', 'expert_weighted'))
        
        for expert_config in config.get('experts', []):
            expert_type = expert_config['type']
            weight = expert_config.get('weight', 1.0)
            
            if expert_type == 'accuracy':
                expert = AccuracyExpert(self.metric, weight)
            elif expert_type == 'diversity':
                diversity_metric = expert_config.get('diversity_metric', 'disagreement')
                expert = DiversityExpert(weight, diversity_metric)
            elif expert_type == 'complexity':
                complexities = expert_config.get('complexities', [1.0] * len(self.base_models))
                expert = ComplexityExpert(complexities, weight)
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")
            
            experts.append(expert)
            expert_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(expert_weights)
        if total_weight > 0:
            expert_weights = [w / total_weight for w in expert_weights]
        
        self.expert_selector = MultiCriteriaExpertSelector(
            experts=experts,
            strategy=strategy,
            expert_weights=expert_weights
        )

    def ensemble_fit(self, predictions: List[np.ndarray], labels: np.ndarray) -> AbstractWeightedEnsemble:
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')
        if not isinstance(self.metric, AbstractMetric):
            raise ValueError("The provided metric must be an instance of a AbstractMetric, "
                             "nevertheless it is {}({})".format(
                self.metric,
                type(self.metric),
            ))

        self._fit(predictions, labels)
        self.apply_use_best()
        self._calculate_final_weights()

        # -- Set metadata correctly
        self.iteration_batch_size_ = len(predictions)

        return self

    def _fit(self, predictions: List[np.ndarray], labels: np.ndarray) -> None:
        """Fast version of Rich Caruana's ensemble selection method with multi-criteria support."""
        self.num_input_models_ = len(predictions)
        rand = check_random_state(self.random_state)

        ensemble = []  # type: List[np.ndarray]
        trajectory = []  # contains iteration best
        self.val_loss_over_iterations_ = []  # contains overall best
        order = []

        ensemble_size = self.ensemble_size

        weighted_ensemble_prediction = np.zeros(
            predictions[0].shape,
            dtype=np.float64,
        )
        fant_ensemble_prediction = np.zeros(
            weighted_ensemble_prediction.shape,
            dtype=np.float64,
        )

        # Available model indices
        available_models = list(range(len(predictions)))

        for i in range(ensemble_size):
            s = len(ensemble)
            if s > 0:
                np.add(
                    weighted_ensemble_prediction,
                    ensemble[-1],
                    out=weighted_ensemble_prediction,
                )

            # -- Model selection based on strategy
            if self.expert_selector:
                # Use multi-criteria expert selection
                best = self.expert_selector.select_model(
                    available_models, predictions, labels, order
                )
                
                # Calculate the loss for the selected model for tracking
                np.add(weighted_ensemble_prediction, predictions[best], out=fant_ensemble_prediction)
                np.multiply(fant_ensemble_prediction, (1. / float(s + 1)), out=fant_ensemble_prediction)
                ensemble_loss = self.metric(labels, fant_ensemble_prediction, to_loss=True)
            else:
                # Traditional greedy selection
                if self._use_mp:
                    losses = self._compute_losses_mp(weighted_ensemble_prediction, labels, predictions, s)
                else:
                    losses = self._compute_losses_single(weighted_ensemble_prediction, labels, predictions, s)

                # Select best model
                all_best = np.argwhere(losses == np.nanmin(losses)).flatten()
                best = rand.choice(all_best)
                ensemble_loss = losses[best]

            ensemble.append(predictions[best])
            trajectory.append(ensemble_loss)
            order.append(best)

            # Build Correct Validation loss list
            if not self.val_loss_over_iterations_:
                self.val_loss_over_iterations_.append(ensemble_loss)
            elif self.val_loss_over_iterations_[-1] > ensemble_loss:
                self.val_loss_over_iterations_.append(ensemble_loss)
            else:
                self.val_loss_over_iterations_.append(self.val_loss_over_iterations_[-1])

            # -- Handle special cases
            if len(predictions) == 1:
                break

            # If we find a perfect ensemble/model, stop early
            if ensemble_loss == 0:
                break

        self.indices_ = order
        self.trajectory_ = trajectory

    def _compute_losses_single(self, weighted_ensemble_prediction, labels, predictions, s):
        """Compute losses for all models (single-process version)"""
        losses = np.zeros((len(predictions)), dtype=np.float64)
        
        for j, pred in enumerate(predictions):
            np.add(weighted_ensemble_prediction, pred, out=fant_ensemble_prediction)
            np.multiply(fant_ensemble_prediction, (1. / float(s + 1)), out=fant_ensemble_prediction)
            losses[j] = self.metric(labels, fant_ensemble_prediction, to_loss=True)
        
        return losses

    def _compute_losses_mp(self, weighted_ensemble_prediction, labels, predictions, s):
        # -- Process Iteration Solutions
        func_args = (weighted_ensemble_prediction, labels, s, self.metric, predictions)
        pred_i_list = list(range(len(predictions)))

        with ProcessPoolExecutor(self._n_jobs, initializer=_pool_init, initargs=func_args) as ex:
            results = ex.map(_init_wrapper_evaluate_single_solution, pred_i_list)

        return np.array(list(results))

    def apply_use_best(self):
        if self.use_best:
            min_score = np.min(self.trajectory_)
            idx_best = self.trajectory_.index(min_score)
            self.indices_ = self.indices_[:idx_best + 1]
            self.trajectory_ = self.trajectory_[:idx_best + 1]
            self.ensemble_size = idx_best + 1
            self.validation_loss_ = self.trajectory_[idx_best]
        else:
            self.validation_loss_ = self.trajectory_[-1]
            self.val_loss_over_iterations_ = self.trajectory_

    def _calculate_final_weights(self) -> None:
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=np.float64)
        
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def get_selection_strategy_info(self) -> Dict:
        """Get information about the selection strategy used"""
        if self.expert_selector:
            return {
                'strategy': self.expert_selector.strategy.value,
                'experts': [expert.get_name() for expert in self.expert_selector.experts],
                'weights': self.expert_selector.expert_weights
            }
        else:
            return {'strategy': 'traditional_greedy'}


def _pool_init(_weighted_ensemble_prediction, _labels, _sample_size, _score_metric, _predictions):
    global p_weighted_ensemble_prediction, p_labels, p_sample_size, p_score_metric, p_predictions
    p_weighted_ensemble_prediction = _weighted_ensemble_prediction
    p_labels = _labels
    p_sample_size = _sample_size
    p_score_metric = _score_metric
    p_predictions = _predictions


def _init_wrapper_evaluate_single_solution(pred_index):
    return evaluate_single_solution(p_weighted_ensemble_prediction, p_labels, p_sample_size, p_score_metric,
                                    p_predictions[pred_index])


def evaluate_single_solution(weighted_ensemble_prediction, labels, sample_size, score_metric, pred):
    fant_ensemble_prediction = np.add(weighted_ensemble_prediction, pred)
    np.multiply(fant_ensemble_prediction, (1. / float(sample_size + 1)), out=fant_ensemble_prediction)
    return score_metric(labels, fant_ensemble_prediction, to_loss=True)


# Example usage and demonstration
if __name__ == "__main__":
    # Example configuration for multi-criteria expert selection
    multi_criteria_config = {
        'strategy': 'expert_weighted',
        'experts': [
            {'type': 'accuracy', 'weight': 0.6},
            {'type': 'diversity', 'weight': 0.3},
            {'type': 'complexity', 'weight': 0.1, 'complexities': [1.0, 2.0, 1.5, 3.0]}
        ]
    }
    
    print("Multi-criteria Expert Ensemble Selection Module")
    print("Enhanced with multiple selection strategies:")
    print("- Greedy: Traditional accuracy-based selection")
    print("- Rank-based: Combines rankings from multiple experts")
    print("- Diversity-aware: Balances accuracy and diversity")
    print("- Expert-weighted: Weighted combination of multiple criteria")