import torch
import torchmetrics
from torchmetrics import Metric
import numpy as np
from pascient.data.data_structures import SampleBatch
from pascient.components.masking import Masking


class R2ScoreMetric(torchmetrics.R2Score):
    """
    Wrapper around torchmetrics.R2Score to handle 3D tensors and multiple outputs from the model.
    Only preds_cells and true_cells are taken into account
    """
    def __init__(self, num_outputs: int, name: str = "r2"):
        super().__init__(num_outputs=num_outputs)
        self.name = name

    def convert_shape(self, preds: torch.Tensor, targets: torch.Tensor):
        if len(preds.shape) > 2:
            return preds.reshape((-1,preds.shape[-1])), targets.reshape((-1,targets.shape[-1]))  
        else:
            return preds, targets
    
    def update(self, preds_cells: torch.Tensor, true_batch: SampleBatch, masking_strategy: Masking, **kwargs) -> None:
        
        true_x, pred_x = masking_strategy.compare_reconstructions(true_batch, preds_cells)

        super().update(*self.convert_shape(pred_x, true_x))


class AccuracyMetric(Metric):
    """
    Wrapper around torchmetrics.Accuracy to handle 3D tensors and multiple outputs from the model.
    """
    def __init__(self, task: str, num_classes: int, labels: str, name: str = "accuracy"):
        super().__init__()
        self.name = name
        self.labels = labels
        assert len(labels) == 1 # if we want a metric for each label, we need to create multiple instances of this class
        
        self.accuracy_class = torchmetrics.Accuracy(num_classes=num_classes, task = task)

    def update(self, patient_preds: torch.Tensor, true_batch: SampleBatch, **kwargs) -> None:
        """
        patient preds is a tensor of shape (N, V, num_classes) with V the number of views
        true_batch is a SampleBatch object

        We update the metrics with all predictions from the different views.
        """
        
        for view_id in range(patient_preds.shape[1]):
            self.accuracy_class.update(patient_preds[:,view_id], true_batch.sample_metadata[self.labels[0]])

    def compute(self):
        return self.accuracy_class.compute()


class MSEMaskMetric(torchmetrics.MeanSquaredError):
    """
    MSE metric on the masked values
    """
    def __init__(self, name: str = "mse_masked"):
        super().__init__()
        self.name = name

    def convert_shape(self, preds:torch.Tensor, targets: torch.Tensor):
        if len(preds.shape) > 2:
            return preds.reshape((-1,preds.shape[-1])), targets.reshape((-1,targets.shape[-1]))  
        else:
            return preds, targets
        
    def update(self, preds_cells: torch.Tensor, true_batch: SampleBatch, masking_strategy: Masking, **kwargs) -> None:
        
        true_x, pred_x = masking_strategy.compare_reconstructions(true_batch, preds_cells)

        super().update(*self.convert_shape(pred_x, true_x))

class ContrastiveAccuracy(Metric):
    """
    Metric to compute the accuracy of the
    contrastive learning model.
    """

    def __init__(self, name: str = "contrastive_accuracy"):
        super().__init__()
        self.embds = []
        self.labels = []
        self.name = name

    def update(self, patient_embds: torch.Tensor, true_batch: SampleBatch, **kwargs):
        """
        Update the accuracy metric.

        Args:
            outputs (torch.Tensor): The model outputs.
            targets (torch.Tensor): The targets.
        """
        
        self.embds = self.embds + [patient_embds[:,i_view] for i_view in range(patient_embds.shape[1]) ]
        self.labels.append(torch.tile(true_batch.sample_metadata["study::::sample"], (patient_embds.shape[1],)))

    def compute(self):
        """
        Compute the accuracy.

        Returns:
            float: The accuracy.
        """
        labels = torch.cat(self.labels).to(self.embds[0].device)
        embds = torch.cat(self.embds)
        
        cross_sim = embds @ embds.T
        closest_idx = (cross_sim - torch.eye(cross_sim.shape[0], device=cross_sim.device)).argmax(1) # Exclude the diagonal
        closest_label = labels[closest_idx]

        acc = (closest_label == labels).float().mean()
        
        self.embds = []
        self.labels = []

        return acc

class LinearProbeMetric(Metric):
    """
    Metric to compute the accuracy of the
    linear probe model.
    """

    def __init__(self, name: str = "linear_probe_accuracy", target_name: str = "tissue", view_name: str = "view_0"):
        super().__init__()
        self.embds = []
        self.labels = []
        self.name = name
        self.target_name = target_name
        self.view_name = view_name

    def update(self, patient_embds: torch.Tensor, true_batch: SampleBatch, **kwargs):
        """
        Update the linear probing metric.

        Args:
            outputs (torch.Tensor): The model outputs.
            targets (torch.Tensor): The targets.
        """
        
        view_idx = true_batch.view_names.index(self.view_name)
        self.embds.append(patient_embds[:,view_idx])
        self.labels.append(true_batch.sample_metadata[self.target_name])

    def compute(self):
        """
        Compute the accuracy over 3 folds CV.

        Returns:
            float: The accuracy averaged over 3 folds.
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression

        labels = torch.cat(self.labels).cpu().numpy()
        embds = torch.cat(self.embds).cpu().numpy()
        
        model = LogisticRegression()
        # evaluate model
        scores = cross_val_score(model, embds, labels, scoring='accuracy', cv = 3)
        
        self.embds = []
        self.labels = []

        return scores.mean()

#@torch.no_grad()
#def simclr_accuracy(embeds: torch.Tensor) -> torch.Tensor:
#    cross_sim = torch.mm(embeds[:, 0], embeds[:, 1].transpose(1, 0))
#    return (cross_sim.argmax(dim=1) == torch.arange(cross_sim.shape[0], device=cross_sim.device)).float().mean()