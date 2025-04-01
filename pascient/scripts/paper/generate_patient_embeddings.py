import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
import hydra
import tqdm
import numpy as np

# Load project root directory
import rootutils
rootutils.setup_root(search_from=".")

from pascient.utils.reprod import load_model, load_binary_model

class ForwardModel(torch.nn.Module):
    def __init__(self, base_model, last_layer = True):
        super().__init__()
        self.base_model = base_model
        self.last_layer = last_layer
    def forward(self, x, padding_mask):
        #assert x.shape[0] == 1
        cell_embds = self.base_model.gene2cell_encoder(x)
        cell_cross_embds = self.base_model.cell2cell_encoder(cell_embds, padding_mask = padding_mask)
        patient_embds = self.base_model.cell2patient_aggregation.aggregate(data = cell_cross_embds, mask = padding_mask)
        patient_embds_2 = self.base_model.patient_encoder(patient_embds)
        patient_preds = self.base_model.patient_predictor(patient_embds_2)
        if self.last_layer:
            return patient_embds_2, cell_cross_embds, patient_preds
        else:
            return patient_embds, cell_cross_embds, patient_preds
            
def load_model_():
    # Run Name
    run_name = "2025-02-09_22-25-15"
    model_name = "epoch_002" 
    config_path = f"/homefs/home/debroue1/projects/cellm/logs/train/runs/{run_name}/.hydra/"
    checkpoint_path = f"/homefs/home/debroue1/projects/cellm/logs/train/runs/{run_name}/checkpoints/{model_name}.ckpt"
    return load_model(config_path, checkpoint_path)


def load_binary_model_():
    # Run Name
    run_name = "2025-02-11_19-52-59"
    model_name = "epoch_099" 
    config_path = f"/homefs/home/debroue1/projects/cellm/logs/train/runs/{run_name}/.hydra/"
    checkpoint_path = f"/homefs/home/debroue1/projects/cellm/logs/train/runs/{run_name}/checkpoints/{model_name}.ckpt"
    return load_binary_model(config_path, checkpoint_path)

if __name__ == "__main__":

    OUTPUT_DIR = "cellm/interpretation/UMAP/"

    last_layer = True

    model, datamodule = load_model_()
    model_fwd = ForwardModel(model, last_layer = last_layer)

    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()

    h_tensor = []
    cell_tensor = []
    prob_tensor = []

    disease_tensor = []
    tissue_tensor = []
    celltype_tensor = []
    cellindex_tensor = []
    fold_list = []
    sample_list = []
    print("Number of batches:")
    print(len(train_dl))
    for i_b, batch in tqdm.tqdm(enumerate(train_dl)):
        
        x = batch.x
        padding_mask = batch.padded_mask
        
        h_pred, cell_h, prob_pred = model_fwd(x, padding_mask)
        
        h_tensor.append(h_pred.detach().cpu().numpy())
        cell_tensor.append(cell_h.detach().cpu().numpy())
        prob_tensor.append(prob_pred.detach().cpu().numpy())

        disease_tensor.append(batch.sample_metadata["disease"].detach().cpu().numpy())
        tissue_tensor.append(batch.sample_metadata["tissue"].detach().cpu().numpy())

        celltype_tensor.append(batch.cell_metadata.cell_level_labels["celltype_id"].detach().cpu().numpy())
        cellindex_tensor.append(batch.cell_metadata.cell_level_labels["index"].detach().cpu().numpy())

        fold_list.append("train")
        sample_list.append(batch.sample_metadata["study::::sample"].detach().cpu().numpy())
    
    print(len(val_dl))
    for i_b, batch in tqdm.tqdm(enumerate(val_dl)):
        
        x = batch.x
        padding_mask = batch.padded_mask
        
        h_pred, cell_h, prob_pred = model_fwd(x, padding_mask)
        
        h_tensor.append(h_pred.detach().cpu().numpy())
        cell_tensor.append(cell_h.detach().cpu().numpy())
        prob_tensor.append(prob_pred.detach().cpu().numpy())

        disease_tensor.append(batch.sample_metadata["disease"].detach().cpu().numpy())
        tissue_tensor.append(batch.sample_metadata["tissue"].detach().cpu().numpy())

        celltype_tensor.append(batch.cell_metadata.cell_level_labels["celltype_id"].detach().cpu().numpy())
        cellindex_tensor.append(batch.cell_metadata.cell_level_labels["index"].detach().cpu().numpy())

        fold_list.append("val")
        sample_list.append(batch.sample_metadata["study::::sample"].detach().cpu().numpy())


    print(len(test_dl))
    for i_b, batch in tqdm.tqdm(enumerate(val_dl)):
        
        x = batch.x
        padding_mask = batch.padded_mask
        
        h_pred, cell_h, prob_pred = model_fwd(x, padding_mask)
        
        h_tensor.append(h_pred.detach().cpu().numpy())
        cell_tensor.append(cell_h.detach().cpu().numpy())
        prob_tensor.append(prob_pred.detach().cpu().numpy())

        disease_tensor.append(batch.sample_metadata["disease"].detach().cpu().numpy())
        tissue_tensor.append(batch.sample_metadata["tissue"].detach().cpu().numpy())

        celltype_tensor.append(batch.cell_metadata.cell_level_labels["celltype_id"].detach().cpu().numpy())
        cellindex_tensor.append(batch.cell_metadata.cell_level_labels["index"].detach().cpu().numpy())

        fold_list.append("test")
        sample_list.append(batch.sample_metadata["study::::sample"].detach().cpu().numpy())

    # Concatenate all tensors.
    h_tensor = np.concatenate(h_tensor)[:,0]
    prob_tensor = np.concatenate(prob_tensor)
    disease_tensor = np.concatenate(disease_tensor)
    tissue_tensor = np.concatenate(tissue_tensor)

    tissue_dict = datamodule.output_map.labels2int["tissue"]
    tissue_dict = {v:k for k,v in tissue_dict.items()}
    disease_dict = datamodule.output_map.labels2int["disease"]
    disease_dict = {v:k for k,v in disease_dict.items()}
    tissue_list = [tissue_dict[i] for i in tissue_tensor]
    disease_list = [disease_dict[i] for i in disease_tensor]

    from io import BytesIO
    import boto3
    import pickle

    s3_resource = boto3.client('s3')
    bucket_name = 'prescient-braid-data'

    pickle_buffer = BytesIO()
    pickle.dump(h_tensor, pickle_buffer)
    pickle_buffer.seek(0)
    s3_file_name = f'{OUTPUT_DIR}full_patient_embeddings_last_layer_{last_layer}.pkl'
    s3_resource.upload_fileobj(pickle_buffer, bucket_name, s3_file_name)
    print(f"Saved to the s3 bucket at {s3_file_name}")

    pickle_buffer = BytesIO()
    pickle.dump(disease_list, pickle_buffer)
    pickle_buffer.seek(0)
    s3_file_name = f'{OUTPUT_DIR}full_disease_labels_last_layer_{last_layer}.pkl'
    s3_resource.upload_fileobj(pickle_buffer, bucket_name, s3_file_name)
    print(f"Saved to the s3 bucket at {s3_file_name}")

    pickle_buffer = BytesIO()
    pickle.dump(tissue_list, pickle_buffer)
    pickle_buffer.seek(0)
    s3_file_name = f'{OUTPUT_DIR}full_tissue_labels_last_layer_{last_layer}.pkl'
    s3_resource.upload_fileobj(pickle_buffer, bucket_name, s3_file_name)
    print(f"Saved to the s3 bucket at {s3_file_name}")

    pickle_buffer = BytesIO()
    pickle.dump(fold_list, pickle_buffer)
    pickle_buffer.seek(0)
    s3_file_name = f'{OUTPUT_DIR}full_fold_list_last_layer_{last_layer}.pkl'
    s3_resource.upload_fileobj(pickle_buffer, bucket_name, s3_file_name)
    print(f"Saved to the s3 bucket at {s3_file_name}")

    pickle_buffer = BytesIO()
    pickle.dump(cell_tensor, pickle_buffer)
    pickle_buffer.seek(0)
    s3_file_name = f'{OUTPUT_DIR}full_cell_embeddings_last_layer_{last_layer}.pkl'
    s3_resource.upload_fileobj(pickle_buffer, bucket_name, s3_file_name)
    print(f"Saved to the s3 bucket at {s3_file_name}")

    pickle_buffer = BytesIO()
    pickle.dump(celltype_tensor, pickle_buffer)
    pickle_buffer.seek(0)
    s3_file_name = f'{OUTPUT_DIR}full_celltype_labels_last_layer_{last_layer}.pkl'
    s3_resource.upload_fileobj(pickle_buffer, bucket_name, s3_file_name)
    print(f"Saved to the s3 bucket at {s3_file_name}")

    pickle_buffer = BytesIO()
    pickle.dump(cellindex_tensor, pickle_buffer)
    pickle_buffer.seek(0)
    s3_file_name = f'{OUTPUT_DIR}full_cellindex_labels_last_layer_{last_layer}.pkl'
    s3_resource.upload_fileobj(pickle_buffer, bucket_name, s3_file_name)
    print(f"Saved to the s3 bucket at {s3_file_name}")

    pickle_buffer = BytesIO()
    pickle.dump(prob_tensor, pickle_buffer)
    pickle_buffer.seek(0)
    s3_file_name = f'{OUTPUT_DIR}full_patient_predictions_last_layer_{last_layer}.pkl'
    s3_resource.upload_fileobj(pickle_buffer, bucket_name, s3_file_name)
    print(f"Saved to the s3 bucket at {s3_file_name}")

    pickle_buffer = BytesIO()
    pickle.dump(sample_list, pickle_buffer)
    pickle_buffer.seek(0)
    s3_file_name = f'{OUTPUT_DIR}full_sample_labels_last_layer_{last_layer}.pkl'
    s3_resource.upload_fileobj(pickle_buffer, bucket_name, s3_file_name)
    print(f"Saved to the s3 bucket at {s3_file_name}")
