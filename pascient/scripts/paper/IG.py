from io import StringIO
import tqdm
import boto3
import pandas as pd

import torch
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf
import hydra
import tqdm
import numpy as np

from captum.attr import IntegratedGradients

import argparse

from pascient.utils.reprod import load_model, load_binary_model


# Load project root directory
import rootutils
rootutils.setup_root(search_from=".")


class ForwardModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, x, padding_mask):
        #assert x.shape[0] == 1
        cell_embds = self.base_model.gene2cell_encoder(x)
        cell_cross_embds = self.base_model.cell2cell_encoder(cell_embds, padding_mask = padding_mask)
        patient_embds = self.base_model.cell2patient_aggregation.aggregate(data = cell_cross_embds, mask = padding_mask)
        patient_embds = self.base_model.patient_encoder(patient_embds)
        patient_preds = self.base_model.patient_predictor(patient_embds)

        return torch.softmax(patient_preds[:,0], dim=-1)
    
class ForwardDiffModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, x, padding_mask):
        #assert x.shape[0] == 1
        cell_embds = self.base_model.gene2cell_encoder(x)
        cell_cross_embds = self.base_model.cell2cell_encoder(cell_embds, padding_mask = padding_mask)
        patient_embds = self.base_model.cell2patient_aggregation.aggregate(data = cell_cross_embds, mask = padding_mask)
        patient_embds = self.base_model.patient_encoder(patient_embds)
        patient_preds = self.base_model.patient_predictor(patient_embds)
        #breakpoint()
        #healty 4, covid 2
        return (patient_preds[:,0,2] - patient_preds[:,0,4]).unsqueeze(-1)
    

def load_model_():
    # Run Name
    #run_name = "2025-02-09_22-25-15"
    #model_name = "epoch_002"
    run_name = "2025-03-20_01-45-22" # model with platelet study removed
    model_name = "epoch_001" # model with platelet study removed 
    config_path = f"/homefs/home/debroue1/projects/cellm/logs/train/runs/{run_name}/.hydra/"
    checkpoint_path = f"/homefs/home/debroue1/projects/cellm/logs/train/runs/{run_name}/checkpoints/{model_name}.ckpt"
    return load_model(config_path, checkpoint_path)
    with initialize_config_dir(version_base=None, config_dir=config_path, job_name="test_app"):
        cfg = compose(config_name="config.yaml", return_hydra_config=True, 
                    overrides=["data.multiprocessing_context=null", "data.batch_size=16","data.sampler_cls._target_=pascient.data.data_samplers.BaseSampler","+data.output_map.return_index=True"])#, "data.num_workers=0","data.persistent_workers=False"])
        print(OmegaConf.to_yaml(cfg))

    checkpoint = torch.load(checkpoint_path)
    metrics = hydra.utils.instantiate(cfg.get("metrics"))
    model = hydra.utils.instantiate(cfg.model, metrics = metrics)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    datamodule = hydra.utils.instantiate(cfg.data)

    cfg.paths.output_dir = ""
    trainer = hydra.utils.instantiate(cfg.trainer)

    return model, datamodule

def load_binary_model_():
    # Run Name
    run_name = "2025-02-11_19-52-59"
    model_name = "epoch_099" 
    config_path = f"/homefs/home/debroue1/projects/cellm/logs/train/runs/{run_name}/.hydra/"
    checkpoint_path = f"/homefs/home/debroue1/projects/cellm/logs/train/runs/{run_name}/checkpoints/{model_name}.ckpt"
    return load_binary_model(config_path, checkpoint_path)

    with initialize_config_dir(version_base=None, config_dir=config_path, job_name="test_app"):
        cfg = compose(config_name="config.yaml", return_hydra_config=True, 
                    overrides=["data.multiprocessing_context=null", "+data.output_map.return_index=True"])
        print(OmegaConf.to_yaml(cfg))

    checkpoint = torch.load(checkpoint_path)
    metrics = hydra.utils.instantiate(cfg.get("metrics"))
    model = hydra.utils.instantiate(cfg.model, metrics = metrics)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    datamodule = hydra.utils.instantiate(cfg.data)

    cfg.paths.output_dir = ""
    trainer = hydra.utils.instantiate(cfg.trainer)

    return model, datamodule

def main():
    """
    This script computes the integrated gradients for the model predictions on the validation set and stores them on the s3 bucket.
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--binary', action='store_true', help='For COVID Binary')
    parser.add_argument('--multilabel', action='store_true', help='For multi-label disease classification')
    parser.add_argument('--y_index', type=int, default=0, help='The index of the prediction vector to compute IG against') #In multilabel, COVID-19 is label 2
    parser.add_argument('--split', type=str, default = "val", help='what split of the data to use')
    parser.add_argument('--label_match', action='store_true', help='If true, only compute IG for samples with the label y_index')
    parser.add_argument('--include_healthy', action='store_true', help='If true, includes healthy patients as well')


    args = parser.parse_args()

    if args.binary:
        model, datamodule = load_binary_model_()
        model_tag = "binary"
        model_fwd = ForwardModel(model)
        reference = None
        y_index_mod = y_index
    elif args.multilabel:
        model, datamodule = load_model_()
        model_tag = "multilabel"
        model_fwd = ForwardDiffModel(model)
        y_index_mod = 0

        reference = None
        #reference = torch.load("/homefs/home/debroue1/projects/cellm/data/pascient/attributions/healthy_reference.pt")[None,None]
        #reference_pad = torch.load("/homefs/home/debroue1/projects/cellm/data/pascient/attributions/healthy_reference_mask.pt")

    else:
        raise ValueError("Please specify the model type (binary or multilabel)")

    if args.split == "train":
        dl = datamodule.train_dataloader()
    elif args.split == "val":
        dl = datamodule.val_dataloader()
    ig = IntegratedGradients(model_fwd)

    y_index = args.y_index # Disease index

    labels2int = datamodule.output_map.labels2int
    cell_type_map = labels2int["celltype_id"]
    cell_type_map = {v:k for k,v in cell_type_map.items()}

    disease_map = {v:k for k,v in labels2int["disease"].items()}
    tissue_map = {v:k for k,v in labels2int["tissue"].items()}

    if args.include_healthy:
        healthy_tag = "_with_healthy_"
    else:
        healthy_tag = ""
    
    print("Number of batches:")
    print(len(dl))
    for i_b, batch in tqdm.tqdm(enumerate(dl)):
        attr_batch = []
        x = batch.x
        padding_mask = batch.padded_mask
        y = batch.sample_metadata[model.prediction_labels[0]]
        if reference is not None:
            baseline = reference
        else:
            baseline = torch.zeros_like(x[[0]])

        if args.label_match:
            idx_evals = np.where(y==y_index)[0] #only evaluate for patients with label y == y_index
            label_match_prefix = ""
            
            if args.include_healthy:
                idx_evals = np.concatenate((idx_evals, np.where(y==4)[0])) # healthy is index 4
                
        else:
            idx_evals = range(x.shape[0])
            label_match_prefix = "all_"

        x.requires_grad = True
        for i in idx_evals:
            attr, delta  = ig.attribute(x[[i]], additional_forward_args=padding_mask[[i]], target = y_index_mod, return_convergence_delta=True, baselines = baseline)

            attr_non_padded = attr[0,0][padding_mask[[i]][0,0]].detach()
            #Store in pandas
            df_ = pd.DataFrame(attr_non_padded,columns = datamodule.gene_order)
            df_["study::::sample"] = batch.sample_metadata["study::::sample"][i].item()
            df_["disease"] = batch.sample_metadata["disease"][i].item()
            df_["tissue"] = batch.sample_metadata["tissue"][i].item()
            df_["cell_type"] = batch.cell_metadata.cell_level_labels["celltype_id"][i,0][padding_mask[[i]][0,0]]
            df_["cell_index"] = batch.cell_metadata.cell_level_labels["index"][i,0][padding_mask[[i]][0,0]]

            df_["cell_type"] = df_["cell_type"].map(cell_type_map)
            df_["disease"] = df_["disease"].map(disease_map)
            df_["tissue"] = df_["tissue"].map(tissue_map)

            attr_batch.append(df_)
    
        if len(attr_batch) == 0:
            continue
        attr_batch = pd.concat(attr_batch)
        # Serialize the tensor to a byte stream
        csv_buffer = StringIO()
        attr_batch.to_csv(csv_buffer, index=False)
        s3_resource = boto3.resource('s3')
        bucket_name = 'prescient-braid-data'
        s3_file_name = f'cellm/interpretation/IG_attributions/new_diff_{label_match_prefix}{healthy_tag}indexed_attributions_{model_tag}_{args.split}_batch_{i_b}_disease_{y_index}.csv'
        s3_resource.Object(bucket_name, s3_file_name).put(Body=csv_buffer.getvalue())
        print(f"Saved to the s3 bucket at {s3_file_name}")

if __name__ == "__main__":
    main()