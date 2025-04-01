import pandas as pd
import os 

from io import StringIO
import tqdm
import boto3
import io

import subprocess
import sys

import anndata
from scipy.sparse import csr_matrix


def grep_for_word(filepath, word="COVID"):
    """
    Returns True if 'word' is found in 'filepath'.
    Uses grep in a subprocess for speed.
    """
    # -q (quiet) makes grep exit with 0 if found, 1 if not found, with no output.
    cmd = ["grep", "-q", word, filepath]
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # grep exit code: 0 => found, 1 => not found, anything else => error
    if result.returncode == 0:
        return True
    elif result.returncode == 1:
        return False
    else:
        # If something else happened, print or raise an error
        print("Error running grep:", result.stderr, file=sys.stderr)
        raise RuntimeError("grep returned non-zero exit code: {}".format(result.returncode))



if __name__ == "__main__":
    """
    This script aggregates in a single dataframe all the attributions from COVID patients with blood and lung in the train and validation data.
    """
    ig_path = "/braid/cellm/interpretation/IG_attributions/"

    #class_type = "healthy_baseline_indexed_attributions_multilabel"
    class_type = "new_diff"
    target_id = 2

    an_covid = []

    for i_f, f in enumerate(os.listdir(ig_path)):
        if (class_type in f) and (f"_{target_id}.csv" in f):
            print(f)
            if grep_for_word(ig_path + f, word="COVID-19"):
                print("grep found")
                df = pd.read_csv(ig_path + f)
                print("loaded")
                # Only tissue lung or blood and COVID19 patients
                df = df.loc[(df["disease"].isin(["COVID-19","healthy"])) & (df["tissue"].isin(["blood", "lung"]))]

                gene_cols = [c for c in df.columns if c not in ['disease','tissue','cell_type','study::::sample','cell_index']]
                array = csr_matrix(df[gene_cols].values)

                adata_ = anndata.AnnData(array, 
                                 obs = df[['disease','tissue','cell_type','study::::sample','cell_index']],
                                 var = gene_cols)    
                an_covid.append(adata_)       

    ann_covid = anndata.concat(an_covid, axis=0, join = "outer", merge = "unique")
    ann_covid.var.columns = ['gene_symbol']
    ann_covid.var.set_index('gene_symbol', inplace=True)

    ann_covid.write_h5ad("/homefs/home/debroue1/projects/cellm/data/pascient/attributions/new_multilabel_diff_attributions_multilabel_covid_with_healthy.h5ad", compression="gzip")
    print("Saved to disk")