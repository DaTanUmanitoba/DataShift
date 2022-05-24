from wilds import get_dataset
import os
import numpy as np
import pandas as pd
from datatable import dt

def get_data(
    dataset, version, root_dir, split_scheme, out_dir, frac=1.0, download=False, data_frame=False
):
    # Load the full dataset, and download it if necessary
    full_dataset = get_dataset(
                        dataset= dataset,#"civilcomments", 
                        version= version,#'1.0',
                        download= download, #False,
                        root_dir = root_dir,# "D:/distributionShiftAmazonData/data/civilComments/"
                    )

    # get the data
    np.random.seed(1)
    required_data = full_dataset.get_subset(
        split=split_scheme,
        frac=frac,
        transform=None,
    )

    #get the split indices and input raw file name
    split_indices = required_data.indices
    input_dir = root_dir+"/"+dataset+"_v"+version
    if dataset == 'amazon':
        input_dir += "/reviews.csv"
    elif dataset == 'civilcomments':
        input_dir += "/all_data_with_identities.csv"

    print("The input file is %s" % input_dir)
    if os.path.isfile(input_dir):
        df = dt.fread(input_dir)
        df = df[split_indices, :]
    else:
        raise FileNotFoundError

    #output to csv or return dataframe
    if data_frame:
        return df
    else:
        if not os.path.isdir(out_dir):
            raise FileNotFoundError
        else:
            out_file_name = out_dir+"/split_"+split_scheme+".csv"
            print("The output file is: "+out_file_name)
            df.to_csv(out_file_name)

