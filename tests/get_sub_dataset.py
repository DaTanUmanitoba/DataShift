from wilds import get_dataset
import os
import numpy as np
import pandas as pd
from datatable import dt

def get_data(dataset,  root_dir, split_scheme, version=None, out_dir=None, frac=1.0, download=False, data_frame=True):
    """
    Load the full dataset, and returns the appropriate WILDS dataset Dataframe or True.
    Input:
        dataset (str):     Name of the dataset (required)
        root_dir(str):     The directory to find the raw data or download the data if download is True (required)
        split_scheme(str): The split dataset that is accquired. (required)
                              For amazon dataset: 'train' 'val' 'test' 'id-val' 'id-test'
                              For civilcomments dataset: 'train' 'val' 'test'
        version (str):     Dataset version number, e.g., '1.0'.
                                Default = None: the latest version
        out_dir(str):      The direcotry to write the split subset
                                Default = None: not write to a file
        frac (float):      The fraction of the subset to randomly extract, for development purpose.
                                Default = 1.0: extract the whole subset
        download (bool):   Indicate whether the raw data needs to be downloaded to "root_dir"
                                Default = False: no need for download, the dataset is already in the "root_dir"
        data_frame(bool):  Indicate the output type: a dataframe or written to a specific file.
                                Default = True: Return the required datafrme of the subset
        dataset_kwargs:    Other keyword arguments to pass to the dataset constructors.
    Output:
        If data_frame is True: The specified WILDSDataset subset dataframe;
        If data_frame is False: Return True if the subset is successfully written to the specified directory.
    """
    full_dataset = get_dataset(
                        dataset= dataset,# i.e. "civilcomments", 
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
    input_dir = root_dir+"/"+dataset+"_v"+full_dataset.version
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
            raise NotADirectoryError
        else:
            out_file_name = out_dir+"/split_"+split_scheme+".csv"
            print("The output file is: "+out_file_name)
            df.to_csv(out_file_name)
            return True

