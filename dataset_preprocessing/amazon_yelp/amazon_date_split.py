from datatable import dt, f, by, update, g
import argparse
import os, re
import numpy as np

def main(input, outdir):
    #read input and create "month" column
    dff = dt.fread(input) #("final_combined_toy.csv")
    month_list = np.array([re.sub('-.*$', '', x) for x in dff['date'].to_list()[0] ])
    dff['reviewMonth'] = month_list

    #the three columns to split the dataset
    splitCols = ["reviewYear", "reviewMonth", "category"]
    #outdir = "D:/distributionShiftAmazonData/splits/"
    splitDataRec(dff, splitCols, 0, outdir, "")

#split the original dataset and write the split file recursively
#paras: data, columns to split and folders to put the files
def splitDataRec(df, splitCols, index, dir, prefix ):
    if index >= len(splitCols):
        return
    col = splitCols[index]
    path = dir+col+"/"
    try:
        os.stat(path)
    except:
        os.mkdir(path)
    uniqs = set(np.array(df[col].to_list()[0]))
    for val in uniqs:
        new_prefix = prefix + "_" + col + str(val)
        file_name = path+"split"+new_prefix+".csv"
        subdf = df[f[col] == val,:]
        print("The output file and number of rows in the data: "+file_name+"; "+str(subdf.nrows))
        subdf.to_csv(file_name)
        new_index = index + 1
        splitDataRec(subdf, splitCols, new_index, dir, new_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split the dataset.")
    parser.add_argument(
        "--input",
        type=str,
        help="File path of the input data",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Directory that the split data will be written",
    )

    args = parser.parse_args()
    main(args.input, args.outdir)
