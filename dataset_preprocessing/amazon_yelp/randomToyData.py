from datatable import dt, f, by, update, g
import argparse
import numpy as np

#when frac = 0.01, the toy file is 30M with 110000+ records
def main(path, frac, out):
    df = dt.fread(path) #("final_combined_vec_itr0.csv")
    # Fix the seed for reproducibility
    np.random.seed(0)
    subsampled_count = int(frac * df.nrows )
    index = np.random.choice(range(df.nrows), subsampled_count, replace=False)
    df2 = df[index,:]
    #print(df2.nrows)
    df2.to_csv(out) # (final_combined_toy.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ToySample the Amazon dataset.")
    parser.add_argument(
        "--path",
        type=str,
        help="Path (full name) to the Amazon dataset",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Path (full name) to the toy Amazon dataset output",
    )
    parser.add_argument(
        "--frac",
        type=float,
        help="Toysample fraction",
    )

    args = parser.parse_args()
    main(args.path, args.frac, args.out)
