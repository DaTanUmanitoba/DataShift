import get_sub_dataset as gd

dt = gd.get_data(
    dataset= 'civilcomments',
    #version= '','2.1',
    root_dir="C:/work/internship/data/",
    #root_dir="D:/distributionShiftAmazonData/data/civilComments/",
    split_scheme="val",
    out_dir="C:/work/internship/data/splits",  #specify if the subset is going to be written in a folder
    frac=0.1, 
    download=False, 
    data_frame=True
)

print("The reuturned dataframe of the subset has shape: ")
print(dt.shape)