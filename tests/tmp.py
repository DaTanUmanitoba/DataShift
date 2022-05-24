import get_sub_dataset as gd

dt = gd.get_data(
    dataset='amazon',# 'civilcomments',
    version='2.1',
    root_dir="D:/distributionShiftAmazonData/data/amazon/",
    #root_dir="D:/distributionShiftAmazonData/data/civilComments/",
    split_scheme="val",
    out_dir="D:/distributionShiftAmazonData/data/amazon/", 
    frac=0.01, 
    download=False, 
    data_frame=True
)

print(dt.shape)