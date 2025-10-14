import pandas as pd

def test_code(filepath1, filepath2):
    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)
    # Take the sum of the 'Value' column in both dataframes
    sum1 = df1['Value'].sum()
    sum2 = df2['Value'].sum()
    assert sum1 == sum2, f"Sums do not match: {sum1} != {sum2}"
    # Take the number of rows in both dataframes
    len1 = len(df1)
    len2 = len(df2)
    # Take the number of non-zero values in both dataframes
    nonzero1 = (df1['Value'] != 0).sum()
    nonzero2 = (df2['Value'] != 0).sum()
    assert nonzero1 == nonzero2, f"Non-zero counts do not match: {nonzero1} != {nonzero2}"
    print(nonzero1, nonzero2)
    # Take the number of zero values in both dataframes
    zero1 = (df1['Value'] == 0).sum()
    zero2 = (df2['Value'] == 0).sum()
    assert len1 == len2+zero1, f"Lengths do not match: {len1} != {len2}+{zero2}"