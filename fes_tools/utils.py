import pandas as pd

def read_plumed_colvar(filename, keep_zero=True, nan_handling="exception"):
    """
    A simple script to read a PLUMED COLVAR file into a pandas dataframe
    
    Args:
        filename (str): path to desired COLVAR file
        keep_zero (bool, optional): Whether to keep rows where the time is 0.0, default = True
        nan_handling (str, optional): How to handle NaN values. May be "exception" or "drop". Defaults to exception.
    
    Raises:
        Exception: if nan_handling is not a valid choice.
    
    Returns:
        df (pd.Dataframe): Contains the contents of the plumed COLVAR file
    """

    with open(filename, "r") as f:
        columns = f.readline()

    columns = columns.rstrip("\n").split()

    df = pd.read_csv(filename, sep="\s+", comment="#", header=None)

    df.columns = columns[2:]
    
    # ensure no lines contain nan values. Deal with them appropriately.
    if df.isnull().values.any():
        
        if nan_handling.lower() == "exception":
            print( df[df.isnull().any(axis=1)] )
            raise Exception("COLVAR file {filename} contains nans: terminating")
        elif nan_handling.lower() == "drop":
            print("Dropping rows with any nan values...")
            df.dropna(axis=0, how="any", inplace=True)
        else:
            raise Exception("Not a valid option for nan_handling")
    
    # Removes the zero time rows which occur in Plumed file every run with CHARMM
    if keep_zero == False:
        df = df.loc[df["time"] != 0.0]

    # reset times to start at zero
    df["time"] = df["time"] - df["time"].min()
    
    # remove any duplicate times. Would only occur if a simulation started, then failed, then was restarted without backing up the
    # COLVAR file to before the failed run. This means you never have to worry about this failed simualtion issue
    # (except in the context of nan values due to incompletely written rows)
    df.drop_duplicates(subset="time", keep="last", inplace=True, ignore_index=True)

    return df


def read_plumed_fes(filename):
    """
    A simple script to read a PLUMED FES file into a pandas dataframe
    
    Args:
        filename (str): path to desired COLVAR file
        keep_zero (bool, optional): Whether to keep rows where the time is 0.0, default = True
        nan_handling (str, optional): How to handle NaN values. May be "exception" or "drop". Defaults to exception.
    
    Raises:
        Exception: if nan_handling is not a valid choice.
    
    Returns:
        df (pd.Dataframe): Contains the contents of the plumed COLVAR file
    """

    with open(filename, "r") as f:
        columns = f.readline()

    columns = columns.rstrip("\n").split()

    df = pd.read_csv(filename, sep="\s+", comment="#", header=None)

    df.columns = columns[2:]

    return df    
