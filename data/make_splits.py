import pandas as pd
from pathlib import Path
from sklearn.preprocessing import train_test_split

def make_stratified_splits(working_csv, test_ratio, val_ratio, seed, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    working_df = pd.read_csv(working_csv)
    if (val_ratio+test_ratio>=1):
        raise ValueError("Invalid train-test-val split.")
    working_df['class_gen'] = working_df.generator+working_df.class_name

    train_val_df, test_df = train_test_split(working_df,test_size=test_ratio,stratify=working_df['class_gen'])
    train_df, val_df = train_test_split(train_val_df,test_size=val_ratio/(1-test_ratio),stratify=working_df['class_gen'])
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    print(f"Sanity checks:")
    for df in [train_df,val_df,test_df]:
        df.drop(columns=['class_gen'],inplace=True)
        print(f"\n{df['split'].iloc[0]} df: {df.shape[0]} rows")
        print(df.class_name.value_counts())
        print(df.generator.value_counts())
    
    train_df.to_csv(out_dir/'train.csv', index=False)
    val_df.to_csv(out_dir/'val.csv', index=False)
    test_df.to_csv(out_dir/'test.csv', index=False)
    

