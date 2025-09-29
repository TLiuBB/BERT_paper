import os
import pandas as pd

def baseline_characteristics(df, name):
    import os
    import pandas as pd

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'left')

    os.makedirs("/Users/tonyliubb/not in iCloud/CPRD/BERT/saved/characteristics", exist_ok=True)
    log_file = open(f"/Users/tonyliubb/not in iCloud/CPRD/BERT/saved/characteristics/{name}", "w")

    continuous_vars = ['age', 'QRISK3_2017', 'SBP', 'SBP_sd', 'DBP', 'Total/HDL_ratio', 'townsend', 'BMI']
    categorical_vars = [v for v in ['ethnicity_num', 'smoking_num', 'risk_group'] if v in df.columns]
    binary_vars = ['family_history', 'AF', 'Erectile_dysfunction', 'HIV_AIDS', 'Migraine',
                   'Rheumatoid_arthritis', 'SLE', 'Severe_mental_illness',
                   'Diabetes_bin', 'CKD345',
                   'Antihypertensive', 'Antipsychotic', 'Corticosteroid',
                   'Hypertension', 'bp_treatment',
                   'CHD', 'MI', 'Stroke_ischaemic', 'Stroke_NOS', 'TIA',
                   'HF', 'PAD', 'AAA', 'Angina_stable', 'Angina_unstable',
                   'cvd_q', 'cvd_all',
                   'Dementia']

    # ==== Mapping dictionaries ====
    ethnicity_map = {
        0: "White",
        1: "Indian",
        2: "Pakistani",
        3: "Bangladeshi",
        4: "Other Asian",
        5: "Black Caribbean",
        6: "Black African",
        7: "Chinese",
        8: "Other"
    }
    smoking_map = {
        0: "Non-smoker",
        1: "Former smoker",
        2: "Light smoker",
        3: "Moderate smoker",
        4: "Heavy smoker"
    }

    print("\nContinuous Variables:", file=log_file)
    continuous_summary = df[continuous_vars].describe().transpose()
    continuous_summary['%recorded'] = 100 * continuous_summary['count'] / len(df)
    print(continuous_summary, file=log_file)
    complete_records = df[continuous_vars].dropna().shape[0]
    total_rows = df.shape[0]
    percentage = (complete_records / total_rows) * 100
    print(f'Percentage of rows with complete records for all continuous variables: {percentage:.3f}%', file=log_file)

    print("\nCategorical Variables:", file=log_file)
    for var in categorical_vars:
        na_percentage = df[var].isna().sum() / len(df) * 100
        print(f'\n{var} NA: {na_percentage:.3f}%', file=log_file)

        # Map if needed
        if var == 'ethnicity_num':
            mapped_counts = df[var].map(ethnicity_map).value_counts(normalize=True) * 100
        elif var == 'smoking_num':
            mapped_counts = df[var].map(smoking_map).value_counts(normalize=True) * 100
        else:
            mapped_counts = df[var].value_counts(normalize=True) * 100

        print(mapped_counts.sort_index(), file=log_file)

    print("\nBinary Variables:", file=log_file)
    binary_summary = pd.DataFrame(df[binary_vars].mean() * 100)
    binary_summary.columns = ['%']
    print(binary_summary, file=log_file)

    log_file.close()

file_paths = [
    "/Users/tonyliubb/not in iCloud/CPRD/BERT/data/original data/men_imputed.csv",
    "/Users/tonyliubb/not in iCloud/CPRD/BERT/data/original data/women_imputed.csv",
    "/Users/tonyliubb/not in iCloud/CPRD/BERT/data/original data/men_London_imputed.csv",
    "/Users/tonyliubb/not in iCloud/CPRD/BERT/data/original data/women_London_imputed.csv"
]

names = [
    "men.txt",
    "women.txt",
    "men_London.txt",
    "women_London.txt"
]

for path, name in zip(file_paths, names):
    df = pd.read_csv(path)

    baseline_characteristics(df, name)
    print(f"✅ Done: {name}")