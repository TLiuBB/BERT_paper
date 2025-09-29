from Bert_localnow import BERTMLP_run

# Define data path and column names
base_path = "/Users/tonyliubb/not in iCloud/CPRD/BERT"
# base_path = "/users/k20125149/CPRD"

con_features = ['age', 'QRISK3_2017', 'SBP', 'SBP_sd', 'DBP', 'Total/HDL_ratio', 'townsend', 'BMI']
# labels = ['cvd_q', 'cvd_all', 'CHD', 'Stroke_ischaemic', 'MI', 'HF','Angina_stable', 'Dementia']
labels = ['cvd_all']

# Loop through each label
for label in labels:
    print("=" * 50)
    print(f"Starting training for label: {label}")
    print("=" * 50)

    # Run Bert_semi_run for the current label
    BERTMLP_run(
        gender='men',
        path=base_path,
        features=con_features,
        label=label,
        batch=128,
        epoch=50,
        freeze=5,
        patience=5,
        search_trials=1,
        threshold_range=(0.0, 1, 0.001),
        f1_mode='f1'
    )

    # Indicate completion of the current label
    print("=" * 50)
    print(f"Finished training for label: {label}")
    print("=" * 50)

