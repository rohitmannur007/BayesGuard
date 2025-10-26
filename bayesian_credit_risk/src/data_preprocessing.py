def load_data():
    import os
    # Anchor to project root (../ from src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'german.data')
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    print(f"Project root: {project_root}")
    print(f"Data path: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Local file not found at {data_path}. Falling back to URL...")
        df = pd.read_csv(url, sep=r'\s+', header=None)
    else:
        print(f"Local file found at {data_path}.")
        df = pd.read_csv(data_path, sep=r'\s+', header=None)
    
    column_names = [
        'status_account',           # A1: qualitative
        'duration_month',           # A2: numerical
        'credit_history',           # A3: qualitative
        'purpose',                  # A4: qualitative (for hierarchical groups)
        'credit_amount',            # A5: numerical
        'savings_account',          # A6: qualitative
        'present_employment_since', # A7: qualitative
        'installment_rate_percent', # A8: numerical
        'personal_status_sex',      # A9: qualitative
        'other_debtors_guarantors', # A10: qualitative
        'present_residence_since',  # A11: numerical
        'property',                 # A12: qualitative
        'age_years',                # A13: numerical
        'other_installment_plans',  # A14: qualitative
        'housing',                  # A15: qualitative
        'num_existing_credits_this_bank', # A16: numerical
        'job',                      # A17: qualitative
        'num_people_liable_maintenance', # A18: numerical
        'telephone',                # A19: qualitative
        'foreign_worker',           # A20: qualitative
        'default'                   # Target: 1=good, 2=bad
    ]
    
    df.columns = column_names
    
    # Remap target: 0=good (was 1), 1=bad (was 2)
    df['default'] = (df['default'] == 2).astype(int)
    
    # Quick validation
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"
    print(f"Loaded {len(df)} samples. Default rate: {df['default'].mean():.1%} bad credits")
    
    return df