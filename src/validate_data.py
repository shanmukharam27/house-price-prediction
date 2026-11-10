import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

def get_housing_schema():
    """Defines the strict schema specification for the Ames House Price dataset."""
    return pa.DataFrameSchema({
        "Id": Column(pa.Int, Check.ge(1)),
        "MSSubClass": Column(pa.Int),
        "LotArea": Column(pa.Int, Check.ge(0)),
        "OverallQual": Column(pa.Int, Check.in_range(1, 10)),
        "OverallCond": Column(pa.Int, Check.in_range(1, 10)),
        "YearBuilt": Column(pa.Int, Check.in_range(1800, 2030)),
        "YearRemodAdd": Column(pa.Int, Check.in_range(1800, 2030)),
        "GrLivArea": Column(pa.Int, Check.ge(0)),
        "FullBath": Column(pa.Int, Check.ge(0)),
        "HalfBath": Column(pa.Int, Check.ge(0)),
        "BedroomAbvGr": Column(pa.Int, Check.ge(0)),
        "KitchenAbvGr": Column(pa.Int, Check.ge(0)),
        "TotRmsAbvGrd": Column(pa.Int, Check.ge(0)),
        "Fireplaces": Column(pa.Int, Check.ge(0)),
        "YrSold": Column(pa.Int, Check.in_range(2000, 2030)),
        "SalePrice": Column(pa.Int, Check.ge(0), nullable=True)
    }, strict=False)

def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates the input DataFrame against the defined schema."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_housing_schema()
    
    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")
        
        failures = err.failure_cases[['schema_context', 'column', 'check', 'failure_case', 'index']]
        print(failures.to_string())
        
        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False

if __name__ == "__main__":
    data_path = 'data/raw/train.csv'
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        validate_schema(raw_df, output_report_name="baseline_validation.csv")
    else:
        print(f"[ERROR] Target file not found at: {data_path}")
