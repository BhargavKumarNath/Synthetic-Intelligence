import pandera as pa
from pandera.typing import DataFrame, Series
import pandas as pd

class BaseDataSchema(pa.DataFrameModel):
    target: Series[int] = pa.Field(isin=[0, 1])
    
    class Config:
        strict = False  # Allows dynamically generated columns (e.g. num_*, cat_*)
        coerce = True

def validate_dataset(df: pd.DataFrame, num_features: int, cat_features: int):
    """
    Dynamically validate dataset schema based on the config.
    """
    schema_dict = {
        "target": pa.Column(int, pa.Check.isin([0, 1]))
    }
    
    # Add numerical columns
    for i in range(num_features):
        schema_dict[f"num_{i}"] = pa.Column(float)
        
    # Add categorical columns
    for i in range(cat_features):
        schema_dict[f"cat_{i}"] = pa.Column("category")
        
    schema = pa.DataFrameSchema(schema_dict, strict=True, coerce=True)
    return schema.validate(df)
