from credit_risk.data import load_raw_data, prepare_modeling_table
from credit_risk.features import build_preprocessor


def test_account_missing_values_are_explicit_category():
    df = load_raw_data()
    X, _ = prepare_modeling_table(df)
    assert X["Saving accounts"].isna().sum() == 0
    assert X["Checking account"].isna().sum() == 0


def test_preprocessor_fits_after_train_split_inputs():
    df = load_raw_data()
    X, _ = prepare_modeling_table(df)
    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X.head(20))
    assert transformed.shape[0] == 20
