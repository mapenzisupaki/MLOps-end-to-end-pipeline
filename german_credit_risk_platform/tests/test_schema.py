from credit_risk.data import load_raw_data, prepare_modeling_table


def test_expected_columns_present():
    df = load_raw_data()
    expected = {"Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Credit amount", "Duration", "Purpose", "Risk"}
    assert expected.issubset(df.columns)


def test_target_is_binary_after_preparation():
    df = load_raw_data()
    _, y = prepare_modeling_table(df)
    assert set(y.unique()) == {0, 1}
