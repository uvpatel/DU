from __future__ import annotations

import pandas as pd

from DU import evaluate, insights, load, summary, train



def test_load_csv(tmp_path):
	df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
	path = tmp_path / "sample.csv"
	df.to_csv(path, index=False)

	loaded = load(path)
	assert loaded.shape == (2, 2)
	assert list(loaded.columns) == ["a", "b"]


def test_summary_contains_expected_fields():
	df = pd.DataFrame(
		{
			"x": [1.0, 2.0, None],
			"y": [10, 20, 30],
			"label": ["a", "b", "a"],
		}
	)
	result = summary(df)
	assert set(result.keys()) == {
		"shape",
		"missing_values",
		"dtypes",
		"correlation_matrix",
	}
	assert result["shape"] == (3, 3)
	assert result["missing_values"]["x"] == 1


def test_insights_returns_human_readable_messages():
	df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
	msgs = insights(df, corr_threshold=0.8)
	assert any("Strong correlation" in m for m in msgs)


def test_train_and_evaluate_classification_flow():
	df = pd.DataFrame(
		{
			"age": [21, 22, 25, 40, 42, 39, 30, 29, 35, 45],
			"income": [30, 32, 35, 80, 78, 72, 50, 48, 60, 85],
			"segment": ["a", "a", "a", "b", "b", "b", "a", "a", "b", "b"],
			"target": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
		}
	)
	result = train(df, target="target", test_size=0.3, random_state=42)
	metrics = evaluate(result.model, result.X_test, result.y_test)

	assert result.task_type == "classification"
	assert "accuracy" in metrics