# Prediction Files

服务器产物中已经包含以下官方 test split 预测结果：

- `predictions/esol/test_predictions/test.predict.0.csv`
- `predictions/esol/test_predictions/test_metric.json`
- `predictions/bbbp/test_predictions/test.predict.0.csv`
- `predictions/bbbp/test_predictions/test_metric.json`
- `predictions/tox21/test_predictions/test.predict.0.csv`
- `predictions/tox21/test_predictions/test_metric.json`
- `predictions/qm9dft/test_predictions/test.predict.0.csv`
- `predictions/qm9dft/test_predictions/test_metric.json`

现在四个任务均已提供官方 test split prediction。若最后做统一消融，请优先使用这些 `test.predict.0.csv` 重新计算各模型指标；若做 5-fold CV 口径比较，则使用 `metrics/<task>/cv.data` 和 `metric.result`。
