import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load datasets (you must maintain these)
reference_data = pd.read_csv("data/train_data.csv")
current_data = pd.read_csv("data/production_data.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

report.save_html("reports/drift_report.html")

print("Drift report generated!")