import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("CarRentalDataV1.csv")
profile = ProfileReport(df, minimal=True, title="Car Rental Dataset - Baseline Profile")
profile.to_file("reports/baseline_profile.html")

desc = profile.get_description()
print(f"\n{len(desc.alerts)} alerts found:\n")
for alert in desc.alerts:
    print(alert)