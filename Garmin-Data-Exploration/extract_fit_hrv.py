"""Extract nightly HRV summaries (FIT message hrv_status_summary) from raw Garmin FIT files.

Usage: python extract_fit_hrv.py <fit dir> <output csv>
"""
import glob, sys
import fitdecode
import pandas as pd

rows = []
for f in glob.glob(f"{sys.argv[1]}/*.fit"):
    with fitdecode.FitReader(f, check_crc=fitdecode.CrcCheck.DISABLED) as fr:
        for fm in fr:
            if isinstance(fm, fitdecode.FitDataMessage) and fm.name == "hrv_status_summary":
                rows.append({fld.name: fld.value for fld in fm.fields})
df = pd.DataFrame(rows)
print(df.columns.tolist()); print(df.head(3).to_string())
df.to_csv(sys.argv[2], index=False)
