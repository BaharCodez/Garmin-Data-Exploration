"""Scan raw Garmin FIT files: file types, date range, and any HRV messages."""
import collections, glob, sys
import fitdecode

types = collections.Counter(); hrv_msgs = collections.Counter(); months = collections.defaultdict(collections.Counter)
for f in glob.glob(f"{sys.argv[1]}/*.fit"):
    try:
        with fitdecode.FitReader(f, check_crc=fitdecode.CrcCheck.DISABLED) as fr:
            ftype = None
            for fm in fr:
                if not isinstance(fm, fitdecode.FitDataMessage):
                    continue
                if fm.name == "file_id":
                    ftype = fm.get_value("type", fallback=None); tc = fm.get_value("time_created", fallback=None)
                    types[ftype] += 1
                    if tc: months[str(tc)[:7]][ftype] += 1
                if "hrv" in fm.name.lower() or fm.global_mesg_num in (370, 371, 275):
                    hrv_msgs[(fm.name, ftype)] += 1
    except Exception as e:
        types["error"] += 1
print(types); print(hrv_msgs)
for m in sorted(months): print(m, dict(months[m]))
