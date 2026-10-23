"""Spark ETL: Garmin export + Clover period export -> tidy daily & run tables (parquet + csv).

Usage: python etl.py <garmin DI_CONNECT dir> <clover file> <output dir> [fit_hrv csv]

The optional fit_hrv csv (from extract_fit_hrv.py) fills nightly HRV from the raw watch
FIT files for nights the Health Status export doesn't cover (May–Sep 2025).
"""
import base64
import json
import sys
from datetime import date, timedelta

from pyspark.sql import SparkSession, functions as F, Window

GARMIN, CLOVER, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FIT_HRV = sys.argv[4] if len(sys.argv) > 4 else None
TZ = "Europe/London"
EXPORT_DATE = date(2026, 10, 18)

# Training phases (user-provided): consistent running from late Aug 2025,
# 10-week block before the 24 May 2026 marathon.
BASE_START = date(2025, 8, 25)
RACE = date(2026, 5, 24)
BLOCK_START = RACE - timedelta(weeks=10)

spark = (SparkSession.builder.master("local[*]").appName("garmin-sleep-study")
         .config("spark.sql.session.timeZone", "UTC").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
rd = spark.read.option("multiLine", True)

# ---------- Sleep ----------
sleep = (rd.json(f"{GARMIN}/DI-Connect-Wellness/*_sleepData.json")
         .where(F.col("calendarDate").isNotNull() & F.col("deepSleepSeconds").isNotNull())
         .select(
             F.to_date("calendarDate").alias("date"),
             F.from_utc_timestamp(F.to_timestamp("sleepStartTimestampGMT"), TZ).alias("bedtime"),
             F.from_utc_timestamp(F.to_timestamp("sleepEndTimestampGMT"), TZ).alias("waketime"),
             (F.col("deepSleepSeconds") / 3600).alias("deep_h"),
             (F.col("lightSleepSeconds") / 3600).alias("light_h"),
             (F.coalesce("remSleepSeconds", F.lit(0)) / 3600).alias("rem_h"),
             (F.col("awakeSleepSeconds") / 3600).alias("awake_h"),
             F.col("remSleepSeconds").isNotNull().alias("has_rem"),
             "awakeCount", "avgSleepStress", "averageRespiration", "restlessMomentCount",
             F.col("sleepScores.overallScore").alias("sleep_score"),
             F.col("sleepScores.qualityScore").alias("quality_score"),
             F.col("sleepScores.durationScore").alias("duration_score"),
             F.col("sleepScores.recoveryScore").alias("recovery_score"),
         )
         .withColumn("total_sleep_h", F.col("deep_h") + F.col("light_h") + F.col("rem_h"))
         .withColumn("deep_pct", 100 * F.col("deep_h") / F.col("total_sleep_h"))
         .withColumn("rem_pct", F.when(F.col("has_rem"), 100 * F.col("rem_h") / F.col("total_sleep_h")))
         # bedtime as hours after 18:00 so midnight-crossing bedtimes stay continuous
         .withColumn("bedtime_h", ((F.hour("bedtime") + F.minute("bedtime") / 60 - 18 + 24) % 24) + 18)
         .dropDuplicates(["date"]))

# ---------- Overnight HRV / RHR (health status) ----------
hs = (rd.json(f"{GARMIN}/DI-Connect-Wellness/*_healthStatusData.json")
      .select(F.to_date("calendarDate").alias("date"), F.explode("metrics").alias("m"))
      .where(F.col("m.type").isin("HRV", "HR") & F.col("m.value").isNotNull()))
hrv = (hs.groupBy("date").pivot("m.type", ["HRV", "HR"]).agg(F.first("m.value"))
       .withColumnRenamed("HRV", "hrv").withColumnRenamed("HR", "sleep_rhr")
       .join(hs.where("m.type = 'HRV'").select(
           "date", F.col("m.baselineLowerLimit").alias("hrv_base_lo"),
           F.col("m.baselineUpperLimit").alias("hrv_base_hi")), "date", "left"))

if FIT_HRV:
    # Raw FIT hrv_status_summary: timestamp is the morning the summary was written.
    fit = (spark.read.option("header", True).option("inferSchema", True).csv(FIT_HRV)
           .where(F.col("last_night_average").isNotNull())
           .withColumn("date", F.to_date(F.from_utc_timestamp(F.to_timestamp("timestamp"), TZ)))
           .withColumn("rn", F.row_number().over(Window.partitionBy("date").orderBy(F.desc("timestamp"))))
           .where("rn = 1")
           .select("date", F.col("last_night_average").cast("double").alias("fit_hrv")))
    hrv = (hrv.join(fit, "date", "full")
           .withColumn("hrv_source", F.when(F.col("hrv").isNotNull(), "health_status").otherwise("fit"))
           .withColumn("hrv", F.coalesce("hrv", "fit_hrv")).drop("fit_hrv"))

# ---------- Daily summary: stress, body battery, RHR, steps ----------
uds = rd.json(f"{GARMIN}/DI-Connect-Aggregator/UDSFile_*.json")
stress = (uds.select(F.to_date("calendarDate").alias("date"),
                     F.explode("allDayStress.aggregatorList").alias("a"))
          .groupBy("date").pivot("a.type", ["TOTAL", "AWAKE", "ASLEEP"])
          .agg(F.first("a.averageStressLevel"))
          .select("date", F.col("TOTAL").alias("stress_day"), F.col("AWAKE").alias("stress_awake")))
bb = (uds.select(F.to_date("calendarDate").alias("date"),
                 F.col("bodyBattery.chargedValue").alias("bb_charged"),
                 F.col("bodyBattery.drainedValue").alias("bb_drained"),
                 F.explode_outer("bodyBattery.bodyBatteryStatList").alias("s"))
      .groupBy("date", "bb_charged", "bb_drained")
      .agg(F.max(F.when(F.col("s.bodyBatteryStatType") == "HIGHEST", F.col("s.statsValue"))).alias("bb_high"),
           F.max(F.when(F.col("s.bodyBatteryStatType") == "LOWEST", F.col("s.statsValue"))).alias("bb_low")))
uds_flat = (uds.select(F.to_date("calendarDate").alias("date"),
                       F.col("restingHeartRate").alias("rhr"), F.col("totalSteps").alias("steps"),
                       "vigorousIntensityMinutes", "moderateIntensityMinutes")
            .dropDuplicates(["date"]).join(stress, "date", "left").join(bb, "date", "left"))

# ---------- Menstrual cycles (Clover; obfuscated base64 with a 1-char salt per '~' chunk) ----------
parts = open(CLOVER).read().split("~")
b64 = parts[0] + "".join(p[1:] for p in parts[1:])
clover = json.loads(base64.b64decode(b64 + "=" * (-len(b64) % 4)))
cycle_rows = []
for i, c in enumerate(clover["cycle_items"]):
    start, pend, end = (date.fromisoformat(c[k]) for k in ("cycle_start", "period_end", "cycle_end"))
    complete = end <= EXPORT_DATE
    length = (end - start).days
    period_len = (pend - start).days + 1
    ovu = length - 14 if complete else None  # standard ~14-day luteal phase assumption
    for d in range(length):
        day = d + 1
        if day <= period_len:
            phase = "Menstrual"
        elif ovu is None:
            phase = None
        elif day < ovu - 1:
            phase = "Follicular"
        elif day <= ovu + 1:
            phase = "Ovulatory"
        else:
            phase = "Luteal"
        cycle_rows.append((start + timedelta(days=d), i, start, length if complete else None,
                           period_len, day, (day - (length + 1)) if complete else None, phase))
cycles = spark.createDataFrame(cycle_rows, "date date, cycle_id int, cycle_start date, cycle_len int, "
                                           "period_len int, cycle_day int, days_to_next_period int, phase string")

# ---------- Runs ----------
acts = (rd.json(f"{GARMIN}/DI-Connect-Fitness/*_summarizedActivities.json")
        .select(F.explode("summarizedActivitiesExport").alias("a")).select("a.*")
        .where(F.col("activityType").isin("running", "treadmill_running", "trail_running")))
runs = (acts.select(
            "activityId", "activityType",
            # startTimeLocal is local wall-clock time encoded as epoch ms
            F.to_date(F.timestamp_millis(F.col("startTimeLocal").cast("long"))).alias("date"),
            F.hour(F.timestamp_millis(F.col("startTimeLocal").cast("long"))).alias("start_hour"),
            (F.col("distance") / 100000).alias("distance_km"),
            (F.col("movingDuration") / 60000).alias("moving_min"),
            F.col("avgHr").alias("avg_hr"), F.col("maxHr").alias("max_hr"),
            F.col("workoutRpe").alias("rpe"), F.col("workoutFeel").alias("feel"),
            F.col("aerobicTrainingEffect").alias("aerobic_te"),
            F.col("vO2MaxValue").alias("vo2max"), F.col("avgRunCadence").alias("cadence"),
            F.col("elevationGain").alias("elev_gain_cm"))
        .where("distance_km >= 1")
        .withColumn("pace_min_km", F.col("moving_min") / F.col("distance_km"))
        .withColumn("week", F.date_trunc("week", "date").cast("date"))
        .withColumn("rank_in_week", F.row_number().over(
            Window.partitionBy("week").orderBy(F.desc("distance_km"))))
        .withColumn("is_long_run", (F.col("rank_in_week") == 1) & (F.col("distance_km") >= 10))
        .withColumn("is_race", F.col("date") == F.lit(RACE))
        .withColumn("training_phase",
                    F.when(F.col("date") < F.lit(BASE_START), "Pre-training")
                     .when(F.col("date") < F.lit(BLOCK_START), "Base build")
                     .when(F.col("date") <= F.lit(RACE), "10-week block")
                     .otherwise("Post-race"))
        .drop("rank_in_week"))

# ---------- Daily master table ----------
# Garmin's sleep calendarDate is the wake-up date, so daytime stress of the previous
# day precedes that night's sleep.
prev_day = uds_flat.select(F.date_add("date", 1).alias("date"),
                           F.col("stress_awake").alias("stress_prev_day"),
                           F.col("steps").alias("steps_prev_day"))
run_load = runs.groupBy("date").agg(F.sum("distance_km").alias("run_km"))
prev_run = run_load.select(F.date_add("date", 1).alias("date"), F.col("run_km").alias("run_km_prev_day"))

daily = (sleep.join(hrv, "date", "full").join(uds_flat, "date", "full")
         .join(prev_day, "date", "left").join(prev_run, "date", "left")
         .join(cycles, "date", "left")
         .withColumn("run_km_prev_day", F.coalesce("run_km_prev_day", F.lit(0.0)))
         .withColumn("weekday", F.date_format("date", "E"))
         .withColumn("is_weekend_night", F.dayofweek("date").isin(1, 7))  # Sat/Sun wake = Fri/Sat night
         .withColumn("month", F.date_trunc("month", "date").cast("date"))
         .where(F.col("date") <= F.lit(date(2026, 9, 23)))
         .orderBy("date"))

runs = runs.join(daily.select("date", "sleep_score", "total_sleep_h", "hrv", "sleep_rhr",
                              "stress_prev_day", "phase", "cycle_day"), "date", "left")

for name, df in [("daily", daily), ("runs", runs.orderBy("date")), ("cycles", cycles)]:
    df.write.mode("overwrite").parquet(f"{OUT}/{name}.parquet")
    df.toPandas().to_csv(f"{OUT}/{name}.csv", index=False)
    print(name, df.count(), "rows")

spark.stop()
