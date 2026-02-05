# ClickHouse 数据库结构文档

> 导出时间: 2026-01-06 13:44:39

# 数据库: `feature_data`

## 表: `sensor_feature_data`

```sql
CREATE TABLE feature_data.sensor_feature_data
(
    `temple_id` String,
    `device_id` String,
    `stats_start_time` DateTime,
    `monitored_variable` String,
    `stats_cycle` String,
    `feature_key` String,
    `feature_value` String,
    `created_at` DateTime
)
ENGINE = MergeTree
ORDER BY temple_id
SETTINGS index_granularity = 8192
```

---

## 表: `test`

```sql
CREATE TABLE feature_data.test
(
    `temple_id` String,
    `device_id` String,
    `stats_start_time` DateTime,
    `monitored_variable` String,
    `stats_cycle` String,
    `feature_key` String,
    `feature_value` String,
    `created_at` DateTime
)
ENGINE = MergeTree
ORDER BY temple_id
SETTINGS index_granularity = 8192
```

---

# 数据库: `original_data`

## 表: `YC`

```sql
CREATE TABLE original_data.YC
(
    `temple_id` String,
    `device_id` String,
    `time` DateTime,
    `humidity` Float32
)
ENGINE = MergeTree
ORDER BY time
SETTINGS index_granularity = 8192
```

---

## 表: `battery_data`

```sql
CREATE TABLE original_data.battery_data
(
    `UUID` String,
    `device_id` String,
    `time` DateTime,
    `battery_level` Float32,
    `created_at` DateTime
)
ENGINE = MergeTree
ORDER BY time
SETTINGS index_granularity = 8192
```

---

## 表: `cave_entrance`

```sql
CREATE TABLE original_data.cave_entrance
(
    `time` DateTime,
    `min_battery_voltage(10min)` Float32,
    `avg_vapor_pressure` Float32,
    `total_radiation` Float32,
    `relative_humidity` Float32,
    `instantaneous_wd` Float32,
    `max_ws(10min)` Float32,
    `wd_max_ws` Float32,
    `total_rainfall(10min)` Float32,
    `air_temperature` Float32,
    `avg_surface_temperature` Float32,
    `avg_horizontal_ws(200cm)` Float32,
    `unit_vector_avg_wd` Float32,
    `wd_standard_deviation` Float32,
    `created_at` DateTime
)
ENGINE = MergeTree
ORDER BY time
SETTINGS index_granularity = 8192
```

---

## 表: `mogao_tourist_flow_9_13_to_9_19`

```sql
CREATE TABLE original_data.mogao_tourist_flow_9_13_to_9_19
(
    `time` DateTime,
    `real_time_headcount` Int32,
    `same_day_reception` Int32,
    `past_seven_days_reception` Int32,
    `dwell_time_range` String,
    `temple_id` String,
    `created_at` DateTime
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(time)
ORDER BY (temple_id, time)
SETTINGS index_granularity = 8192
```

---

## 表: `mountain_peak`

```sql
CREATE TABLE original_data.mountain_peak
(
    `time` DateTime,
    `min_battery_voltage(10min)` Float32,
    `avg_vapor_pressure` Float32,
    `total_radiation` Float32,
    `relative_humidity` Float32,
    `radiant_exposure` Float32,
    `max_ws(10min)` Float32,
    `wd_max_ws` Float32,
    `soil_moisture_content(20cm)` Float32,
    `soil_moisture_content(50cm)` Float32,
    `avg_atmospheric_pressure` Float32,
    `total_rainfall(10min)` Float32,
    `air_temperature` Float32,
    `avg_surface_temperature` Float32,
    `soil_moisture_content(100cm)` Float32,
    `avg_horizontal_ws(200cm)` Float32,
    `unit_vector_avg_wd` Float32,
    `wd_standard_deviation` Float32,
    `created_at` DateTime
)
ENGINE = MergeTree
ORDER BY time
SETTINGS index_granularity = 8192
```

---

## 表: `sensor_co2`

```sql
CREATE TABLE original_data.sensor_co2
(
    `temple_id` String,
    `device_id` String,
    `time` DateTime,
    `co2_collected` Float32,
    `co2_corrected` Float32,
    `created_at` DateTime
)
ENGINE = MergeTree
ORDER BY time
SETTINGS index_granularity = 8192
```

---

## 表: `sensor_temp_humidity`

```sql
CREATE TABLE original_data.sensor_temp_humidity
(
    `temple_id` String,
    `device_id` String,
    `time` DateTime,
    `humidity` Float32,
    `temperature` Float32,
    `created_at` DateTime
)
ENGINE = MergeTree
ORDER BY time
SETTINGS index_granularity = 8192
```

---

## 表: `wireless_vw_sensor`

```sql
CREATE TABLE original_data.wireless_vw_sensor
(
    `temple_id` String,
    `device_id` String,
    `time` DateTime,
    `vibrating_wire_displacement` Float32,
    `vibrating_wire_temperature` Float32,
    `vibrating_wire_modulus` Float32,
    `created_at` DateTime
)
ENGINE = MergeTree
ORDER BY time
SETTINGS index_granularity = 8192
```

---

## 表: `zcp_checkpoint_g1`

```sql
CREATE TABLE original_data.zcp_checkpoint_g1
(
    `time` DateTime,
    `uninspected` Int32,
    `inspected` Int32,
    `digital_exhibition_center` Int32,
    `nine_storey_building_inspection` Int32,
    `small_archway_inspection` Int32,
    `en_route` Int32,
    `digital_exhibition_inspection` Int32,
    `total_ticket_sold` Int32,
    `created_at` DateTime
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(time)
ORDER BY time
SETTINGS index_granularity = 8192
```

---

# 数据库: `original_data_interpolated`

## 表: `sensor_temp_humidity_interpolated`

```sql
CREATE TABLE original_data_interpolated.sensor_temp_humidity_interpolated
(
    `time` DateTime,
    `humidity` Float64,
    `temperature` Float64,
    `device_id` String,
    `temple_id` String
)
ENGINE = MergeTree
ORDER BY time
SETTINGS index_granularity = 8192
```

---

