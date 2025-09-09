import pymysql
import paho.mqtt.client as mqtt
import time
import json
import logging
from datetime import datetime

# --- 配置区 (请确认这里的 EMQX host 是正确的) ---
MYSQL_CONFIG = {
    'host': '127.0.0.1', 'port': 3306, 'user': 'dbeaver_user', 
    'password': 'study2025', 'database': 'cave45', 'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}
EMQX_CONFIG = {
    'host': 'host.docker.internal', 'port': 1883, 'topic': 'sensor/cave45',
    'username': 'admin', 'password': 'study2025'
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. 连接 MQTT
    mqtt_client = mqtt.Client(); mqtt_client.username_pw_set(EMQX_CONFIG['username'], EMQX_CONFIG['password'])
    try:
        mqtt_client.connect(EMQX_CONFIG['host'], EMQX_CONFIG['port'], 60); mqtt_client.loop_start() 
        logging.info("✅ 成功连接到 EMQX Broker。")
    except Exception as e:
        logging.error(f"❌ 无法连接到 EMQX: {e}"); return

    # 2. 连接数据库
    try:
        with pymysql.connect(**MYSQL_CONFIG) as db_conn:
            with db_conn.cursor() as cursor:
                cursor.execute("SELECT id, reading_time, air_temperature, air_humidity FROM cave45_201A ORDER BY id ASC")
                all_readings = cursor.fetchall()
                logging.info(f"从数据库读取到 {len(all_readings)} 条记录，准备发送 JSON...")

                for row in all_readings:
                    try:
                        reading_time_obj = row.get('reading_time')
                        temp = row.get('air_temperature')
                        humidity = row.get('air_humidity')
                        if not all([reading_time_obj, temp, humidity]):
                            continue
                        
                        # =======================================================
                        # ### 核心修改：生成包含“字符串”时间戳的 JSON ###
                        # =======================================================
                        timestamp_ns_str = str(int(reading_time_obj.timestamp() * 1_000_000_000))
                        
                        payload_to_send = {
                            "air_temperature": float(temp),
                            "air_humidity": float(humidity),
                            "timestamp_ns": timestamp_ns_str  # <--- 发送字符串格式的时间戳
                        }
                        # =======================================================

                        mqtt_client.publish(EMQX_CONFIG['topic'], json.dumps(payload_to_send), qos=0)
                        
                        # 为了不刷屏，可以注释掉下面这行
                        # logging.info(f" > 已发送: {json.dumps(payload_to_send)}")
                        time.sleep(0.01)

                    except Exception as e:
                        logging.error(f"处理行 {row.get('id')} 时出错: {e}")
                        continue
    except Exception as e:
        logging.error(f"发生严重错误: {e}")
    finally:
        logging.info("--- 发送完成，关闭连接 ---")
        time.sleep(2)
        mqtt_client.loop_stop(); mqtt_client.disconnect()

if __name__ == '__main__':
    main()