import pymysql
import paho.mqtt.client as mqtt
import time
import logging

# --- 配置区 (保持不变) ---
MYSQL_CONFIG = {
    'host': '127.0.0.1', 'port': 3306, 'user': 'dbeaver_user', 
    'password': 'study2025', 'database': 'cave45', 'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}
EMQX_CONFIG = {
    'host': 'host.docker.internal', 'port': 1883, 'topic': 'sensor/cave45',
    'username': 'admin', 'password': 'study2025'
}
BATCH_SIZE = 5000 # <-- 设置批处理大小，每 5000 条数据发送一次

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def publish_batch(client, topic, batch):
    """用于发送一个批次的数据"""
    if not batch:
        return
    # 使用换行符 \n 将多行数据拼接成一个大的 payload
    payload = "\n".join(batch)
    client.publish(topic, payload, qos=0)
    logging.info(f"成功发送一个批次，包含 {len(batch)} 条数据。")
    batch.clear()

def main():
    # --- MQTT 连接部分 (保持不变) ---
    mqtt_client = mqtt.Client()
    mqtt_client.username_pw_set(EMQX_CONFIG['username'], EMQX_CONFIG['password'])
    try:
        mqtt_client.connect(EMQX_CONFIG['host'], EMQX_CONFIG['port'], 60)
        mqtt_client.loop_start() 
        logging.info("✅ 成功连接到 EMQX Broker。")
    except Exception as e:
        logging.error(f"❌ 无法连接到 EMQX: {e}"); return

    # --- 数据处理和发送部分 ---
    batch = [] 
    try:
        with pymysql.connect(**MYSQL_CONFIG) as db_conn:
            with db_conn.cursor() as cursor: # <-- 变量名是 cursor
                
                # =======================================================
                # ### 错误修正 ###
                # =======================================================
                query = "SELECT reading_time, air_temperature, air_humidity FROM cave45_201A ORDER BY id ASC"
                cursor.execute(query) # <-- 这里之前错误地写成了 db_cursor
                # =======================================================
                
                all_rows = cursor.fetchall()
                total_rows = len(all_rows)
                logging.info(f"从数据库读取到 {total_rows} 条记录，准备分批发送...")

                for i, row in enumerate(all_rows):
                    try:
                        reading_time_obj = row.get('reading_time')
                        temp = row.get('air_temperature')
                        humidity = row.get('air_humidity')
                        if not all([reading_time_obj, temp, humidity]):
                            continue
                        
                        timestamp_ns = int(reading_time_obj.timestamp() * 1_000_000_000)
                        # 使用你之前能成功写入的表名，比如 adata45
                        line_protocol = f"adata45,source=python_batch air_temperature={float(temp)},air_humidity={float(humidity)} {timestamp_ns}"
                        
                        batch.append(line_protocol)

                        if len(batch) >= BATCH_SIZE:
                            publish_batch(mqtt_client, EMQX_CONFIG['topic'], batch)
                            time.sleep(0.5)

                    except Exception as e:
                        logging.error(f"处理行时出错: {e} | 数据: {row}")
                        continue
                
                publish_batch(mqtt_client, EMQX_CONFIG['topic'], batch)

    except Exception as e:
        logging.error(f"发生严重错误: {e}")
    finally:
        logging.info(f"--- 所有批次发送完成，关闭连接... ---")
        time.sleep(2)
        mqtt_client.loop_stop(); mqtt_client.disconnect()

if __name__ == '__main__':
    main()