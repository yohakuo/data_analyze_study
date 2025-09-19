import pymysql  # <--- 1. 更换为 PyMySQL
import paho.mqtt.client as mqtt
import time
import json
import logging

# --- 日志配置，比 print 更专业 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 配置信息 ---
# MySQL 数据库配置
MYSQL_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'dbeaver_user',
    'password': 'study2025',
    'database': 'cave45',
    'charset': 'utf8mb4'
}

# EMQX Broker 配置
EMQX_CONFIG = {
    'host': 'localhost',
    'port': 1883,
    'topic': 'sensor/cave45',
    'username': 'admin',
    'password': 'study2025'
}

def on_connect(client, userdata, flags, rc):
    """MQTT 连接成功时的回调函数"""
    if rc == 0:
        logging.info("成功连接到 EMQX Broker！")
    else:
        logging.error(f"连接 EMQX 失败，返回码: {rc}")

def main():
    # 1. 初始化并连接 MQTT
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.username_pw_set(EMQX_CONFIG['username'], EMQX_CONFIG['password']) # <--- 2. 添加用户名和密码认证

    # 设置遗言消息 (LWT - Last Will and Testament)
    lwt_topic = 'sensor/cave45/status'
    lwt_payload = 'offline'
    mqtt_client.will_set(lwt_topic, payload=lwt_payload, qos=1, retain=True)

    try:
        mqtt_client.connect(EMQX_CONFIG['host'], EMQX_CONFIG['port'], 60)
        mqtt_client.loop_start()  # 启动网络循环
    except Exception as e:
        logging.error(f"无法连接到 EMQX: {e}")
        return

    # 2. 连接数据库并发送数据
    try:
        logging.info("正在使用 PyMySQL 连接数据库...")
        # 使用 'with' 语句确保连接和游标能被自动关闭
        with pymysql.connect(**MYSQL_CONFIG) as db_conn:
            logging.info("PyMySQL 连接成功！")
            with db_conn.cursor(pymysql.cursors.DictCursor) as cursor: # 使用 DictCursor可以直接得到字典格式的结果
                
                # <--- 3. 修正 SQL 查询中的表名
                cursor.execute("SELECT id, reading_time, air_temperature, air_humidity FROM cave45_201A ORDER BY reading_time ASC")
                all_readings = cursor.fetchall()

                total_rows = len(all_readings)
                if total_rows == 0:
                    logging.warning("数据库中没有数据可供发送。")
                    return
                
                logging.info(f"从数据库中读取到 {total_rows} 条记录，准备开始模拟发送...")
                
                # 发送一条在线消息
                mqtt_client.publish(lwt_topic, payload='online', qos=1, retain=True)

                for row in all_readings:
                    # 准备要发送的 payload
                    payload_data = {
                        "id": row['id'],
                        "air_temperature": float(row['air_temperature']),
                        "air_humidity": float(row['air_humidity']),
                        "reading_time": row['reading_time'].isoformat() # isoformat() 已经是标准的UTC格式
                    }
                    payload_json = json.dumps(payload_data)

                    # 发布 MQTT 消息
                    mqtt_client.publish(EMQX_CONFIG['topic'], payload_json, qos=1)
                    
                    logging.info(f" > 已发送: {payload_json}")

                    # 暂停 0.1 秒，模拟真实的数据间隔
                    time.sleep(0.1)

    except pymysql.Error as err: # <--- 4. 捕获 PyMySQL 的错误
        logging.error(f"数据库操作错误: {err}")
    except Exception as e:
        logging.error(f"发生未知错误: {e}")
    finally:
        # 清理工作
        logging.info("--- 模拟结束，关闭连接 ---")
        # 遗言消息会在断开连接时自动由Broker发送，这里可以选择不手动发送 offline
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

if __name__ == '__main__':
    main()