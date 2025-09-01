import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime

# --- 配置区 ---
EMQX_CONFIG = {
    'host': 'host.docker.internal', # 使用能连通的地址
    'port': 1883,
    'topic': 'sensor/cave45',
    'username': 'admin',
    'password': 'study2025'
}

# --- 要发送的单条测试数据 (已更新) ---
# 创建一个和主脚本格式一致的 payload，包含纳秒时间戳
now_dt = datetime.now()
timestamp_ns_str = str(int(now_dt.timestamp() * 1_000_000_000))

test_payload = {
    "id": 99999,  
    "air_temperature": 25.5,
    "air_humidity": 60.8,
    "timestamp_ns": timestamp_ns_str # <--- 关键修改：发送字符串
}


def main():
    """主函数，用于连接并发送单条消息"""
    client = mqtt.Client()
    client.username_pw_set(EMQX_CONFIG['username'], EMQX_CONFIG['password'])

    try:
        # 1. 连接到 EMQX Broker
        client.connect(EMQX_CONFIG['host'], EMQX_CONFIG['port'], 60)
        client.loop_start() 
        print(f"✅ 成功连接到 EMQX Broker ({EMQX_CONFIG['host']})。")

        # 2. 将数据字典转换为 JSON 字符串
        payload_json = json.dumps(test_payload)

        # 3. 发布单条消息
        result = client.publish(EMQX_CONFIG['topic'], payload_json, qos=1)
        result.wait_for_publish() # 等待消息发送回执

        if result.is_published():
            print(f"🚀 已成功发送单条测试消息到主题 '{EMQX_CONFIG['topic']}'")
            print(f"   内容: {payload_json}")
        else:
            print("❌ 消息发送失败！")

    except Exception as e:
        print(f"❌ 操作失败: {e}")
    finally:
        # 4. 断开连接
        time.sleep(1) 
        client.loop_stop()
        client.disconnect()
        print("🔌 已断开连接。")


if __name__ == '__main__':
    main()