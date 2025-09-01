import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime

# --- 配置区 ---
EMQX_CONFIG = {
    'host': 'localhost',
    'port': 1883,
    'topic': 'sensor/cave45',
    'username': 'admin',
    'password': 'study2025'
}

# --- 要发送的单条测试数据 ---
# 我们可以手动编一条数据，用一个特殊的ID方便在数据库里识别
test_payload = {
    "id": 99999,  
    "air_temperature": 25.5,
    "air_humidity": 60.8,
    "reading_time": datetime.now().isoformat() # 使用当前时间
}


def main():
    """主函数，用于连接并发送单条消息"""
    client = mqtt.Client()
    client.username_pw_set(EMQX_CONFIG['username'], EMQX_CONFIG['password'])

    try:
        # 1. 连接到 EMQX Broker
        client.connect(EMQX_CONFIG['host'], EMQX_CONFIG['port'], 60)
        client.loop_start() # 启动网络循环以确保回调和消息发送
        print("✅ 成功连接到 EMQX Broker。")

        # 2. 将数据字典转换为 JSON 字符串
        payload_json = json.dumps(test_payload)

        # 3. 发布单条消息
        # 使用 qos=1 保证消息至少送达一次
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
        time.sleep(1) # 等待一秒确保所有操作完成
        client.loop_stop()
        client.disconnect()
        print("🔌 已断开连接。")


if __name__ == '__main__':
    main()