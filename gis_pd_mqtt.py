import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Broker信息
broker_address = "192.168.16.135"  # 替换为实际的Broker地址
broker_port = 1883  # 替换为实际的端口号



# 主题
topic = "pub1"  # 替换为实际主题

# 连接回调函数
def on_connect(client, userdata, flags, rc, properties):
    print(f"连接结果码: {rc}")
    client.subscribe(topic, qos=1)  # 订阅主题，QoS为1

# 消息接收回调函数
def on_message(client, userdata, msg):
    hex_message = msg.payload.hex() # 解码消息内容为十六进制字符串
    results = []
    for i in range(0, len(hex_message), 4):  # 每4个字符解析为一个16进制数
        if i + 4 <= len(hex_message):
            hex_value = hex_message[i:i+4]
            decimal_value = int(hex_value, 16)
            converted_value = decimal_value * 3.3 / 4096
            results.append(round(converted_value, 2))  # 保留两位小数
    meaningful_data = results[4:-1]  # 去掉前4个和最后一个数据后，剩下的数据共360个，为相位信息，可绘制PRPD图
    print(f"收到消息: {meaningful_data}，主题: {msg.topic}")
    plt.scatter(np.linspace(0, 360, len(meaningful_data)), meaningful_data)
    plt.xlabel("相位 (0~360)")
    plt.ylabel("数据值")
    plt.title("实时数据散点图")
    plt.pause(0.1)


# 创建MQTT客户端

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)


# 设置回调函数
client.on_connect = on_connect
client.on_message = on_message

# 连接到Broker
client.connect(broker_address, broker_port)

# 启动消息处理循环
rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体支持
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.ion()  # 开启实时绘图模式
client.loop_forever()
