import sys
import numpy as np
import paho.mqtt.client as mqtt
import matplotlib
# 在导入Figure前设置matplotlib使用PySide6后端
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                              QGroupBox, QGridLayout, QSpinBox, QComboBox, 
                              QStatusBar, QMessageBox)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, QMutex
from matplotlib import rcParams
import time
import queue

# 设置matplotlib中文支持
rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体支持
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 优化matplotlib性能
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000

class MplCanvas(FigureCanvas):
    """Matplotlib画布类，用于在Qt界面中嵌入matplotlib图形"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()
        # 设置动画效果
        self.blit = False
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.scatter = None
        self.line = None

class MQTTThread(QThread):
    """MQTT处理线程，避免阻塞主线程"""
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.running = True
        
    def run(self):
        while self.running:
            self.client.loop(0.1)  # 非阻塞处理MQTT消息
            time.sleep(0.01)  # 避免CPU占用过高
            
    def stop(self):
        self.running = False
        self.wait()

class MQTTClient(QWidget):
    """MQTT客户端类，处理MQTT连接和消息接收"""
    message_received = Signal(list)  # 信号：接收到新消息时发出
    connection_status = Signal(bool, str)  # 信号：连接状态变化时发出

    def __init__(self):
        super().__init__()
        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        
        self.broker_address = "192.168.16.135"
        self.broker_port = 1883
        self.topic = "pub1"
        self.connected = False
        self.mqtt_thread = None
        self.message_queue = queue.Queue(maxsize=10)  # 限制队列大小，避免内存溢出
        
        # 创建一个定时器来处理消息队列
        self.queue_timer = QTimer()
        self.queue_timer.timeout.connect(self.process_message_queue)
        self.queue_timer.start(50)  # 每50ms处理一次队列

    def connect_to_broker(self, broker_address, broker_port, topic):
        """连接到MQTT Broker"""
        self.broker_address = broker_address
        self.broker_port = int(broker_port)
        self.topic = topic
        
        try:
            self.client.connect(self.broker_address, self.broker_port)
            # 使用线程处理MQTT消息循环，避免阻塞主线程
            if self.mqtt_thread is None or not self.mqtt_thread.isRunning():
                self.mqtt_thread = MQTTThread(self.client)
                self.mqtt_thread.start()
            return True
        except Exception as e:
            self.connection_status.emit(False, f"连接失败: {str(e)}")
            return False

    def disconnect_from_broker(self):
        """断开与MQTT Broker的连接"""
        if self.connected:
            if self.mqtt_thread and self.mqtt_thread.isRunning():
                self.mqtt_thread.stop()
            self.client.disconnect()
            self.connected = False

    def on_connect(self, client, userdata, flags, rc, properties):
        """连接回调函数"""
        if rc == 0:
            self.connected = True
            client.subscribe(self.topic, qos=1)
            self.connection_status.emit(True, f"已连接到 {self.broker_address}:{self.broker_port}")
        else:
            self.connected = False
            self.connection_status.emit(False, f"连接失败，返回码: {rc}")

    def on_disconnect(self, client, userdata, rc):
        """断开连接回调函数"""
        self.connected = False
        self.connection_status.emit(False, "已断开连接")

    def process_message_queue(self):
        """处理消息队列"""
        if not self.message_queue.empty():
            try:
                data = self.message_queue.get_nowait()
                self.message_received.emit(data)
                self.message_queue.task_done()
            except queue.Empty:
                pass

    def on_message(self, client, userdata, msg):
        """消息接收回调函数"""
        try:
            hex_message = msg.payload.hex()  # 解码消息内容为十六进制字符串
            results = []
            for i in range(0, len(hex_message), 4):  # 每4个字符解析为一个16进制数
                if i + 4 <= len(hex_message):
                    hex_value = hex_message[i:i+4]
                    decimal_value = int(hex_value, 16)
                    converted_value = decimal_value * 3.3 / 4096
                    results.append(round(converted_value, 2))  # 保留两位小数
            
            meaningful_data = results[4:-1]  # 去掉前4个和最后一个数据
            
            # 将数据放入队列，而不是直接发送信号
            # 如果队列已满，则丢弃这条消息，避免处理积压
            try:
                self.message_queue.put_nowait(meaningful_data)
            except queue.Full:
                pass
                
        except Exception as e:
            print(f"消息处理错误: {str(e)}")

class MainWindow(QMainWindow):
    """主窗口类"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GIS局部放电在线监测系统")
        self.setMinimumSize(1000, 700)
        
        # 数据存储
        self.data_buffer = []
        self.max_buffer_size = 360  # 默认缓冲区大小
        self.data_mutex = QMutex()  # 用于线程安全访问数据
        self.last_update_time = time.time()
        self.update_interval = 0.2  # 控制更新频率，每0.2秒更新一次
        
        # 周期数据存储
        self.cycle_count = 1  # 当前周期计数
        self.max_cycles = 10  # 默认最大周期数
        self.accumulated_data = []  # 累积的数据
        
        # 创建MQTT客户端
        self.mqtt_client = MQTTClient()
        self.mqtt_client.message_received.connect(self.update_plot)
        self.mqtt_client.connection_status.connect(self.update_connection_status)
        
        # 创建界面
        self.setup_ui()
        
        # 创建定时器用于更新界面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)  # 每秒更新一次状态
        
        # 创建定时器用于限制绘图频率
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.redraw_plot)
        self.plot_timer.start(200)  # 每200ms重绘一次图表
        
        # 标记是否需要重绘
        self.need_redraw = False
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 创建连接设置组
        connection_group = QGroupBox("MQTT连接设置")
        connection_layout = QGridLayout()
        
        # 添加Broker地址设置
        connection_layout.addWidget(QLabel("Broker地址:"), 0, 0)
        self.broker_address_input = QLineEdit(self.mqtt_client.broker_address)
        connection_layout.addWidget(self.broker_address_input, 0, 1)
        
        # 添加Broker端口设置
        connection_layout.addWidget(QLabel("Broker端口:"), 0, 2)
        self.broker_port_input = QLineEdit(str(self.mqtt_client.broker_port))
        connection_layout.addWidget(self.broker_port_input, 0, 3)
        
        # 添加主题设置
        connection_layout.addWidget(QLabel("主题:"), 1, 0)
        self.topic_input = QLineEdit(self.mqtt_client.topic)
        connection_layout.addWidget(self.topic_input, 1, 1)
        
        # 添加连接按钮
        self.connect_button = QPushButton("连接")
        self.connect_button.clicked.connect(self.toggle_connection)
        connection_layout.addWidget(self.connect_button, 1, 3)
        
        connection_group.setLayout(connection_layout)
        main_layout.addWidget(connection_group)
        
        # 创建图表设置组
        chart_settings_group = QGroupBox("图表设置")
        chart_settings_layout = QGridLayout()
        
        # 添加图表类型选择
        chart_settings_layout.addWidget(QLabel("图表类型:"), 0, 0)
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["散点图", "线图"])  # 移除柱状图选项
        self.chart_type_combo.currentIndexChanged.connect(self.update_plot_type)
        chart_settings_layout.addWidget(self.chart_type_combo, 0, 1)
        
        # 添加数据缓冲区大小设置
        chart_settings_layout.addWidget(QLabel("数据缓冲区大小:"), 0, 2)
        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setRange(10, 1000)
        self.buffer_size_spin.setValue(self.max_buffer_size)
        self.buffer_size_spin.valueChanged.connect(self.update_buffer_size)
        chart_settings_layout.addWidget(self.buffer_size_spin, 0, 3)
        
        # 添加周期数设置
        chart_settings_layout.addWidget(QLabel("累积周期数:"), 1, 0)
        self.cycles_spin = QSpinBox()
        self.cycles_spin.setRange(1, 800)
        self.cycles_spin.setValue(self.max_cycles)
        self.cycles_spin.valueChanged.connect(self.update_max_cycles)
        chart_settings_layout.addWidget(self.cycles_spin, 1, 1)
        
        # 添加周期计数显示
        chart_settings_layout.addWidget(QLabel("当前周期:"), 1, 2)
        self.cycle_count_label = QLabel(f"{self.cycle_count}/{self.max_cycles}")
        chart_settings_layout.addWidget(self.cycle_count_label, 1, 3)
        
        # 添加清除数据按钮
        self.clear_button = QPushButton("清除数据")
        self.clear_button.clicked.connect(self.clear_data)
        chart_settings_layout.addWidget(self.clear_button, 0, 4)
        
        # 添加重置周期按钮
        self.reset_cycles_button = QPushButton("重置周期")
        self.reset_cycles_button.clicked.connect(self.reset_cycles)
        chart_settings_layout.addWidget(self.reset_cycles_button, 1, 4)
        
        chart_settings_group.setLayout(chart_settings_layout)
        main_layout.addWidget(chart_settings_group)
        
        # 创建matplotlib画布
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        main_layout.addWidget(self.canvas)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 添加连接状态标签
        self.connection_status_label = QLabel("未连接")
        self.status_bar.addWidget(self.connection_status_label)
        
        # 添加数据点数量标签
        self.data_count_label = QLabel("数据点: 0")
        self.status_bar.addPermanentWidget(self.data_count_label)
    
    def toggle_connection(self):
        """切换MQTT连接状态"""
        if not self.mqtt_client.connected:
            # 连接到Broker
            broker_address = self.broker_address_input.text()
            broker_port = self.broker_port_input.text()
            topic = self.topic_input.text()
            
            if self.mqtt_client.connect_to_broker(broker_address, broker_port, topic):
                self.connect_button.setText("断开")
        else:
            # 断开连接
            self.mqtt_client.disconnect_from_broker()
            self.connect_button.setText("连接")
    
    def update_connection_status(self, connected, message):
        """更新连接状态"""
        if connected:
            self.connection_status_label.setText(message)
            self.connection_status_label.setStyleSheet("color: green")
        else:
            self.connection_status_label.setText(message)
            self.connection_status_label.setStyleSheet("color: red")
            self.connect_button.setText("连接")
    
    def update_buffer_size(self, size):
        """更新数据缓冲区大小"""
        self.max_buffer_size = size
        self.data_mutex.lock()
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size:]
            self.need_redraw = True
        self.data_mutex.unlock()
    
    def update_max_cycles(self, cycles):
        """更新最大周期数"""
        self.max_cycles = cycles
        self.cycle_count_label.setText(f"{self.cycle_count}/{self.max_cycles}")
        
        # 如果当前累积的数据周期数超过新设置的最大周期数，则截取最新的周期数据
        if len(self.accumulated_data) > self.max_cycles:
            self.accumulated_data = self.accumulated_data[-self.max_cycles:]
            self.need_redraw = True
    
    def reset_cycles(self):
        """重置周期计数和累积数据"""
        self.data_mutex.lock()
        self.cycle_count = 1
        self.accumulated_data = []
        self.cycle_count_label.setText(f"{self.cycle_count}/{self.max_cycles}")
        self.need_redraw = True
        self.data_mutex.unlock()
        self.status_bar.showMessage("周期已重置", 2000)
    
    def update_plot_type(self):
        """更新图表类型"""
        self.need_redraw = True
    
    def clear_data(self):
        """清除数据"""
        self.data_mutex.lock()
        self.data_buffer = []
        self.accumulated_data = []
        self.cycle_count = 1
        self.cycle_count_label.setText(f"{self.cycle_count}/{self.max_cycles}")
        self.data_mutex.unlock()
        self.canvas.axes.clear()
        self.canvas.axes.grid(True, linestyle='--', alpha=0.7)
        self.canvas.scatter = None
        self.canvas.line = None
        self.canvas.draw()
        self.data_count_label.setText("数据点: 0")
    
    def update_plot(self, data):
        """更新数据，但不立即重绘"""
        current_time = time.time()
        
        # 更新数据缓冲区
        self.data_mutex.lock()
        self.data_buffer = data
        
        # 处理周期数据
        # 每收到一次数据视为一个周期
        if len(data) > 0:
            # 添加新周期数据
            self.accumulated_data.append(data)
            
            # 如果累积的周期数超过最大周期数，则移除最早的周期数据
            if len(self.accumulated_data) > self.max_cycles:
                self.accumulated_data.pop(0)
            
            # 更新周期计数
            self.cycle_count = min(self.cycle_count + 1, self.max_cycles)
            self.cycle_count_label.setText(f"{self.cycle_count}/{self.max_cycles}")
        
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size:]
        
        self.need_redraw = True
        self.data_mutex.unlock()
        
        # 更新数据点数量标签
        total_points = sum(len(cycle_data) for cycle_data in self.accumulated_data)
        self.data_count_label.setText(f"数据点: {total_points}")
    
    def redraw_plot(self):
        """重绘图表，由定时器触发"""
        if not self.need_redraw:
            return
            
        self.data_mutex.lock()
        accumulated_data_copy = self.accumulated_data.copy()
        self.data_mutex.unlock()
        
        if not accumulated_data_copy:
            return
            
        # 清除当前图表
        self.canvas.axes.clear()
        
        # 根据选择的图表类型绘制
        chart_type = self.chart_type_combo.currentText()
        
        # 合并所有周期的数据用于绘图
        all_data = []
        for cycle_data in accumulated_data_copy:
            all_data.extend(cycle_data)
        
        if not all_data:
            return
            
        # 创建X轴数据（相位）
        # 对于累积数据，我们需要为每个周期的每个数据点分配相位值
        phase_per_cycle = 360  # 每个周期的相位范围
        x_data = []
        
        for i, cycle_data in enumerate(accumulated_data_copy):
            cycle_phases = np.linspace(0, phase_per_cycle, len(cycle_data))
            x_data.extend(cycle_phases)
        
        if chart_type == "散点图":
            self.canvas.axes.scatter(x_data, all_data, alpha=0.7, s=10)
        elif chart_type == "线图":
            # 对于线图，我们可能需要按周期分别绘制
            for i, cycle_data in enumerate(accumulated_data_copy):
                cycle_phases = np.linspace(0, phase_per_cycle, len(cycle_data))
                self.canvas.axes.plot(cycle_phases, cycle_data, linewidth=1.0, 
                                     label=f"周期 {i+1}")
            # 如果周期数较多，可以选择不显示图例
            if len(accumulated_data_copy) <= 10:
                self.canvas.axes.legend(loc='upper right')
        
        # 设置图表标题和轴标签
        cycle_info = f"({self.cycle_count}/{self.max_cycles}周期)"
        self.canvas.axes.set_title(f"GIS局部放电PRPD图 {cycle_info}")
        self.canvas.axes.set_xlabel("相位 (0~360°)")
        self.canvas.axes.set_ylabel("幅值 (V)")
        
        # 设置网格
        self.canvas.axes.grid(True, linestyle='--', alpha=0.7)
        
        # 重绘画布
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
        self.need_redraw = False
    
    def update_status(self):
        """更新状态信息"""
        # 这里可以添加其他需要定时更新的状态信息
        pass
    
    def closeEvent(self, event):
        """关闭窗口事件"""
        # 断开MQTT连接
        self.mqtt_client.disconnect_from_broker()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 