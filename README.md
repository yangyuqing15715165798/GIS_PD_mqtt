# GIS局部放电在线监测系统

这是一个基于MQTT协议的GIS（气体绝缘开关设备）局部放电在线监测系统，使用PySide6构建GUI界面，并通过matplotlib实现数据可视化。

## 功能特点

- 通过MQTT协议实时接收局部放电数据
- 支持多种图表显示方式（散点图、线图）
- 左右双图布局：左侧PRPD图，右侧PRPS三维图
- 可调整数据缓冲区大小
- 实时显示连接状态和数据点数量
- 美观的用户界面
- 多线程处理MQTT通信，避免主线程阻塞
- 消息队列缓冲机制，提高数据处理效率
- 优化的图表绘制策略，减少UI卡顿
- 线程安全的数据访问机制
- 支持数据周期累积显示，可自定义累积周期数
- **参考正弦波显示**：在PRPD图中叠加显示参考正弦波，便于分析放电相位
- **自定义颜色方案**：支持多种预设颜色方案，增强PRPS三维图的可视化效果
- **数据库存储**：支持将接收到的数据保存到SQLite数据库，便于后期分析和查询
- **历史数据可视化**：支持从数据库查询历史数据并生成PRPD和PRPS图表
- **单位转换功能**：支持在毫伏(mV)和dBm单位之间切换，满足不同分析需求

![image](https://github.com/user-attachments/assets/f33521ad-5467-4829-aa53-b996937ea39c)
![image](https://github.com/user-attachments/assets/9fc7e4f3-171f-4c80-a2cf-5e00924b5e8e)
![image](https://github.com/user-attachments/assets/85fb6f8b-361a-4034-bf7b-42ab2f0c81a1)
![image](https://github.com/user-attachments/assets/f5f4825c-2db6-4528-8b26-ada3042e43c5)
![image](https://github.com/user-attachments/assets/d7929378-b7e0-4e13-87f0-7080a933a0d3)
![image](https://github.com/user-attachments/assets/b83920e0-cb76-46d4-89b2-cb11621122db)
![image](https://github.com/user-attachments/assets/c1315c0a-3fbd-4d6d-98d1-25816708dc2d)
![image](https://github.com/user-attachments/assets/ba5bf493-14ec-4d69-98fb-e8c5398fb629)
![image](https://github.com/user-attachments/assets/be498b8d-5020-4e5e-9d2a-11b08f8e4d09)
![image](https://github.com/user-attachments/assets/40447295-98ba-424c-a5ae-2173ef75076e)
![image](https://github.com/user-attachments/assets/a9aa1d4e-0664-4826-a605-3be3d53c9c4c)
![image](https://github.com/user-attachments/assets/273bdeb5-e3af-4888-9b79-69f1712cd479)
![image](https://github.com/user-attachments/assets/3408db10-dac5-4c53-99b3-dbb585824e78)
![image](https://github.com/user-attachments/assets/5fe72bdc-6dce-49ef-9039-08cdd09fd51c)
![image](https://github.com/user-attachments/assets/24476168-75c4-4301-a389-1fa5d04be962)





## 技术实现

- **PySide6**: 用于构建现代化GUI界面
- **MQTT**: 使用paho-mqtt库实现MQTT通信
- **Matplotlib**: 用于数据可视化，支持多种图表类型
- **Matplotlib 3D**: 使用mplot3d工具包实现三维PRPS图
- **多线程**: 使用QThread处理MQTT通信，避免阻塞主线程
- **消息队列**: 使用Python的queue模块实现消息缓冲
- **互斥锁**: 使用QMutex保证线程安全
- **定时器**: 使用QTimer控制UI更新频率
- **自定义颜色映射**: 使用LinearSegmentedColormap创建自定义颜色方案
- **SQLite**: 使用轻量级的SQLite数据库进行数据存储与管理

## 数据可视化

系统提供两种互补的数据可视化方式：

1. **PRPD图** (相位分辨局部放电图)：
   - 二维散点图或线图
   - X轴表示相位(0~360°)
   - Y轴表示放电幅值
   - 可累积多个周期数据
   - **参考正弦波**：可叠加显示正弦波，帮助分析放电相位与电压周期的关系
   - 正弦波振幅可调，自动适配数据范围

2. **PRPS图** (相位分辨脉冲序列图)：
   - 三维表面图
   - X轴表示相位(0~360°)
   - Y轴表示周期序号
   - Z轴表示放电幅值
   - 默认显示最新的50个周期数据
   - **多种颜色方案**：支持多种预设颜色映射方案，包括：
     - 默认方案（黑-白-黄-红）
     - 蓝绿红
     - 黑蓝紫
     - 绿黄红
   - 颜色映射增强数据可视化效果，便于识别不同幅值的放电

3. **历史数据可视化**：
   - 支持从数据库查询历史数据并生成图表
   - 可选择PRPD散点图、PRPD线图或PRPS三维图
   - 可调整显示的周期数量
   - 支持与实时监测相同的参考正弦波和颜色方案设置
   - 支持导出高分辨率图像用于报告和分析

用户可以通过界面选项切换是否显示PRPS三维图和参考正弦波。

## 数据处理流程

系统采用多级缓冲和定时更新机制处理数据：

1. **MQTT消息接收**：
   - MQTT消息在独立线程中接收，避免阻塞主线程
   - 接收到的原始十六进制数据经过解析和转换
   - 支持断开连接后重新连接，确保数据流的连续性
   - 原始数据可选择性地保存到数据库

2. **消息队列缓冲**：
   - 处理后的数据放入消息队列，而不是直接更新UI
   - 队列大小限制为10条消息，避免内存溢出
   - 如果队列已满，新消息将被丢弃，保证系统稳定性

3. **数据更新频率**：
   - 消息队列处理频率：每50毫秒处理一次队列中的数据
   - 图表重绘频率：每200毫秒重绘一次图表（相当于每秒5次更新）
   - 状态信息更新：每1000毫秒（1秒）更新一次状态信息

4. **数据周期累积**：
   - 每收到一次完整数据视为一个周期
   - 系统可累积多个周期的数据进行显示（默认50个周期）
   - 用户可通过界面设置PRPD图的累积周期数
   - 当累积周期数达到设定值后，新数据会替换最早的周期数据
   - PRPS三维图始终显示最新的50个周期数据，独立于PRPD图的设置
   - 周期数据可选择性地保存到数据库

5. **数据流向**：
   ```
   MQTT消息 → 消息队列 → 数据周期累积 → 图表显示(PRPD+PRPS) → 数据库存储
   ```

6. **线程安全**：
   - 使用互斥锁保护数据缓冲区的读写操作
   - 图表绘制前复制数据，避免在绘制过程中数据被修改
   - 连接/断开操作使用延时执行，避免UI卡顿
   - 数据库操作使用异常处理，确保系统稳定性

这种设计确保了系统在高频率数据流下仍能保持流畅运行，同时提供近乎实时的数据可视化。

## 数据库存储

系统使用SQLite数据库存储接收到的数据，具有以下特点：

1. **数据表结构**：
   - `cycle_data`: 存储处理后的周期数据，包含时间戳、周期编号和数据内容
   - `raw_data`: 存储原始接收到的十六进制数据，包含时间戳、Broker地址、主题和数据内容

2. **存储选项**：
   - 用户可通过界面选择是否启用数据保存功能
   - 默认情况下，数据保存功能处于关闭状态

3. **状态显示**：
   - 状态栏显示数据库连接状态
   - 显示已存储的周期数据和原始数据数量
   - 显示数据保存功能的启用状态

4. **数据查询**：
   - 支持按时间范围查询周期数据
   - 支持获取最新的周期数据
   - 支持获取数据统计信息

5. **数据库位置**：
   - 数据库文件 `gis_pd_data.db` 保存在程序所在目录
   - 每次程序启动时自动连接或创建数据库

通过数据库存储功能，用户可以在后期对历史数据进行深入分析，无需担心实时数据的丢失。

## 历史数据查看与可视化

系统提供了强大的历史数据查看与可视化功能：

1. **数据库查询界面**：
   - 支持查询周期数据和原始数据
   - 可选择查询最新数据或按时间范围查询
   - 表格形式显示查询结果，支持查看详细数据内容
   - 双击数据行可查看完整数据详情

2. **历史数据可视化**：
   - 支持将查询到的历史数据生成PRPD或PRPS图表
   - 提供三种图表类型：PRPD散点图、PRPD线图和PRPS三维图
   - 可调整显示的周期数量，灵活控制数据范围
   - 支持与实时监测相同的参考正弦波和颜色方案设置

3. **图表导出功能**：
   - 支持将生成的历史数据图表导出为高分辨率图像
   - 支持PNG、JPEG等常见图像格式
   - 导出的图像分辨率为300DPI，适合用于报告和出版物

4. **数据分析优势**：
   - 可比较不同时间点的放电特性
   - 可分析特定事件前后的放电模式变化
   - 可研究局部放电的长期趋势和周期性变化
   - 为设备状态评估提供数据支持

通过历史数据可视化功能，系统不仅能实时监测局部放电情况，还能回溯历史数据，进行全面的设备健康评估。

## 性能优化

- 使用独立线程处理MQTT消息，避免主线程阻塞
- 实现消息队列缓冲机制，控制数据处理频率
- 分离数据更新和图表绘制逻辑，提高UI响应性
- 优化matplotlib绘图参数，提高绘图效率
- 使用互斥锁保护共享数据，确保线程安全
- 对于三维图，采用数据重采样策略，确保不同周期数据点数一致
- 重建3D轴而非清除，解决三维图缩小问题
- 固定图形纵横比，保持一致的显示效果
- 数据库操作使用异步方式，避免阻塞UI线程

减轻主线程负担：MQTT消息处理在单独的线程中进行
控制更新频率：不再每收到消息就更新图表，而是以固定频率更新
避免数据竞争：使用互斥锁保护共享数据

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行程序：

```bash
python gis_pd_mqtt_gui.py
```

2. 在界面中设置MQTT Broker的地址、端口和主题
3. 点击"连接"按钮连接到MQTT服务器
4. 连接成功后，系统将自动接收数据并绘制PRPD和PRPS图
5. 可以通过下拉菜单选择PRPD图的类型（散点图、线图）
6. 可以调整数据缓冲区大小，控制显示的数据点数量
7. 可以使用"显示PRPS三维图"选项切换是否显示三维图
8. 设置"PRPD累积周期数"可以控制显示多少次接收到的数据
9. 使用"显示参考正弦波"选项可以在PRPD图中叠加显示正弦波
10. 通过"正弦波振幅"调节参考正弦波的大小
11. 通过"PRPS颜色方案"下拉菜单选择三维图的颜色映射
12. 使用"单位: mV"按钮可以在毫伏(mV)和dBm单位之间切换
13. 使用"保存数据到数据库"选项可以控制是否将数据保存到数据库（默认关闭）
14. 使用"查看数据库"按钮可以打开数据库查询界面
15. 在数据库查询界面中，可以查询历史数据并生成PRPD/PRPS图表
16. 使用"清除数据"按钮可以重置图表
17. 使用"重置周期"按钮可以清除已累积的周期数据，重新开始累积

## 数据格式

系统接收十六进制格式的数据，每4个字符解析为一个16进制数，并按以下公式转换：
- 转换值 = 十进制值 * 3.3 / 4096，单位为毫伏(mV)
- 支持将毫伏(mV)转换为dBm：dBm = 毫伏值 * 54.545 - 81.818

## 系统要求

- Python 3.6+
- PySide6
- paho-mqtt
- matplotlib (含mplot3d)
- numpy
- sqlite3 (Python标准库)

## 代码架构

系统由以下几个主要类组成：

- **MplCanvas**: Matplotlib画布类，用于在Qt界面中嵌入matplotlib图形，支持2D和3D子图
- **MQTTThread**: MQTT处理线程，避免阻塞主线程
- **MQTTClient**: MQTT客户端类，处理MQTT连接和消息接收
- **DatabaseManager**: 数据库管理类，负责数据的存储和查询
- **DatabaseViewDialog**: 数据库查看对话框，提供数据查询和可视化功能
- **HistoricalChartsDialog**: 历史数据可视化对话框，支持生成PRPD和PRPS图表
- **MainWindow**: 主窗口类，管理GUI和业务逻辑 

## 主要功能详解

### 参考正弦波

参考正弦波功能在PRPD图中叠加显示一个正弦曲线，帮助用户分析放电相位与交流电压周期的关系。该功能具有以下特点：

- 可通过界面上的复选框开启或关闭
- 正弦波振幅可调，范围从0.1到3.0
- 振幅会根据实际数据的范围自动缩放，确保正弦波始终可见且不会过大
- 正弦波居中显示，便于与放电数据对比
- 使用红色线条显示，透明度为0.7，避免遮挡放电数据点
- 在图例中标记为"参考正弦波"

通过观察放电点与参考正弦波的相对位置，用户可以直观地判断放电是发生在电压正半周、负半周还是过零点附近，为分析放电特性提供重要参考。

### 自定义颜色方案

PRPS三维图支持多种预设颜色方案，增强数据可视化效果：

1. **默认方案**：黑-白-黄-红渐变，适合常规观察
2. **蓝绿红**：从蓝色到红色的渐变，突出高值区域
3. **黑蓝紫**：从黑色到紫色的渐变，提供良好的深度感知
4. **绿黄红**：从绿色到红色的渐变，类似热力图效果

每种颜色方案都经过精心设计，使用LinearSegmentedColormap从颜色列表创建平滑渐变。用户可以根据个人偏好或特定分析需求选择不同的颜色方案。

### 数据库存储功能

数据库存储功能使用SQLite实现，提供以下特性：

1. **自动存储**：
   - 周期数据：每收到一个完整周期的数据后自动保存
   - 原始数据：保存完整的原始十六进制数据，便于后期深入分析
   - 每条记录都包含精确的时间戳信息

2. **灵活控制**：
   - 可通过界面上的复选框启用或禁用数据保存功能（默认关闭）
   - 状态栏实时显示数据库中存储的记录数量

3. **查询功能**：
   - 支持按时间范围查询历史数据
   - 支持获取最新的周期数据记录
   - 提供数据统计功能，显示总记录数

4. **数据安全**：
   - 使用事务机制确保数据完整性
   - 异常处理机制防止数据库操作导致程序崩溃
   - 程序退出时自动关闭数据库连接，确保数据安全

通过数据库存储功能，用户可以轻松保存和管理长时间的监测数据，为设备的预测性维护和故障分析提供有力支持。

### 历史数据可视化功能

历史数据可视化功能允许用户查看和分析存储在数据库中的历史数据：

1. **数据查询与筛选**：
   - 支持按最新数据或时间范围查询
   - 可调整查询数量，灵活控制数据量
   - 表格形式显示查询结果，便于浏览

2. **多种图表类型**：
   - PRPD散点图：直观显示放电点的分布
   - PRPD线图：显示放电随相位的变化趋势
   - PRPS三维图：同时展示放电相位、周期和幅值的关系

3. **图表自定义选项**：
   - 可调整显示的周期数量
   - 可开启/关闭参考正弦波
   - 可调整正弦波振幅
   - 可选择不同的颜色方案

4. **图像导出功能**：
   - 支持导出高质量图像用于报告和分析
   - 保持300DPI的高分辨率，确保图像清晰度
   - 支持多种常见图像格式

通过历史数据可视化功能，用户可以回溯分析设备的历史运行状态，识别潜在问题，并为预测性维护提供依据。

### 连接管理优化

系统的连接管理机制经过优化，提供更可靠的MQTT连接体验：

- 连接和断开操作使用延时执行，避免UI卡顿
- 断开连接时正确清理资源，包括停止线程和定时器
- 重新连接时创建新的MQTT客户端，确保状态干净
- 使用互斥锁保护线程共享变量
- 添加超时机制，防止线程停止操作无限等待
- 完善的错误处理和状态反馈

这些优化确保了系统在频繁连接/断开操作下仍能保持稳定运行。

### 单位转换功能

系统支持在毫伏(mV)和dBm单位之间切换，方便用户根据不同需求进行数据分析：

1. **单位切换按钮**：
   - 位于图表设置区域，默认显示"单位: mV"
   - 点击后切换到dBm单位，按钮文本变为"单位: dBm"
   - 再次点击可切换回毫伏单位

2. **转换公式**：
   - 毫伏转dBm: dBm = 毫伏值 * 54.545 - 81.818
   - dBm转毫伏: mV = (dBm值 + 81.818) / 54.545

3. **界面更新**：
   - 切换单位后，PRPD图和PRPS图的Y轴/Z轴标签会自动更新
   - 图表中的数据点会根据当前选择的单位进行转换
   - 参考正弦波也会根据转换后的数据范围自动调整

4. **历史数据视图**：
   - 历史数据查看对话框也支持单位转换功能
   - 单位设置与主界面独立，可以在不同视图中使用不同的单位

单位转换功能使系统更加灵活，能够适应不同用户的习惯和分析需求，特别是对于需要在不同单位体系下比较数据的专业用户。

## 更新日志

### 2025年6月20日
- 添加了单位转换功能，支持在毫伏(mV)和分贝毫瓦(dBm)之间切换
- 修复了PRPS 3D图表的Z轴范围自动调整功能：
  - 针对dBm单位进行特殊处理，确保Z轴最小值不低于-80 dBm
  - 自动调整Z轴范围，确保数据显示清晰且有意义
  - 对于毫伏单位，确保最小值不低于0（物理上有意义）
  - 修复了在单位切换时Z轴范围不正确的问题
- 改进了历史数据查询功能，支持更精确的时间范围筛选

## 单位转换功能

### 功能描述
系统支持在毫伏(mV)和分贝毫瓦(dBm)单位之间进行切换，方便用户根据需要选择合适的单位进行数据分析。

### 使用方法
- 在主界面和历史数据查询界面中，点击"单位: mV"按钮可以切换显示单位
- 切换后，所有图表的单位标签和数据值都会自动更新
- PRPS 3D图表的Z轴范围会根据所选单位自动调整，确保数据显示清晰

### 单位转换公式
- 毫伏转dBm: dBm = 毫伏值 * 54.545 - 81.818
- dBm转毫伏: mV = (dBm值 + 81.818) / 54.545

### Z轴范围自动调整
- 当使用dBm单位时：
  - Z轴最小值不低于-80 dBm（一个合理的下限）
  - 确保Z轴范围足够宽（至少20 dBm），以便更好地显示数据变化
  - 如果范围太窄，则以数据的平均值为中心，向上下各扩展10 dBm
- 当使用毫伏单位时：
  - 确保Z轴最小值不低于0（物理上有意义）
  - 根据数据的最大值和最小值动态调整Z轴范围

 
