import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                              QHBoxLayout, QWidget, QPushButton, QFileDialog, 
                              QLabel, QTabWidget, QMessageBox, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

# 设置matplotlib中文支持
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class SignalProcessor(QThread):
    """信号处理线程"""
    progress_updated = Signal(int)
    processing_finished = Signal(object, object, object, object)
    error_occurred = Signal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            self.progress_updated.emit(10)
            
            # 读取CSV文件
            df = pd.read_csv(self.file_path, header=None)
            self.progress_updated.emit(30)
            
            # 提取时间和信号数据
            time = df[0].values
            signal = df[1].values
            self.progress_updated.emit(50)
            
            # 去除直流分量
            signal_processed = signal - np.mean(signal)
            self.progress_updated.emit(70)
            
            # FFT计算
            fft_result = np.fft.fft(signal_processed)
            time_step = np.mean(np.diff(time))
            n = len(signal_processed)
            freqs = np.fft.fftfreq(n, d=time_step)
            
            # 提取正频率部分
            positive_freqs = freqs[:n//2]
            positive_magnitude = np.abs(fft_result[:n//2])
            self.progress_updated.emit(100)
            
            self.processing_finished.emit(time, signal, positive_freqs, positive_magnitude)
            
        except Exception as e:
            self.error_occurred.emit(f"处理数据时发生错误: {str(e)}")

class PlotCanvas(FigureCanvas):
    """matplotlib画布组件"""
    def __init__(self, parent=None, main_window=None, width=12, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='white')
        super().__init__(self.fig)
        self.setParent(parent)
        self.main_window = main_window
        self.freqs = None
        self.magnitude = None
        self.time_data = None
        self.signal_data = None
        self.is_spectrum = False
        self.cursor_line = None
        
    def format_frequency(self, freq):
        """格式化频率显示"""
        if abs(freq) >= 1e6:
            return f"{freq/1e6:.3f} MHz"
        elif abs(freq) >= 1e3:
            return f"{freq/1e3:.3f} kHz"
        else:
            return f"{freq:.3f} Hz"
            
    def format_magnitude(self, mag):
        """格式化幅度显示"""
        if mag >= 1e6:
            return f"{mag/1e6:.3f}M"
        elif mag >= 1e3:
            return f"{mag/1e3:.3f}k"
        else:
            return f"{mag:.3f}"
            
    def on_mouse_move(self, event):
        """鼠标移动事件处理"""
        if not event.inaxes:
            if self.main_window:
                self.main_window.status_label.setText("就绪")
            return
            
        if self.is_spectrum and self.freqs is not None:
            # FFT频谱图的鼠标跟踪
            freq = event.xdata
            if freq is not None:
                # 找到最接近的数据点
                if hasattr(self, 'freq_scale_factor'):
                    actual_freq = freq * self.freq_scale_factor
                else:
                    actual_freq = freq
                    
                idx = np.argmin(np.abs(self.freqs - actual_freq))
                exact_freq = self.freqs[idx]
                magnitude = self.magnitude[idx]
                
                # 更新十字光标
                self.update_cursor(event.xdata, event.ydata)
                
                # 格式化显示文本
                freq_text = self.format_frequency(exact_freq)
                mag_text = self.format_magnitude(magnitude)
                
                if self.main_window:
                    self.main_window.status_label.setText(
                        f"频率: {freq_text} | 幅度: {mag_text} | 坐标: ({event.xdata:.3f}, {event.ydata:.2e})"
                    )
                    
        elif not self.is_spectrum and self.time_data is not None:
            # 时域信号的鼠标跟踪
            time_val = event.xdata
            if time_val is not None:
                # 找到最接近的数据点
                idx = np.argmin(np.abs(self.time_data - time_val))
                exact_time = self.time_data[idx]
                signal_val = self.signal_data[idx]
                
                # 更新十字光标
                self.update_cursor(event.xdata, event.ydata)
                
                if self.main_window:
                    self.main_window.status_label.setText(
                        f"时间: {exact_time:.6f} s | 幅度: {signal_val:.6f} | 坐标: ({event.xdata:.6f}, {event.ydata:.6f})"
                    )
                    
    def update_cursor(self, x, y):
        """更新十字光标"""
        ax = self.fig.gca()
        
        # 移除旧的十字光标
        if self.cursor_line:
            for line in self.cursor_line:
                line.remove()
        
        # 绘制新的十字光标
        self.cursor_line = []
        self.cursor_line.append(ax.axvline(x, color='gray', linestyle='--', alpha=0.7, linewidth=0.8))
        self.cursor_line.append(ax.axhline(y, color='gray', linestyle='--', alpha=0.7, linewidth=0.8))
        
        self.draw_idle()
        
    def on_mouse_leave(self, event):
        """鼠标离开事件"""
        if self.cursor_line:
            for line in self.cursor_line:
                line.remove()
            self.cursor_line = None
            self.draw_idle()
            
        if self.main_window:
            self.main_window.status_label.setText("就绪")
        
    def plot_signal(self, time, signal, title="信号波形"):
        """绘制时域信号"""
        self.fig.clear()
        self.is_spectrum = False
        self.time_data = time
        self.signal_data = signal
        
        ax = self.fig.add_subplot(111)
        ax.plot(time, signal, 'b-', linewidth=1)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('信号幅度')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # 连接鼠标事件
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('axes_leave_event', self.on_mouse_leave)
        
        self.fig.tight_layout()
        self.draw()
        
    def plot_spectrum(self, freqs, magnitude, title="FFT频谱"):
        """绘制频谱"""
        self.fig.clear()
        self.freqs = freqs
        self.magnitude = magnitude
        self.is_spectrum = True
        
        ax = self.fig.add_subplot(111)
        
        # 智能选择频率单位和缩放
        max_freq = np.max(freqs)
        if max_freq >= 1e6:
            display_freqs = freqs / 1e6
            self.freq_scale_factor = 1e6
            ax.set_xlabel('频率 (MHz)')
        elif max_freq >= 1e3:
            display_freqs = freqs / 1e3
            self.freq_scale_factor = 1e3
            ax.set_xlabel('频率 (kHz)')
        else:
            display_freqs = freqs
            self.freq_scale_factor = 1
            ax.set_xlabel('频率 (Hz)')
            
        ax.plot(display_freqs, magnitude, 'r-', linewidth=1)
        ax.set_ylabel('幅度')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # 连接鼠标事件
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('axes_leave_event', self.on_mouse_leave)
        
        self.fig.tight_layout()
        self.draw()

class SignalAnalyzerGUI(QMainWindow):
    """主窗口类"""
    def __init__(self):
        super().__init__()
        self.time_data = None
        self.signal_data = None
        self.freq_data = None
        self.magnitude_data = None
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("信号分析工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 原始信号标签页
        self.original_canvas = PlotCanvas(self, main_window=self)
        self.tab_widget.addTab(self.original_canvas, "原始信号")
        
        # FFT频谱标签页
        self.fft_canvas = PlotCanvas(self, main_window=self)
        self.tab_widget.addTab(self.fft_canvas, "FFT频谱")
        
        main_layout.addWidget(self.tab_widget)
        
        # 状态栏
        self.status_label = QLabel("就绪")
        self.statusBar().addWidget(self.status_label)
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # 文件选择按钮
        self.load_button = QPushButton("选择CSV文件")
        self.load_button.clicked.connect(self.load_file)
        layout.addWidget(self.load_button)
        
        # 分析按钮
        self.analyze_button = QPushButton("开始分析")
        self.analyze_button.clicked.connect(self.analyze_signal)
        self.analyze_button.setEnabled(False)
        layout.addWidget(self.analyze_button)
        
        # 导出按钮
        self.export_button = QPushButton("导出图像")
        self.export_button.clicked.connect(self.export_plots)
        self.export_button.setEnabled(False)
        layout.addWidget(self.export_button)
        
        # 文件路径标签
        self.file_label = QLabel("未选择文件")
        layout.addWidget(self.file_label)
        
        layout.addStretch()
        
        return panel
        
    def load_file(self):
        """加载CSV文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择CSV文件", "", "CSV文件 (*.csv);;所有文件 (*)"
        )
        
        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"已选择: {file_path.split('/')[-1]}")
            self.analyze_button.setEnabled(True)
            self.status_label.setText("文件已加载，点击开始分析")
            
    def analyze_signal(self):
        """分析信号"""
        if not hasattr(self, 'file_path'):
            return
            
        # 禁用按钮
        self.analyze_button.setEnabled(False)
        self.load_button.setEnabled(False)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建处理线程
        self.processor = SignalProcessor(self.file_path)
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.processing_finished.connect(self.on_processing_finished)
        self.processor.error_occurred.connect(self.on_error)
        self.processor.start()
        
        self.status_label.setText("正在处理数据...")
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def on_processing_finished(self, time_data, signal_data, freq_data, magnitude_data):
        """处理完成回调"""
        self.time_data = time_data
        self.signal_data = signal_data
        self.freq_data = freq_data
        self.magnitude_data = magnitude_data
        
        # 绘制图形
        self.original_canvas.plot_signal(time_data, signal_data, "原始信号波形")
        self.fft_canvas.plot_spectrum(freq_data, magnitude_data, "FFT频谱（正频率部分）")
        
        # 恢复界面状态
        self.progress_bar.setVisible(False)
        self.analyze_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.export_button.setEnabled(True)
        
        self.status_label.setText("分析完成")
        
        # 显示统计信息
        self.show_statistics()
        
    def on_error(self, error_message):
        """错误处理"""
        QMessageBox.critical(self, "错误", error_message)
        
        # 恢复界面状态
        self.progress_bar.setVisible(False)
        self.analyze_button.setEnabled(True)
        self.load_button.setEnabled(True)
        
        self.status_label.setText("处理失败")
        
    def show_statistics(self):
        """显示统计信息"""
        if self.signal_data is not None and self.magnitude_data is not None:
            # 计算统计信息
            signal_max = np.max(self.signal_data)
            signal_min = np.min(self.signal_data)
            signal_mean = np.mean(self.signal_data)
            signal_std = np.std(self.signal_data)
            
            # 找到主要频率成分
            max_freq_idx = np.argmax(self.magnitude_data[1:]) + 1  # 排除直流分量
            dominant_freq = self.freq_data[max_freq_idx]
            
            stats_text = f"""
信号统计信息:
- 最大值: {signal_max:.4f}
- 最小值: {signal_min:.4f}
- 均值: {signal_mean:.4f}
- 标准差: {signal_std:.4f}
- 主频率: {dominant_freq:.2f} Hz
- 采样点数: {len(self.signal_data)}
            """
            
            QMessageBox.information(self, "统计信息", stats_text)
            
    def export_plots(self):
        """导出图像"""
        if self.time_data is None:
            return
            
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if export_dir:
            try:
                # 导出原始信号图
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                ax1.plot(self.time_data, self.signal_data, 'b-', linewidth=1)
                ax1.set_xlabel('时间 (s)')
                ax1.set_ylabel('信号幅度')
                ax1.set_title('原始信号波形')
                ax1.grid(True, alpha=0.3)
                fig1.tight_layout()
                fig1.savefig(f"{export_dir}/原始信号.png", dpi=300, bbox_inches='tight')
                plt.close(fig1)
                
                # 导出频谱图
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(self.freq_data, self.magnitude_data, 'r-', linewidth=1)
                ax2.set_xlabel('频率 (Hz)')
                ax2.set_ylabel('幅度')
                ax2.set_title('FFT频谱（正频率部分）')
                ax2.grid(True, alpha=0.3)
                fig2.tight_layout()
                fig2.savefig(f"{export_dir}/FFT频谱.png", dpi=300, bbox_inches='tight')
                plt.close(fig2)
                
                QMessageBox.information(self, "导出成功", f"图像已保存到: {export_dir}")
                
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出图像时发生错误: {str(e)}")

def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 设置现代化样式
    
    # 设置应用程序字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    window = SignalAnalyzerGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()