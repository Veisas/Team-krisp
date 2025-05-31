import sys
import sqlite3
import pytest
from datetime import datetime
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton,
                             QVBoxLayout, QHBoxLayout, QLabel,
                             QFileDialog, QTextEdit, QSplitter)
from transformers import pipeline


class SentimentAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()

        # Инициализация модели
        self.model = pipeline("zero-shot-classification",
                              model="DeepPavlov/rubert-base-cased-conversational")

        # Инициализация БД
        self.init_db()

        # Настройка интерфейса
        self.init_ui()

        # Настройка графиков
        self.setup_plots()

    def init_db(self):
        """Инициализация базы данных"""
        self.conn = sqlite3.connect('sentiment_analysis.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                text_content TEXT,
                positive REAL,
                negative REAL,
                neutral REAL
            )
        ''')
        self.conn.commit()

    def init_ui(self):
        """Настройка пользовательского интерфейса"""
        self.setWindowTitle('Анализ тональности текста')
        self.resize(1000, 600)

        # Основные виджеты
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Введите текст для анализа...")

        self.analyze_btn = QPushButton('Анализировать')
        self.analyze_btn.clicked.connect(self.analyze_text)

        self.load_btn = QPushButton('Загрузить из файла')
        self.load_btn.clicked.connect(self.load_from_file)

        # Контейнер для графиков
        self.plot_widget = pg.GraphicsLayoutWidget()

        # Разметка
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel('Входной текст:'))
        left_panel.addWidget(self.text_edit)
        left_panel.addWidget(self.analyze_btn)
        left_panel.addWidget(self.load_btn)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel('Визуализация результатов:'))
        right_panel.addWidget(self.plot_widget)

        splitter = QSplitter()
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(splitter)

    def setup_plots(self):
        """Настройка графиков pyqtgraph"""
        # Основной график (столбчатая диаграмма)
        self.bar_plot = self.plot_widget.addPlot(title="Распределение тональности")
        self.bar_plot.setLabel('left', 'Вероятность')
        self.bar_plot.setLabel('bottom', 'Категория')
        self.bar_plot.setYRange(0, 1)

        # График истории (линейный)
        self.plot_widget.nextRow()
        self.history_plot = self.plot_widget.addPlot(title="История анализов")
        self.history_plot.setLabel('left', 'Оценка')
        self.history_plot.setLabel('bottom', 'Время')
        self.history_plot.addLegend()

        # Инициализация данных
        self.bars = pg.BarGraphItem(x=[0, 1, 2], height=[0, 0, 0], width=0.6,
                                    brushes=['g', 'r', 'b'])
        self.bar_plot.addItem(self.bars)

        # Линии для истории
        self.pos_line = self.history_plot.plot([], [], pen='g', name='Позитивный')
        self.neg_line = self.history_plot.plot([], [], pen='r', name='Негативный')
        self.neu_line = self.history_plot.plot([], [], pen='b', name='Нейтральный')

        # Список для хранения текстовых меток
        self.text_items = []

    def analyze_text(self):
        """Анализ текста и обновление графиков"""
        text = self.text_edit.toPlainText()
        if not text.strip():
            return

        # Получение результатов от модели
        result = self.model(text, ["Позитивный", "Негативный", "Нейтральный"])

        # Сохранение в БД
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        scores = {label: score for label, score in zip(result['labels'], result['scores'])}

        self.cursor.execute('''
            INSERT INTO results (timestamp, text_content, positive, negative, neutral)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, text, scores['Позитивный'], scores['Негативный'], scores['Нейтральный']))
        self.conn.commit()

        # Обновление графиков
        self.update_bar_chart(scores)
        self.update_history_plot()

    def update_bar_chart(self, scores):
        """Обновление столбчатой диаграммы"""
        x = [0, 1, 2]
        height = [scores['Позитивный'], scores['Негативный'], scores['Нейтральный']]
        self.bars.setOpts(x=x, height=height)

        # Удаление старых текстовых меток
        for item in self.text_items:
            self.bar_plot.removeItem(item)
        self.text_items.clear()

        # Добавление новых текстовых меток
        for i, (label, val) in enumerate(scores.items()):
            text = pg.TextItem(f"{label}\n{val:.2f}", anchor=(0.5, 1))
            text.setPos(i, val + 0.05)
            self.bar_plot.addItem(text)
            self.text_items.append(text)

    def update_history_plot(self):
        """Обновление графика истории"""
        self.cursor.execute('''
            SELECT timestamp, positive, negative, neutral 
            FROM results 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        data = self.cursor.fetchall()

        if not data:
            return

        timestamps = [row[0] for row in data]
        pos = [row[1] for row in data]
        neg = [row[2] for row in data]
        neu = [row[3] for row in data]

        x_axis = list(range(len(timestamps)))

        self.pos_line.setData(x_axis, pos)
        self.neg_line.setData(x_axis, neg)
        self.neu_line.setData(x_axis, neu)

        # Настройка оси X
        axis = self.history_plot.getAxis('bottom')
        axis.setTicks([[(i, ts[11:19]) for i, ts in enumerate(timestamps)]])

    def load_from_file(self):
        """Загрузка текста из файла"""
        filename, _ = QFileDialog.getOpenFileName(self, 'Открыть файл', '', 'Текстовые файлы (*.txt)')
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                self.text_edit.setPlainText(f.read())

    def closeEvent(self, event):
        """Закрытие соединения с БД при выходе"""
        self.conn.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SentimentAnalysisApp()
    window.show()
    sys.exit(app.exec_())