import sys
import sqlite3
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, 
                             QVBoxLayout, QFileDialog, QTextEdit, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from transformers import pipeline

class SemanticAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # Инициализация модели и пайплайнов
        self.model = pipeline("zero-shot-classification", model="DeepPavlov/rubert-base-cased-conversational")
        
        # Инициализация базы данных
        self.init_db()
        
        # Настройка окна приложения
        self.setWindowTitle('Семантический Анализ')
        self.resize(800, 600)
        
        # Виджеты
        self.load_button = QPushButton('Загрузить Текст', self)
        self.load_button.clicked.connect(self.open_file_dialog)
        
        self.save_button = QPushButton('Сохранить в БД', self)
        self.save_button.clicked.connect(self.save_to_db)
        self.save_button.setEnabled(False)
        
        self.text_edit = QTextEdit()
        self.result_label = QLabel('')
        
        # Таблица для отображения истории
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(['Дата', 'Текст', 'Результат', 'Метка'])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # Макетирование виджетов
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        
        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.result_label)
        layout.addWidget(QLabel('История анализа:'))
        layout.addWidget(self.history_table)
        self.setLayout(layout)
        
        # Загружаем историю из БД
        self.load_history()
    
    def init_db(self):
        """Инициализация базы данных SQLite"""
        self.conn = sqlite3.connect('semantic_analysis.db')
        self.cursor = self.conn.cursor()
        
        # Создаем таблицу, если ее нет
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                result TEXT NOT NULL,
                label TEXT NOT NULL
            )
        ''')
        self.conn.commit()
    
    def save_to_db(self):
        """Сохранение текущего результата в базу данных"""
        if not hasattr(self, 'current_result'):
            return
            
        text = self.text_edit.toPlainText()
        if not text:
            return
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result_str = str(self.current_result)
        label = self.current_result['labels'][0]
        
        self.cursor.execute('''
            INSERT INTO analysis_results (timestamp, text, result, label)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, text, result_str, label))
        self.conn.commit()
        
        # Обновляем таблицу истории
        self.load_history()
    
    def load_history(self):
        """Загрузка истории анализа из базы данных"""
        self.cursor.execute('''
            SELECT timestamp, text, result, label 
            FROM analysis_results 
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        results = self.cursor.fetchall()
        
        self.history_table.setRowCount(len(results))
        for row_idx, row_data in enumerate(results):
            for col_idx, col_data in enumerate(row_data):
                item = QTableWidgetItem(str(col_data))
                self.history_table.setItem(row_idx, col_idx, item)
    
    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Открыть Файл', '', 'Текстовые файлы (*.txt)')
        if file_name:
            with open(file_name, 'r', encoding='utf-8') as f:
                text = f.read()
            self.text_edit.setPlainText(text)
            self.analyze_text(text)
    
    def analyze_text(self, text):
        """Анализ текста и вывод результата"""
        if not text.strip():
            self.result_label.setText('Введите текст для анализа')
            self.save_button.setEnabled(False)
            return
            
        # Проведение семантического анализа
        self.current_result = self.model(text, ["Позитивный", "Негативный", "Нейтральный"])
        
        # Форматирование результата для отображения
        labels = self.current_result['labels']
        scores = self.current_result['scores']
        result_str = ", ".join([f"{label}: {score:.2f}" for label, score in zip(labels, scores)])
        
        self.result_label.setText(f'Результат: {result_str}')
        self.save_button.setEnabled(True)
    
    def closeEvent(self, event):
        """Закрытие соединения с БД при закрытии приложения"""
        self.conn.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SemanticAnalysisApp()
    window.show()
    sys.exit(app.exec_())
