import sqlite3
import sys
import pytest
from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                             QVBoxLayout, QFileDialog, QTextEdit, QMessageBox,
                             QProgressBar)
from transformers import pipeline


class AnalysisThread(QThread):
    finished = pyqtSignal(dict, str)
    error = pyqtSignal(str)

    def __init__(self, model, text):
        super().__init__()
        self.model = model
        self.text = text

    def run(self):
        try:
            result = self.model(
                self.text,
                candidate_labels=["Позитивный", "Негативный", "Нейтральный"]
            )
            self.finished.emit(result, self.text)
        except Exception as e:
            self.error.emit(str(e))


class SemanticAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.conn = None
        self.cursor = None

        self.init_ui()
        self.init_db()
        self.load_model()

    def init_ui(self):
        self.setWindowTitle('Семантический Анализ Текста')
        self.resize(600, 500)

        self.load_button = QPushButton('Загрузить Текст из Файла')
        self.load_button.clicked.connect(self.open_file_dialog)

        self.analyze_button = QPushButton('Анализировать Текст')
        self.analyze_button.clicked.connect(self.analyze_current_text)
        self.analyze_button.setEnabled(False)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Введите текст для анализа...")

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.result_label = QLabel('Результаты анализа будут отображены здесь')
        self.result_label.setWordWrap(True)

        self.history_label = QLabel('Последние анализы:')
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_label)
        layout.addWidget(self.history_label)
        layout.addWidget(self.history_text)

        self.setLayout(layout)

    def load_model(self):
        try:
            self.model = pipeline(
                "zero-shot-classification",
                model="DeepPavlov/rubert-base-cased-conversational"
            )
            self.analyze_button.setEnabled(True)
            self.result_label.setText("Модель загружена. Введите текст для анализа.")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось загрузить модель: {str(e)}"
            )
            self.analyze_button.setEnabled(False)

    def init_db(self):
        try:
            self.conn = sqlite3.connect('text_analysis.db')
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    text_content TEXT NOT NULL,
                    positive_score REAL,
                    negative_score REAL,
                    neutral_score REAL,
                    predicted_label TEXT
                )
            ''')
            self.conn.commit()
        except sqlite3.Error as e:
            QMessageBox.critical(
                self,
                "Ошибка базы данных",
                f"Ошибка при работе с базой данных: {str(e)}"
            )

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            'Открыть Текстовый Файл',
            '',
            'Текстовые файлы (*.txt);;Все файлы (*)'
        )

        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    text = f.read()
                self.text_edit.setPlainText(text)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Ошибка",
                    f"Ошибка при загрузке файла: {str(e)}"
                )

    def analyze_current_text(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(
                self,
                "Предупреждение",
                "Пожалуйста, введите текст для анализа"
            )
            return

        if not self.model:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Модель анализа не загружена"
            )
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Индикатор процесса без определенного конца

        self.analyze_thread = AnalysisThread(self.model, text)
        self.analyze_thread.finished.connect(self.on_analysis_finished)
        self.analyze_thread.error.connect(self.on_analysis_error)
        self.analyze_thread.start()

    def on_analysis_finished(self, result, text):
        self.progress_bar.setVisible(False)

        try:
            scores = {label: score for label, score in zip(result['labels'], result['scores'])}
            predicted_label = result['labels'][0]

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.cursor.execute('''
                INSERT INTO analysis_results 
                (timestamp, text_content, positive_score, negative_score, neutral_score, predicted_label)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, text, scores['Позитивный'], scores['Негативный'], scores['Нейтральный'], predicted_label))
            self.conn.commit()

            self.result_label.setText(
                f"Результаты анализа:\n"
                f"Позитивный: {scores['Позитивный']:.2f}\n"
                f"Негативный: {scores['Негативный']:.2f}\n"
                f"Нейтральный: {scores['Нейтральный']:.2f}\n"
                f"Основной тон: {predicted_label}"
            )

            self.update_history()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Ошибка при сохранении результатов: {str(e)}"
            )

    def on_analysis_error(self, error_msg):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(
            self,
            "Ошибка анализа",
            f"Ошибка при анализе текста: {error_msg}"
        )

    def update_history(self):
        try:
            self.cursor.execute('''
                SELECT timestamp, predicted_label 
                FROM analysis_results 
                ORDER BY timestamp DESC 
                LIMIT 5
            ''')
            history = self.cursor.fetchall()

            history_text = "\n".join(
                [f"{row[0]}: {row[1]}" for row in history]
            )
            self.history_text.setPlainText(history_text)
        except Exception as e:
            self.history_text.setPlainText(f'Ошибка при загрузке истории: {str(e)}')

    def closeEvent(self, event):
        if hasattr(self, 'analyze_thread') and self.analyze_thread.isRunning():
            self.analyze_thread.terminate()

        if self.conn:
            self.conn.close()

        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Установка стиля для лучшего отображения
    app.setStyle('Fusion')

    window = SemanticAnalysisApp()
    window.show()

    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


