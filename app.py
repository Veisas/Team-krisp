import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, QTextEdit
from transformers import pipeline

class SemanticAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # Инициализация модели и пайплайнов
        self.model = pipeline("zero-shot-classification", model="DeepPavlov/rubert-base-cased-conversational")
        
        # Настройка окна приложения
        self.setWindowTitle('Семантический Анализ')
        self.resize(500, 400)
        
        # Виджеты
        self.load_button = QPushButton('Загрузить Текст', self)
        self.load_button.clicked.connect(self.open_file_dialog)
        
        self.text_edit = QTextEdit()
        self.result_label = QLabel('')
        
        # Макетирование виджетов
        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.result_label)
        self.setLayout(layout)
    
    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Открыть Файл', '', 'Текстовые файлы (*.txt)')
        if file_name:
            with open(file_name, 'r', encoding='utf-8') as f:
                text = f.read()
            self.analyze_text(text)
    
    def analyze_text(self, text):
        # Проведение семантического анализа
        result = self.model(text, ["Позитивный", "Негативный", "Нейтральный"])
        self.result_label.setText(f'Результат: {result}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SemanticAnalysisApp()
    window.show()
    sys.exit(app.exec_())