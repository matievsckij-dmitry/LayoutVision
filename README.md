LayoutVision — автоматическое обнаружение элементов на архитектурных планах

![Python (https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) ![Docker (https://img.shields.io/badge/docker-20.10-blue)](https://www.docker.com/) ![Gradio (https://img.shields.io/badge/gradio-4.44.1-orange)](https://gradio.app/) ![Ultralytics YOLO (https://img.shields.io/badge/YOLO-ultralytics-yellow)](https://github.com/ultralytics/ultralytics) 🔗 Demo на Hugging Face (https://huggingface.co/spaces/DmitryMatievsckij/layout-vision)

Описание  
LayoutVision — это высокоточная система компьютерного зрения на базе YOLO11-seg, которая в автоматическом режиме выявляет на плане квартиры следующие ключевые объекты:  
• двери  
• окна  
• лестницы  
• периметр помещения  
• санузел  

Результатом работы является:  
1. Аннотированное изображение (PNG) с подсветкой найденных элементов  
2. Таблица (CSV) с точными координатами и размерами каждого объекта  

Ключевые особенности  
• Высокая точность: модель обучена на специализированном датасете архитектурных планов  
• Реальное время: интерактивная работа через Gradio-интерфейс  
• Удобный экспорт: сохраняйте только PNG-аннотацию и CSV-отчёт  
• Готовый деплой: контейнеризация Docker для мгновенного старта  

Технологический стек  
• Язык: Python 3.10+  
• Модель: Ultralytics YOLO11-seg  
• UI: Gradio  
• Контейнеры: Docker  

Запуск локально  
1. git clone https://github.com/your-org/LayoutVision.git  
2. cd LayoutVision  
3. docker build -t layoutvision .  
4. docker run gpus all -p 7860:7860 layoutvision  
5. Откройте в браузере http://localhost:7860 и загрузите план  

Попробовать онлайн  
https://huggingface.co/spaces/DmitryMatievsckij/layout-vision  
