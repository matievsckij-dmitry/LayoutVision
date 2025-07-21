🚀 LayoutVision — автоматическое обнаружение элементов на архитектурных планах  

---

<table>
<tr>
<td width="120"><img src="https://xrplace.io/images/tild3265-3934-4332-b136-336334653833__logo_white_1.png" width="100" alt="XR Place Logo"></td>
<td>
<b>Заказчик:</b> <a href="https://xrplace.io">XR Place</a> — компания, создающая интерактивные 3D квартиры и дома для сайтов застройщиков. Проект позволяет получать точные координаты опорных точек по плану помещения для последующей 3D-визуализации.
</td>
</tr>
</table>

---

🔍 Что это?  
LayoutVision — современная система компьютерного зрения на базе YOLO11, которая в пару кликов распознаёт на чертеже квартиры ключевые объекты:  
• 🚪 Двери  
• 🪟 Окна  
• 🏃‍♂️ Лестницы  
• 📐 Периметр помещений  
• 🚽 Санузел  

---

🎁 Что получаете?  
1. 🖼 Аннотированное изображение в формате PNG  
2. 📊 Таблицу с координатами и размерами (CSV)  

--- 

🔗 Попробовать онлайн прямо сейчас:  
https://huggingface.co/spaces/DmitryMatievsckij/layout-vision

--- 

✨ Преимущества  
• 🎯 Высокая точность на узкоспециализированном датасете  
• ⚡️ Обработка в реальном времени через Gradio  
• 💾 Экспорт результатов только в PNG и CSV — ничего лишнего  
• 🐳 Контейнеризация в Docker для мгновенного развёртывания  

---  

🛠 Технологический стек  
• Python 3.10+  
• Ultralytics YOLO11  
• Gradio (веб-интерфейс)  
• Docker & Docker Compose  

---

📂 Итоговая структура проекта  
```
LayoutVision/
├── data/                    ← датасеты для обучения 
│   └── test/
│   └── train/
│   └── valid/
│   └── data.yaml
│
├── gradio_app/              ← исходники веб-приложения  
│   └── app.py 
│
├── models/                  ← конфиги и веса модели  
│   └── args.yaml
│   └── best.pt
│   └── last.pt
│
├── notebooks/               ← Jupyter-ноутбук анализа и обучения  
│   └── train_YOLO11.ipynb  
│
├── train/                   ← артефакты обучения (графики, метрики)   
│
├── Dockerfile               ← сборка основного образа  
├── docker-compose.yml       ← конфигурация сервиса для Docker Compose  
├── requirements.txt         ← Python-зависимости  
└── README.md                ← этот файл  
```

---

🐳 Запуск через Docker Compose  

1. Клонируем репозиторий и переходим в папку:  
   
```
   git clone https://github.com/your-org/LayoutVision.git
   cd LayoutVision
   ```
  
2. Убеждаемся, что рядом с docker-compose.yml лежат оба Dockerfile:  
   - ./Dockerfile — для сборки полного образа LayoutVision  
   - ./gradio_app/app.py — (опционально) для изолированной сборки веб-приложения  
3. Запускаем сервис:  
   
```
   docker-compose up --build
   ```
  
4. Открываем в браузере http://localhost:7860 и загружаем план.
