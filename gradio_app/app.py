import os
import uuid
import tempfile
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import gradio as gr
from ultralytics import YOLO

# 1) Подгружаем модель
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

# 2) Функция детекции и упаковки в ZIP
def detect(image, conf_threshold, selected_classes):
    results = model(image, conf=conf_threshold, verbose=False)[0]
    records = []
    for *box, score, cls_id in results.boxes.data.tolist():
        x1, y1, x2, y2 = map(int, box)
        score = float(score)
        cls_id = int(cls_id)
        cls_name = CLASS_NAMES[cls_id]
        if selected_classes and cls_name not in selected_classes:
            continue
        records.append({
            "class": cls_name,
            "confidence": round(score, 3),
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        })
    annotated = results.plot()
    # сохраняем png и csv во временный ZIP
    tmp_dir = tempfile.mkdtemp()
    img_path = os.path.join(tmp_dir, "annotated.png")
    csv_path = os.path.join(tmp_dir, "boxes.csv")
    zip_path = os.path.join(tmp_dir, f"results_{uuid.uuid4().hex}.zip")
    Image.fromarray(annotated).save(img_path)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(img_path, arcname="annotated.png")
        zf.write(csv_path, arcname="boxes.csv")
    return annotated, pd.DataFrame(records), zip_path

# 3) Собираем список примеров из папки с тренировочными картинками
examples_folder = os.path.join(os.path.dirname(__file__), "..", "data", "test", "images")
example_files = []
if os.path.isdir(examples_folder):
    for fname in sorted(os.listdir(examples_folder)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            example_files.append(os.path.join(examples_folder, fname))
    example_files = example_files[:6]  # первые 6 примеров

# 4) Строим интерфейс Blocks
with gr.Blocks(css=".gr-button {background: #008CBA; color: white}") as demo:
    gr.Markdown("# 🏠 LayoutVision")
    gr.Markdown("Автоматическое обнаружение ключевых элементов на плане квартиры")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="1) Загрузите план")
            conf = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="2) Порог confidence")
            run = gr.Button("🚀 Запустить детекцию")
            # Блок с примерами
            if example_files:
                gr.Examples(
                    examples=example_files,
                    inputs=[input_img],
                    label="📂 Примеры из тестового набора",
                    examples_per_page=6
                )
        with gr.Column(scale=1):
            out_img = gr.Image(type="numpy", label="🔍 Результат")
            out_table = gr.Dataframe(
                headers=["class","confidence","x1","y1","x2","y2"],
                label="📊 Координаты боксов"
            )
            out_zip = gr.File(label="📦 Скачать архив результатов")

    # 5) Привязываем кнопку к функции
    run.click(
        fn=detect,
        inputs=[input_img, conf],
        outputs=[out_img, out_table, out_zip]
    )

# 6) Запуск приложения    
if __name__ == "__main__":
    demo.launch(inbrowser=True, server_name="0.0.0.0", server_port=7860)