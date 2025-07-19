import gradio as gr
from ultralytics import YOLO

model = YOLO('train/weights/best.pt')

def detect(image):
    results = model(image)
    annotated_image = results[0].plot()
    return annotated_image


iface = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil"),  # используем gr.Image для ввода
    outputs=gr.Image(type="pil"),  # используем gr.Image для вывода
    title="YOLOv11 Object Detection",
    description="Загрузите изображение, чтобы увидеть, как модель детектирует объекты."
)

iface.launch()