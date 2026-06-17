import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
 
from ml26.proyectos.P01_facial_expressionsV5.network import Network
from ml26.proyectos.P01_facial_expressionsV5.utils import (
    to_numpy,
    get_transforms,
    add_img_text,
)
from ml26.proyectos.P01_facial_expressionsV5.dataset import EMOTIONS_MAP
import pathlib
 
file_path = pathlib.Path(__file__).parent.absolute()
 
 
# Agregado Funcion para detectar y recortar la cara de la imagen
def crop_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return img  # si no detecta cara, usa la imagen completa
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]
 
 
def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path)
    img_cropped = crop_face(img)  # Agregado recortar la cara antes de transformar
    val_transforms, unnormalize = get_transforms("test", img_size=48)
    tensor_img = val_transforms(img_cropped)  # agregado usar imagen recortada
    denormalized = unnormalize(tensor_img)
    return img, tensor_img, denormalized
 
 
def predict(img_title_paths):
    """
    Hace la inferencia de las imagenes
    args:
    - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    """
    # Cargar el modelo
    modelo = Network(48, 7)
    modelo.load_model("modelo_1.pt")
    for path in img_title_paths:
        # Cargar la imagen
        # np.ndarray, torch.Tensor
        im_file = (file_path / path).as_posix()
        original, transformed, denormalized = load_img(im_file)
 
        # Inferencia cambie esta parte
        logits, proba = modelo.predict(transformed.unsqueeze(0)) #esto
        pred = torch.argmax(proba, -1).item()
        pred_label = EMOTIONS_MAP[pred]
 
        # Original / transformada
        h, w = original.shape[:2]
        resize_value = 300
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        img = add_img_text(img, f"Pred: {pred_label}")
 
        # Mostrar la imagen
        denormalized = to_numpy(denormalized)
        denormalized = cv2.resize(denormalized, (resize_value, resize_value))
        cv2.imshow("Predicción - original", img)
        cv2.imshow("Predicción - transformed", denormalized)
        cv2.waitKey(0)
 
 
if __name__ == "__main__":
    # agregado Lee automaticamente todas las imagenes de la carpeta test_imgs
    test_dir = file_path / "test_imgs"
    img_paths = [f"./test_imgs/{f}" for f in os.listdir(test_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    predict(img_paths)
