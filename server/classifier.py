# -*- coding: utf-8 -*-
"""
classify the posted image
"""
import os
import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from keras.models import Model
from keras.preprocessing import image
from PIL import Image
import tensorflow as tf
import numpy as np

from server.class_map import map
from server.densenet161 import DenseNet

model = None


@csrf_exempt
def classify(request):
    img = request.FILES.get('image')
    img = Image.open(img)
    img = img.resize((224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    global model
    if model is None:
        model = DenseNet(classes=120, weights_path='server/dense161_result.h5')

    # 预测
    preds = model.predict(x)
    preds = preds.flatten()
    preds = np.argmax(preds)

    for key, classItem in map.items():
        if classItem.get('id') == preds + 1:
            classItem['image'] = 'image/'+key+'.jpg'
            preds = classItem
            break

    info = json.dumps(preds, ensure_ascii=False)
    return JsonResponse({"result": 0, "info": info})
