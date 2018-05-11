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
    print(request.FILES)
    img = request.FILES.get('image')
    img = Image.open(img)
    img = img.resize((224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    global model
    if model is None:
        model = DenseNet(classes=130, weights_path='server/dense161_result.h5')

    # 预测
    y = model.predict(x)
    y = y.flatten()
    preds = []
    max = (0, 0)
    for index, item in enumerate(y):
        if item > 0.5:
            preds.append((index, item))
        if max[1] <= item:
            max = (index, item)
        if len(preds) >= 3:
            break

    if len(preds) == 0:
        preds.append(max)

    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    print(preds)

    data = []
    for predItem in preds:
        for key, classItem in map.items():
            if classItem.get('id') == predItem[0] + 1:
                classItem['image'] = 'images/'+key+'.jpg'
                classItem['similarity'] = predItem[1].item()
                data.append(classItem)
                break

    info = json.dumps(data, ensure_ascii=False)
    return JsonResponse({"result": 0, "info": info})
