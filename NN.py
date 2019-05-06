#
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

#Затем создаем сеть VGG16 и загружаем веса, обученные на наборе данных ImageNet:
model = VGG16(weights='imagenet')

#Загружаем картинку, преобразуем ее в массив numpy и выполняем предварительную обработку
img = image.load_img('image_file_name.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#Выполняем распознавание объекта на изображении:
preds = model.predict(x)

#Результат распознавания - это массив из 1000 элементов.
#Выберем 3 элемента с самой высокой вероятностью и напечатаем их:
print('Результаты распознавания:', decode_predictions(preds, top=3)[0])

#