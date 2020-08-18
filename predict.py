#pip install -q -U "tensorflow-gpu==2.0.0b1"
#pip install -q -U tensorflow_hub
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json
import argparse

parser = argparse.ArgumentParser(description='Flower Image Classifier')

parser.add_argument('image_path', action="store")
parser.add_argument('model', action="store")
parser.add_argument('--top_k', action='store', type=int,
                    dest='k',
                    help='Returns the top K most likely classes')
parser.add_argument('--category_names', type=str,
                    help='Maps labels to flower names')

args = parser.parse_args()

model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image=image.numpy()
    return image

def prediction(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    final_image = np.expand_dims(processed_test_image, axis=0)
    preds = model.predict(final_image)
    preds_array = preds.flatten() 
    top_indices = np.argsort(-preds_array)[:top_k]
    probs = np.array([])
    classes = np.array([])
    
    for i in top_indices:
        probs = np.append(probs, preds_array[i])
        classes = np.append(classes, str(i+1))
    
    return probs, classes

if args.k == None:
    top_k=1
elif args.k > 0:
    top_k = args.k
else: top_k=1

probs, classes = prediction(args.image_path, model, top_k)
print('\nMost likely image class:')
print(classes)
print('Associated probability:')
print(probs)

top_classes = np.array([])

if args.category_names != None:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    for i in classes:
        top_classes = np.append(top_classes, class_names.get(i))
    print('Associated flower name:')
    print(top_classes)
