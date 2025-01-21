import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import tensorflow as tf

model = VGG16(weights='imagenet')

last_conv_layer_name = 'block5_conv3'

grad_model = Model(
    inputs=model.input,
    outputs=[model.get_layer(last_conv_layer_name).output, model.output]
)

input_folder = r'D:\2024_Ecological_RiceSeed\seed_10\xi23_10'
output_folder = r'./grad_cam_results' 

os.makedirs(output_folder, exist_ok=True)

all_images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#📌 **Note:** Change the number of images selected, right here!
num_images = min(10, len(all_images)) 
selected_images = random.sample(all_images, num_images)

print(f"📸 Random select {num_images} images to process!")

for filename in selected_images:
    img_path = os.path.join(input_folder, filename)
    
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])  
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_outputs)[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    grad_cam = tf.reduce_sum(weights * conv_outputs[0], axis=-1)

    grad_cam = np.maximum(grad_cam, 0)
    grad_cam /= np.max(grad_cam)
    grad_cam = tf.image.resize(grad_cam[..., np.newaxis], (224, 224)).numpy()
    grad_cam = grad_cam[..., 0]

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.imshow(grad_cam, cmap='jet', alpha=0.5)  
    plt.axis('off')

    output_path = os.path.join(output_folder, f'grad_cam_{filename}')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f'✅ processed and saving {output_path}')

print("🎉 Done!")
