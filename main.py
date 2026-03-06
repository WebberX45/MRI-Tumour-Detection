from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

model = load_model('Model/model.h5')
model.build((None, 256, 256, 3))


class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def predict_tumor(image_path):
    IMAGE_SIZE = 256
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor"
    else:
        return f"Tumor: {class_labels[predicted_class_index]}"



def generate_gradcam(img_path, model):
    try:
        IMAGE_SIZE = 256

        img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Extract VGG16 backbone
        vgg_model = model.get_layer("vgg16")

        last_conv_layer = vgg_model.get_layer("block5_conv3")

        grad_model = tf.keras.models.Model(
            inputs=vgg_model.input,
            outputs=[last_conv_layer.output, vgg_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = tf.reduce_max(predictions)

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)


        # Load original image
        img = cv2.imread(img_path)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * 0.4 + img

        heatmap_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            "heatmap_" + os.path.basename(img_path)
        )

        cv2.imwrite(heatmap_path, superimposed_img)

        return heatmap_path

    except Exception as e:
        print("GradCAM error:", e)
        return None



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        file = request.files['file']
        if file:

            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            result = predict_tumor(file_location)

            try:
                heatmap_path = generate_gradcam(file_location, model)
            except Exception as e:
                print("GradCAM error:", e)
                heatmap_path = None

            heatmap_url = None

            if heatmap_path:
                heatmap_url = f'/uploads/{os.path.basename(heatmap_path)}'

            return render_template(
                'index.html',
                result=result,
                file_path=f'/uploads/{file.filename}',
                heatmap_path=heatmap_url
            )

    return render_template('index.html', result=None)


# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)