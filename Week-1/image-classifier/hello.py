from flask import Flask
from keras.models import load_model
#ResNet50 es un clasificador de proposito general
from keras.applications.resnet50 import ResNet50


app = Flask(__name__)

MODEL_PATH = 'models/your_model.h5'

model = ResNet50(weights = 'imagenet')

@app.route('/')
def index():
    return render_template('index.html')

def model_predict(img_path, model):
    #Cargamos la imagen
    img = image.load(img_path)

    #Preprocesamos la imagen
    #Convertimos la imagen a un array de dos dimensiones para procesarla
    x = image.img_to_array(img)

    preds = model.predict(x)

    return preds



@app.route('/predict')
def upload():
    if request.method == 'POST':
        #f es la imagen que se recibe desde el POST
        f = request.file['file']

        f.save(file_path)

        #Ocupando el metodo mode_predict de Keras le pasamos la imagen y el modelo y nos regresa una serie de prediciones
        preds = model_predict(file_path,model)

        #Obtenemos solo la preddion mas alta
        pred_class = decode_predictions(preds, top=1)

        return pred_class



