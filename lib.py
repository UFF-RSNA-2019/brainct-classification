import configparser
import numpy as np
import pydicom
import datetime
import sys
import cv2
import imutils
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from skimage.filters import threshold_multiotsu

# inicializacao
configparser = configparser.ConfigParser()
configparser.read('config.ini')
try:
    config = configparser['Geral']
except KeyError:
    now = datetime.datetime.now()
    msg = "Problema ao obter configuração, verifique se você está rodando o programa no diretório principal do projeto."
    print("[ERROR] {} : {}".format(now.strftime("%Y-%m-%d %H:%M:%S.%f"), msg), flush=True)
    sys.exit()


# CONSTANTES

# obtem classificacao de um elemento dos dados de treinamento
def get_classification(filename):
    train_file = "{}/stage_1_train.csv".format(config['TrainPath'])
    classes = np.zeros(6)
    id = filename[:12]
    try:
        idx = 0
        with open(train_file) as myfile:
            for line in myfile:
                pos = line.find(id)
                if (pos == 0):
                    pos = line.find(",")
                    value = int(line[pos + 1:pos + 2])
                    classes[idx] = value
                    idx += 1
                    if idx == 6:
                        break
    except:
        e = sys.exc_info()[0]
        log("erro ao abrir arquivo de treinamento: {}".format(e))
    return classes

# FUNCOES UTILITARIAS

def log(mensagem):
    now = datetime.datetime.now()
    print("[INFO] {} : {}".format(now.strftime("%Y-%m-%d %H:%M:%S.%f"), mensagem), flush=True)

def error(mensagem):
    now = datetime.datetime.now()
    print("[ERROR] {} : {}".format(now.strftime("%Y-%m-%d %H:%M:%S.%f"), mensagem), flush=True)

def plot(title, image, color_map=plt.cm.bone):
    plt.imshow(image, cmap=color_map)
    plt.title(title)
    plt.show()
