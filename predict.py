from numpy import genfromtxt
from joblib import load
import datetime
import lib

def append_resultado(texto):
    with open("{}/{}".format(output_path, resultado_filename), "a") as resultado_file:
        resultado_file.write("{}\n".format(texto))

# configuracao
features_path = lib.config['FeaturesPath']
models_path = lib.config['ModelsPath']
output_path = lib.config['ResultadosPath']

# 1. lê os arquivos do desafio com os dados de teste (id e features)
# ndarray features
stage1_test_x = genfromtxt('{}/stage1_test_x.csv'.format(features_path), delimiter=',')

# array IDs
f = open('{}/stage1_test_id.csv'.format(features_path), 'r')
stage1_test_id = f.read().splitlines()
f.close()

# 2. lê o modelo
model_file = '{}/grid_search.joblib'.format(models_path)
clf = load(model_file)

# 3. cria arquivo de saída
resultado_filename = "{}_grid_search.csv".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
append_resultado("ID,Label")

# 4. percorre cada item fazendo predict e gravando em um arquivo para envio
idx = 0
for feature in stage1_test_x:
    y_pred = clf.predict(feature.reshape(1, -1))
    y_pred = y_pred.toarray().astype(int)
    # append no arquivo de saida
    append_resultado("{}_epidural,{}".format(stage1_test_id[idx], y_pred[0][0]))
    append_resultado("{}_intraparenchymal,{}".format(stage1_test_id[idx], y_pred[0][1]))
    append_resultado("{}_intraventricular,{}".format(stage1_test_id[idx], y_pred[0][2]))
    append_resultado("{}_subarachnoid,{}".format(stage1_test_id[idx], y_pred[0][3]))
    append_resultado("{}_subdural,{}".format(stage1_test_id[idx], y_pred[0][4]))
    append_resultado("{}_any,{}".format(stage1_test_id[idx], y_pred[0][5]))
    idx += 1

print ("Done")

