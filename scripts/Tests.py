import numpy as np
import matplotlib.pyplot as plt


def test_data(_model, _labels, _idx, _xtest, _ytest, _print):
    data = []
    data.append(_xtest[_idx])
    data = np.asarray(data)
    dim = data[0].shape
    data = data.astype(np.float32).reshape(data.shape[0], dim[0], dim[1], dim[2])

    prediction = _model.predict(data)
    idxBestPrediction = np.argmax(prediction)
    bestPrediction = _labels[idxBestPrediction]
    res = prediction[0][idxBestPrediction] * 100

    if _print:
        print("PREDICTIONS sur la donnée n°" + str(_idx) + "/" + str(len(_xtest) - 1))
        for i in range(0, len(_labels)):
            print('     ' + _labels[i] + ' -> ' + "{0:.2f}%".format(prediction[0][i] * 100.))
        print('\nRESULTAT :         ' + bestPrediction + ' / ' + "{0:.2f}%".format(res))
        print('Résultat attendu : ' + _labels[int(_ytest[_idx])])
        plt.imshow(_xtest[_idx])

    return bestPrediction


def test_model(_model, _labels, _xtest, _ytest):
    sim = len(_xtest)
    for i in range(len(_xtest)):
        bestPrediction = test_data(_model, _labels, i, _xtest, _ytest, False)
        if (bestPrediction != _labels[int(_ytest[i])]):
            sim -= 1
    print("\nVALIDITE DU MODELE : " + "{0:.2f}%".format(sim / len(_xtest) * 100))