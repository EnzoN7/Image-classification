from random import randint
import matplotlib.pyplot as plt


def plot_training_analysis(_history, _t_metrics):
    acc = _history.history[_t_metrics]
    val_acc = _history.history['val_' + _t_metrics]
    loss = _history.history['loss']
    val_loss = _history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', linestyle="--",label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', linestyle="--",label='Training loss')
    plt.plot(epochs, val_loss,'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def plot_random_images(_xtrain, _ytrain, _labels):
    plt.figure(figsize=(12, 12))
    
    indices = [randint(0, len(_xtrain) - 1) for i in range(0, 9)]
    for i in range(0, 9):
        plt.subplot(3, 3, i+1)
        plt.title(_labels[int(_ytrain[indices[i]])])
        plt.imshow(_xtrain[indices[i]])

    plt.tight_layout()
    plt.show()