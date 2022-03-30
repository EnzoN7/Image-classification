import matplotlib.pyplot as plt


def plot_training_analysis(_history, _t_metrics):
    acc = _history.history[_t_metrics]
    val_acc = _history.history['val_' + _t_metrics]
    loss = _history.history['loss']
    val_loss = _history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', linestyle="--",label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', linestyle="--",label='Training loss')
    plt.plot(epochs, val_loss,'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()