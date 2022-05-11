from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import click
import mann

def build_model(input_shape, use_mann_layers = False):
    input_layer = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Flatten()(input_layer)
    if use_mann_layers:
        for _ in range(5):
            x = mann.layers.MaskedDense(500, activation = 'relu')(x)
        output_layer = mann.layers.MaskedDense(10, activation = 'softmax')(x)
    else:
        for _ in range(5):
            x = tf.keras.layers.Dense(500, activation = 'relu')(x)
        output_layer = tf.keras.layers.Dense(10, activation = 'softmax')(x)
    model = tf.keras.models.Model(input_layer, output_layer)
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    return model

@click.command()
@click.option('--limit', '-l', type = int, default = None)
def main(limit):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    if limit:
        x_train = x_train[:limit]
        y_train = y_train[:limit]
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    callback = tf.keras.callbacks.EarlyStopping(
            'val_accuracy',
            min_delta = 0.005,
            patience = 5,
            restore_best_weights = True
    )

    # Build the control model
    control_model = build_model(x_train.shape[1:])
    control_model.fit(
        x_train,
        y_train,
        validation_split = 0.2,
        epochs = 100,
        batch_size = 512,
        callbacks = [callback],
        verbose = 2
    )
    print('\n')
    print('Control Results:')
    control_preds = control_model.predict(x_test).argmax(axis = 1)
    print(confusion_matrix(y_test, control_preds))
    print(classification_report(y_test, control_preds))
    print('\n')
    control_accuracy = accuracy_score(y_test, control_preds)

    one_shot_model = build_model(x_train.shape[1:], use_mann_layers = True)
    one_shot_model.fit(
        x_train[:1000],
        y_train[:1000],
        verbose = 2
    )
    one_shot_model = mann.utils.mask_model(
        one_shot_model,
        90,
        method = 'magnitude'
    )
    one_shot_model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    one_shot_model.fit(
        x_train,
        y_train,
        validation_split = 0.2,
        epochs = 100,
        batch_size = 512,
        callbacks = [callback],
        verbose = 2
    )
    print('\n')
    print('One Shot Results:')
    one_shot_preds = one_shot_model.predict(x_test).argmax(axis = 1)
    print(confusion_matrix(y_test, one_shot_preds))
    print(classification_report(y_test, one_shot_preds))
    print('\n')

    active_callback = mann.utils.ActiveSparsification(
        control_accuracy - 0.04,
        starting_sparsification = 50,
        sparsification_rate = 5,
        sparsification_patience = 5,
        stopping_delta = 0.005
    )
    iterative_model = build_model(x_train.shape[1:], use_mann_layers = True)
    iterative_model.fit(
        x_train[:1000],
        y_train[:1000],
        verbose = 2
    )
    iterative_model = mann.utils.mask_model(
        iterative_model,
        50,
        method = 'magnitude'
    )
    iterative_model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    iterative_model.fit(
        x_train,
        y_train,
        validation_split = 0.2,
        epochs = 100,
        batch_size = 512,
        callbacks = [active_callback],
        verbose = 2
    )
    print('\n')
    print('Iterative Results:')
    iterative_preds = iterative_model.predict(x_test).argmax(axis = 1)
    print(confusion_matrix(y_test, iterative_preds))
    print(classification_report(y_test, iterative_preds))
    print('\n')

if __name__ == '__main__':
    main()
