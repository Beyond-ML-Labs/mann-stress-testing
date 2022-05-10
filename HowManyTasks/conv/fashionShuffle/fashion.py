from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import click
import mann

def build_task_data(num_tasks, x_train, y_train, x_test, y_test):
    
    train_x = [x_train]
    train_y = [y_train]*num_tasks
    test_x = [x_test]
    test_y = [y_test]*num_tasks

    train_x_placeholder = x_train.reshape((x_train.shape[0], -1))
    test_x_placeholder = x_train.reshape((x_test.shape[0], -1))

    for i in range(num_tasks - 1):
        indices = np.arange(train_x_placeholder.shape[-1])
        np.random.shuffle(indices)
        train_x.append(
            np.asarray([x[indices] for x in train_x_placeholder]).reshape((train_x_placeholder.shape[0], 28, 28, 1))
        )
        test_x.append(
            np.asarray([x[indices] for x in test_x_placeholder]).reshape((test_x_placeholder.shape[0], 28, 28, 1))
        )
    
    return (train_x, train_y), (test_x, test_y)

def build_model(
    num_tasks,
    input_shape,
    output_shape,
    num_blocks,
    num_filters
):
    input_layers = [
        tf.keras.layers.Input(input_shape) for _ in range(num_tasks)
    ]
    x = mann.layers.MultiMaskedConv2D(num_filters, activation = 'relu')(input_layers)
    x = mann.layers.MultiMaskedConv2D(num_filters, activation = 'relu')(x)
    x = [mann.layers.SelectorLayer(i)(x) for i in range(num_tasks)]
    x = [tf.keras.layers.MaxPool2D()(x[i]) for i in range(num_tasks)]

    for block in range(num_blocks - 1):
        x = mann.layers.MultiMaskedConv2D(num_filters * 2**(block + 1), activation = 'relu')(x)
        x = mann.layers.MultiMaskedConv2D(num_filters * 2**(block + 1), activation = 'relu')(x)
        x = [mann.layers.SelectorLayer(i)(x) for i in range(num_tasks)]
        x = [tf.keras.layers.MaxPool2D()(x[i]) for i in range(num_tasks)]

    x = [mann.layers.SelectorLayer(i)(x) for i in range(num_tasks)]
    x = [tf.keras.layers.Flatten()(x[i]) for i in range(num_tasks)]
    x = mann.layers.MultiMaskedDense(num_filters * 4, activation = 'relu')(x)
    output_layer = mann.layers.MultiMaskedDense(output_shape, activation = 'softmax')(x)

    model = tf.keras.models.Model(input_layers, output_layer)
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    return model

@click.command()
@click.option('--num-blocks', '-b', type = int, default = 2)
@click.option('--num-filters', '-f', type = int, default = 8)
@click.option('--min-accuracy', '-a', type = float, default = 0.8)
def main(num_blocks, num_filters, min_accuracy):
    num_tasks = 2
    keep_going = True
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape((x_train.shape + (1,)))/255
    x_test = x_test.reshape((x_test.shape + (1,)))/255
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    while keep_going:

        (train_x, train_y), (test_x, test_y) = build_task_data(
            num_tasks,
            x_train,
            y_train,
            x_test,
            y_test
        )

        model = build_model(
            num_tasks,
            x_train.shape[1:],
            10,
            num_blocks,
            num_filters
        )
        x_subsets = [
            x[:100] for x in train_x
        ]
        y_subsets = [
            y[:100] for y in train_y
        ]
        
        prop = int(100/num_tasks)
        model = mann.utils.mask_model(
            model,
            100 - prop,
            x = x_subsets,
            y = y_subsets
        )
        model.compile(
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'],
            optimizer = 'adam'
        )
        callback = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 20,
            min_delta = 0.01,
            restore_best_weights = True
        )

        print(f'Training model for {num_tasks} tasks')
        
        model.fit(
            train_x,
            train_y,
            batch_size = 512,
            epochs = 100,
            validation_split = 0.2,
            callbacks = [callback],
            verbose = 0
        )
        predictions = [pred.argmax(axis = 1) for pred in model.predict(test_x)]
        accuracies = [
            accuracy_score(y_test, pred) for pred in predictions
        ]
        print(f'Accuracies: {accuracies}')
        if min(accuracies) < min_accuracy:
            print(f'Convergence failed at {num_tasks} tasks')
            keep_going = False
        else:
            print('Model Converged')
            print('\n')
            num_tasks += 1


if __name__ == '__main__':
    main()

