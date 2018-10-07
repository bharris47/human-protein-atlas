import json
import os
import shutil
from argparse import ArgumentParser
from datetime import datetime

from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

from protein_atlas.data import load_data
from protein_atlas.model import simple_convnet

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train-path')
    parser.add_argument('--image-path')
    parser.add_argument('--params-path')
    parser.add_argument('--artifact-directory')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    artifact_directory = os.path.join(args.artifact_directory, timestamp)
    checkpoint_format = os.path.join(artifact_directory, 'weights.{epoch:02d}-{val_loss:.6f}.hdf5')
    log_dir = os.path.join(artifact_directory, 'logs')
    os.makedirs(log_dir)
    shutil.copy(args.params_path, artifact_directory)

    X, y = load_data(args.train_path, args.image_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    del X, y

    mean, std = X_train.mean(), X_train.std()
    print('training mean: {} std: {}'.format(mean, std))

    X_train -= mean
    X_train /= std

    X_val -= mean
    X_val /= std

    with open(args.params_path) as params_file:
        hyperparameters = json.load(params_file)

    model = simple_convnet(
        input_shape=X_train[0].shape,
        n_classes=y_train.shape[-1],
        base_filters=hyperparameters['base_filters'],
        activation=hyperparameters['activation'],
        fc_size=hyperparameters['fc_size'],
        dropout=hyperparameters['dropout'],
        classifier_activation=hyperparameters['classifier_activation']
    )
    model.summary()
    model.compile('adam', loss=hyperparameters['loss'], metrics=['acc'])

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=hyperparameters['batch_size'],
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[
            ModelCheckpoint(checkpoint_format),
            TensorBoard(log_dir=log_dir)
        ]
    )
