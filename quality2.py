#!/usr/bin/env python3
"""test ml"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def main():
    """main"""
    dataframe = pd.read_csv("./data.csv")
    dataframe.head()

    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    batch_size = 5  # A small batch sized is used for demonstration purposes
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    for feature_batch, label_batch in train_ds.take(1):
        print('Every feature:', list(feature_batch.keys()))
        print('A batch of ages:', feature_batch['hindex'])
        print('A batch of targets:', label_batch)

    # We will use this batch to demonstrate several types of feature columns

    feature_columns = []

    # numeric cols
    for header in ['hindex']:
        feature_columns.append(feature_column.numeric_column(header))

    # bucketized cols
    hindex = feature_column.numeric_column("hindex")
    age_buckets = feature_column.bucketized_column(
        hindex, boundaries=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    feature_columns.append(age_buckets)

    # indicator cols
    data_type = feature_column.categorical_column_with_vocabulary_list(
        'data_type', ['test', 'production'])
    data_type_one_hot = feature_column.indicator_column(data_type)
    feature_columns.append(data_type_one_hot)

    # embedding cols
    #data_type_embedding = feature_column.embedding_column(data_type, dimension=8)
    # feature_columns.append(data_type_embedding)

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=5)

    _, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    """df to dataset"""
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)
    return dataset

# A utility method to create a feature column
# and to transform a batch of data


def demo(feature_column1, example_batch):
    """demo feature layer"""
    feature_layer = layers.DenseFeatures(feature_column1)
    print(feature_layer(example_batch).numpy())


if __name__ == "__main__":
    main()
