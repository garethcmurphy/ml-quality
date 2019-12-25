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
    url = 'https://storage.googleapis.com/applied-dl/heart.csv'
    dataframe = pd.read_csv(url)
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
        print('A batch of ages:', feature_batch['age'])
        print('A batch of targets:', label_batch)

    # We will use this batch to demonstrate several types of feature columns
    example_batch = next(iter(train_ds))[0]

    age = feature_column.numeric_column("age")
    demo(age, example_batch)

    age_buckets = feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    demo(age_buckets, example_batch)

    thal = feature_column.categorical_column_with_vocabulary_list(
        'thal', ['fixed', 'normal', 'reversible'])

    thal_one_hot = feature_column.indicator_column(thal)
    demo(thal_one_hot, example_batch)

    crossed_feature = feature_column.crossed_column(
        [age_buckets, thal], hash_bucket_size=1000)
    demo(feature_column.indicator_column(crossed_feature), example_batch)

    feature_columns = []

    # numeric cols
    for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
        feature_columns.append(feature_column.numeric_column(header))

    # bucketized cols
    age_buckets = feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    feature_columns.append(age_buckets)

    # indicator cols
    thal = feature_column.categorical_column_with_vocabulary_list(
        'thal', ['fixed', 'normal', 'reversible'])
    thal_one_hot = feature_column.indicator_column(thal)
    feature_columns.append(thal_one_hot)

    # embedding cols
    thal_embedding = feature_column.embedding_column(thal, dimension=8)
    feature_columns.append(thal_embedding)

    # crossed cols
    crossed_feature = feature_column.crossed_column(
        [age_buckets, thal], hash_bucket_size=1000)
    crossed_feature = feature_column.indicator_column(crossed_feature)
    feature_columns.append(crossed_feature)

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
