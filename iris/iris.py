from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import base

# 데이터셋
# IRIS_TRAINING = "iris_training.csv"
# IRIS_TEST = "iris_test.csv"
IRIS_TRAINING = "cloud_watch_training.csv"
IRIS_TEST = "cloud_watch_test.csv"

# 데이터셋을 불러옵니다.
training_set = base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float64)


test_set = base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float64)

# 모든 특성이 실수값을 가지고 있다고 지정합니다
# feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=30)]

# 10, 20, 10개의 유닛을 가진 3층 DNN를 만듭니다
# classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
#                                             hidden_units=[10, 20, 10],
#                                             n_classes=3,
#                                             model_dir="/tmp/iris_model")
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=2,
                                            model_dir="/tmp/iris_model")


# 모델을 학습시킵니다.
# 0.966667 /
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

# 정확도를 평가합니다.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('정확도: {0:f}'.format(accuracy_score))

# 새로운 두 개의 꽃 표본을 분류합니다.
# new_samples = np.array(
#     [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
# y = list(classifier.predict(new_samples))
print('예측: {}'.format(str(y)))