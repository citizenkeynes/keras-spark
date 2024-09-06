import unittest
import tensorflow as tf
from pyspark.sql import SparkSession
from keras_spark.models import KerasSparkModel
import pyspark.sql.functions as F
import pandas as pd

spark = SparkSession.builder.master("local").appName("Test").getOrCreate()

class TestPredict(unittest.TestCase):

    def test_predict_01(self):

        spark = SparkSession.builder.master("local").appName("Test").getOrCreate()

        pdf = pd.DataFrame({
           "feature1" :  tf.random.normal([100]).numpy().tolist()
          ,"feature2" :  tf.random.normal([100]).numpy().tolist()
        })
        spark_df = spark.createDataFrame(pdf).withColumn("partition_id",F.lit(0))

        with tf.distribute.OneDeviceStrategy("cpu:0").scope():
            inputs0 = tf.keras.Input(shape=[1], name="feature1")
            inputs1 = tf.keras.Input(shape=[1], name="feature2", dtype="int32")
            inputs = [inputs0,inputs1]
            rec = lambda x: tf.keras.layers.Lambda(lambda x: __import__("tensorflow").cast(x,"float32"),output_shape=(1,))(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(tf.keras.layers.Concatenate(-1)([rec(i) for i in inputs]))
            outputs  = tf.keras.layers.Lambda(lambda x:x,name="label",output_shape=(1,))(outputs)
            model = KerasSparkModel(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='binary_crossentropy')

        # Train the model using the Spark DataFrame
        df = model.predict(spark_df, batch_size=10, epochs=10,partition_col="partition_id",partition_values=[0])
        result = df.select(F.col("model_output.label")[0]).first()[0]
        assert(result>0)
        assert(df.count()==100)

    def test_predict_2(self):

        pdf = pd.DataFrame({
              "feature1": tf.random.normal([200]).numpy().tolist()
            , "feature2": tf.random.normal([200]).numpy().tolist()
        })
        spark_df = spark.createDataFrame(pdf).withColumn("partition_id", F.lit(0))

        with tf.distribute.OneDeviceStrategy("cpu:0").scope():
            inputs0 = tf.keras.Input(shape=[1], name="feature1")
            inputs1 = tf.keras.Input(shape=[1], name="feature2", dtype="int32")
            inputs = [inputs0, inputs1]
            rec = lambda x: tf.keras.layers.Lambda(lambda x: __import__("tensorflow").cast(x, "float32"),output_shape=(1,))(x)
            output_pre = tf.keras.layers.Dense(1, activation="sigmoid")(tf.keras.layers.Concatenate(-1)([rec(i) for i in inputs]))
            output1 = tf.keras.layers.Lambda(lambda x: x, name="label", output_shape=(1,))(output_pre)
            output2 = tf.keras.layers.Lambda(lambda x: x + 100, name="label2", output_shape=(1,))(output_pre)
            model = KerasSparkModel(inputs=inputs, outputs=[output1, output2])
            model.compile(optimizer='adam', loss='binary_crossentropy')

        # Train the model using the Spark DataFrame
        df = model.predict(spark_df, batch_size=10, epochs=10, partition_col="partition_id", partition_values=[0])
        result = df.select(F.col("model_output.label2")[0]).first()[0]
        assert (result > 100)
        assert (df.count() == 200)



class TestTrain(unittest.TestCase):

    def test_fit_1(self):
        pdf = pd.DataFrame({
              "feature1": tf.random.normal([100]).numpy().tolist()
            , "feature2": tf.random.normal([100]).numpy().tolist()
            , "label": tf.cast(tf.random.normal([100]).numpy().tolist(), "int32")

        })
        spark_df = spark.createDataFrame(pdf).withColumn("partition_id", F.lit(0))
        strategy = tf.distribute.OneDeviceStrategy("cpu:0")

        with strategy.scope():
            inputs0 = tf.keras.Input(shape=[1], name="feature1")
            inputs1 = tf.keras.Input(shape=[1], name="feature2", dtype="int32")
            inputs = [inputs0, inputs1]
            rec = lambda x: tf.keras.layers.Lambda(lambda x: tf.cast(x, "float32"))(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(
                tf.keras.layers.Concatenate(-1)([rec(i) for i in inputs]))
            outputs = tf.keras.layers.Lambda(lambda x: x, name="label")(outputs)
            model = KerasSparkModel(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='binary_crossentropy')

        # Train the model using the Spark DataFrame
        model.fit(spark_df, batch_size=10, epochs=10, reader_type="PetaStormReader")

        # Check if the model's training weights have been updated (simple check)
        weights_after_training = model.get_weights()
        self.assertIsNotNone(weights_after_training)


    def test_fit_2(self):

        pdf = pd.DataFrame({
            "feature1": tf.random.normal([100]).numpy().tolist()
            , "feature2": tf.random.normal([100, 2]).numpy().tolist()
            , "label": tf.cast(tf.random.normal([100, 2]), "int32").numpy().tolist()

        })
        spark_df = spark.createDataFrame(pdf).withColumn("partition_id",
                                                                   F.when(F.rand() > 0.5, 1).otherwise(0))
        strategy = tf.distribute.OneDeviceStrategy("cpu:0")

        with strategy.scope():
            inputs0 = tf.keras.Input(shape=[1], name="feature1")
            inputs1 = tf.keras.Input(shape=[2], name="feature2", dtype="int32")
            inputs = [inputs0, inputs1]
            rec = lambda x: tf.keras.layers.Lambda(lambda x: tf.cast(x, "float32"))(x)
            outputs = tf.keras.layers.Dense(2, activation="sigmoid")(
                tf.keras.layers.Concatenate(-1)([rec(i) for i in inputs]))
            outputs = tf.keras.layers.Lambda(lambda x: x, name="label")(outputs)
            model = KerasSparkModel(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='binary_crossentropy')

        # Train the model using the Spark DataFrame
        model.fit(spark_df, batch_size=10, epochs=10, reader_type="PetaStormReader")

        # Check if the model's training weights have been updated (simple check)
        weights_after_training = model.get_weights()
        self.assertIsNotNone(weights_after_training)

    def test_string_classifier(self):

        feature_str = ["a" if i <50 else "b" for i in range(100)]
        label = [1.0 if i <50 else 0.0 for i in range(100)]

        pdf = pd.DataFrame({
              "feature_str": feature_str
            , "label": label

        })
        spark_df = spark.createDataFrame(pdf).withColumn("partition_id",
                                                                   F.when(F.rand() > 0.5, 1).otherwise(0))
        strategy = tf.distribute.OneDeviceStrategy("cpu:0")

        with strategy.scope():
            input = tf.keras.Input(shape=[1], name="feature_str" ,dtype="string")
            hashl = tf.keras.layers.Hashing(num_bins=10,output_mode="one_hot")
            nn = tf.keras.models.Sequential([tf.keras.layers.Dense(10,"elu"),tf.keras.layers.Dense(1,"sigmoid")],name="label")
            model = KerasSparkModel(inputs=input, outputs=nn(hashl(input)))
            model.compile(optimizer='adam', loss='binary_crossentropy')

        model.fit(spark_df, batch_size=100, epochs=50, nr_partitions=2, reader_type="PetaStormReader")

        preds = model.predict(spark_df)

        pred_a = preds.filter("feature_str='a'").select(F.col("model_output")["label"][0]).first()[0]
        pred_b = preds.filter("feature_str='b'").select(F.col("model_output")["label"][0]).first()[0]

        assert (pred_a > pred_b)

    def test_higher_dims(self):

        feature_3d =  tf.random.normal([100, 10, 2, 1])
        label = feature_3d*0.5+0.1

        pdf = pd.DataFrame({
              "feature_3d": feature_3d.numpy().tolist()
            , "label": label.numpy().tolist()

        })
        spark_df = spark.createDataFrame(pdf).withColumn("partition_id",
                                                                   F.when(F.rand() > 0.5, 1).otherwise(0))
        strategy = tf.distribute.OneDeviceStrategy("cpu:0")

        with strategy.scope():
            input_3d= tf.keras.Input(shape=[10,2,1], name="feature_3d", dtype="int32")
            dense = tf.keras.models.Sequential([
                 tf.keras.layers.Lambda(lambda x: __import__("tensorflow").reshape(x,[-1,20]),output_shape=(10*2,))
                ,tf.keras.layers.Dense(10,"relu")
                ,tf.keras.layers.Dense(20)
                ,tf.keras.layers.Lambda(lambda x: __import__("tensorflow").reshape(x,[-1,10,2,1]),output_shape=(10,2,1))

            ],name="label")

            model = KerasSparkModel(inputs=input_3d, outputs=dense(input_3d))
            model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(spark_df, batch_size=10, epochs=10, steps_per_epoch=10)

        pred_df = model.predict(spark_df)

        assert(
            list(tf.constant(pred_df.select("model_output.label").first()[0]).shape)==[10,2,1]
        )