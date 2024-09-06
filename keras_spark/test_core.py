from unittest import TestCase
from keras_spark.core import *
import tensorflow as tf
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


class TestKerasOnSparkPredict(TestCase):

    def test_infer_schema_from_keras(self):

        class IntOp(tf.keras.layers.Layer):

            def __init__(self,name):
                super().__init__(name = name)
            def call(self, inputs):
                return tf.cast(inputs,"int32")


        i = tf.keras.layers.Input([1])
        m = tf.keras.models.Model(i, {"a_float": i, "an_int": IntOp("an_int")(i)})
        a = KerasOnSparkPredict().infer_output_schema_from_keras(m)
        assert (a == 'struct<a_float: array<float>,an_int: array<int>>')

    def test_keras_output_to_dict_int(self):
        i = tf.keras.layers.Input([1])
        cast = tf.keras.layers.Lambda(lambda x: tf.cast(x, "int32"), name="op1")
        m = tf.keras.models.Model(i, cast(i))
        assert (list(KerasOnSparkPredict().keras_output_to_dict(m).output.keys()) == ["op1"])

    def test_keras_output_to_dict_int_float(self):
        i = tf.keras.layers.Input([1])
        cast1 = tf.keras.layers.Lambda(lambda x: tf.cast(x, "int32"), name="op1")
        cast2 = tf.keras.layers.Lambda(lambda x: tf.cast(x, "float32"), name="op2")
        m = tf.keras.models.Model(i, [cast1(i), cast2(i)])
        assert (list(KerasOnSparkPredict().keras_output_to_dict(m).output.keys()) == ["op1", "op2"])


class TestPlainReader(TestCase):

    def test_convert_tensor_3d(self):

        pdf = pd.DataFrame({
             "feature1": tf.random.normal([10,10,2]).numpy().tolist()
            ,"op1": tf.random.normal([10,10,2]).numpy().tolist()
        })

        i = tf.keras.layers.Input([10,2],name="feature1")
        cast1 = tf.keras.layers.Lambda(lambda x: tf.cast(x, "int32"), name="op1")
        m = tf.keras.models.Model(i, cast1(i))

        adapter = PlainPythonReader(model=m)

        result =adapter.convert(spark.createDataFrame(pdf),nr_partitions=1,batch_size=5,nr_workers=10,)

        assert(list(next(iter(result))[0]["feature1"].shape)==[5,10,2])
        assert (list(next(iter(result))[1].shape) == [5, 10, 2])

    def test_pandas_to_tensor_3d(self):

        pdf = pd.DataFrame({
             "feature1": tf.random.normal([10,10,2]).numpy().tolist()
            ,"op1": tf.random.normal([10,10,2]).numpy().tolist()
        })

        i = tf.keras.layers.Input([10,2],name="feature1")
        cast1 = tf.keras.layers.Lambda(lambda x: tf.cast(x, "int32"), name="op1")
        m = tf.keras.models.Model(i, cast1(i))

        adapter = PlainPythonReader(model=m)

        result =adapter.pandas_to_tensor_dict(pdf,as_dict=True)

        assert(result["feature1"].shape==[10,10,2])
    def test_pandas_to_tensor_dict(self):

        pdf = pd.DataFrame({
             "feature1": tf.random.normal([200]).numpy().tolist()
            ,"op1": tf.random.normal([200]).numpy().tolist()
            ,"op2": tf.random.normal([200]).numpy().tolist()
        })

        i = tf.keras.layers.Input([1],name="feature1")
        cast1 = tf.keras.layers.Lambda(lambda x: tf.cast(x, "int32"), name="op1")
        cast2 = tf.keras.layers.Lambda(lambda x: tf.cast(x, "float32"), name="op2")
        m = tf.keras.models.Model(i, [cast1(i), cast2(i)])

        adapter = PlainPythonReader(model=m)

        result =adapter.pandas_to_tensor_dict(pdf,as_dict=True)

        assert(list(result.keys())==["feature1","op1","op2"])

    def test_pandas_to_tensor_dict_str_input(self):

        pdf = pd.DataFrame({
             "feature1": [["abc","def"] for i in range(10)]
            ,"op1": [["abc","def"] for i in range(10)]
        })

        i = tf.keras.layers.Input([2],name="feature1",dtype="string")

        class STROp(tf.keras.layers.Layer):

            def __init__(self):
                super().__init__(name = "op1")
            def call(self, inputs):
                return tf.cast(inputs,"string")
        m = tf.keras.models.Model(i, STROp()(i))

        adapter = PlainPythonReader(model=m)

        result =adapter.pandas_to_tensor_dict(pdf,as_dict=True)

        assert(result["feature1"][0][0].numpy()==bytes('abc',"utf8"))

        assert (list(result["feature1"].shape) == [10,2])

        assert (result["op1"][0][1].numpy() ==bytes('def',"utf8"))

        assert(list(result.keys())==["feature1","op1"])

    def test_pandas_to_tensor_dict_str_input_dictout(self):

        pdf = pd.DataFrame({
             "feature1": [["abc","def"] for i in range(10)]
            ,"op1": [["abc","def"] for i in range(10)]
            , "op2": [["abc", "xyz"] for i in range(10)]
        })

        i = tf.keras.layers.Input([2],name="feature1",dtype="string")

        class STROp(tf.keras.layers.Layer):

            def __init__(self,name):
                super().__init__(name = name)
            def call(self, inputs):
                return tf.cast(inputs,"string")
        m = tf.keras.models.Model(i, {"a":STROp("op1")(i),"b":STROp("op2")(i)})

        adapter = PlainPythonReader(model=m)

        result =adapter.pandas_to_tensor_dict(pdf,as_dict=True)

        assert (result["op1"][0][1].numpy() ==bytes('def',"utf8"))

        assert (result["op2"][0][1].numpy() ==bytes('xyz',"utf8"))


    def test_partitioning(self):

        feat1 = tf.random.normal([200]).numpy().tolist()
        op1 = tf.random.normal([200]).numpy().tolist()

        pdf = pd.DataFrame({
            "feature1": feat1
            , "op1": op1
            , "op2": tf.random.normal([200]).numpy().tolist()
            , "partition_id":  tf.reshape(tf.stack([[i]*20 for i in range(10)]),[-1]).numpy().tolist()
        })

        i = tf.keras.layers.Input([1], name="feature1")
        cast1 = tf.keras.layers.Lambda(lambda x: tf.cast(x, "float32"), name="op1")
        cast2 = tf.keras.layers.Lambda(lambda x: tf.cast(x, "float32"), name="op2")
        m = tf.keras.models.Model(i, [cast1(i), cast2(i)])

        dataset = PlainPythonReader(model=m)\
            .convert(spark.createDataFrame(pdf),partition_col="partition_id",nr_partitions=10,batch_size=10)

        feature_1_r,op1_r = [],[]
        batch_counter=0

        for x,y in iter(dataset):
            feature_1_r+= tf.squeeze(x["feature1"]).numpy().tolist()
            op1_r +=  tf.squeeze(y["op1"]).numpy().tolist()
            batch_counter += 1

        assert (batch_counter==20)
        assert (sorted(feature_1_r)==sorted(pdf["feature1"].tolist()))
        assert (sorted(op1_r) == sorted(pdf["op1"].tolist()))

class TestPetaStormReader(TestCase):

    def test_pandas_to_tensor_1d(self):

        pdf = pd.DataFrame({
             "feature1": tf.random.normal([100,10]).numpy().tolist()
            ,"op1": tf.random.normal([100,10]).numpy().tolist()
        })

        i = tf.keras.layers.Input([10],name="feature1")
        cast1 = tf.keras.layers.Lambda(lambda x: tf.cast(x, "int32"), name="op1")
        m = tf.keras.models.Model(i, cast1(i))

        adapter = PetaStormReader(model=m)

        df =adapter.convert(spark.createDataFrame(pdf),nr_partitions=1,batch_size=64)

        assert (next(iter(df))[1].shape[0]==64)

    def test_pandas_to_tensor_3d(self):
            pdf = pd.DataFrame({
                "feature1": tf.random.normal([100, 10,2]).numpy().tolist()
                , "op1": tf.random.normal([100, 10,2]).numpy().tolist()
            })

            i = tf.keras.layers.Input([10, 2], name="feature1")
            cast1 = tf.keras.layers.Lambda(lambda x: tf.cast(x, "int32"), name="op1")
            m = tf.keras.models.Model(i, cast1(i))

            adapter = PetaStormReader(model=m)

            df = adapter.convert(spark.createDataFrame(pdf), nr_partitions=1, batch_size=64)

            assert (list(next(iter(df))[0]["feature1"].shape)== [64,10,2])
            assert (list(next(iter(df))[1].shape) == [64, 10, 2])

    def test_pandas_to_tensor_4d(self):
        pdf = pd.DataFrame({
            "feature1": tf.random.normal([100, 10, 2, 2]).numpy().tolist()
            , "op1": tf.random.normal([100, 10]).numpy().tolist()
        })

        i = tf.keras.layers.Input([10, 2, 2], name="feature1")
        res = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1,10*2*2]), name="res")
        m = tf.keras.models.Model(i, tf.keras.layers.Dense(10,name="op1")(res(i)))

        adapter = PetaStormReader(model=m)

        df = adapter.convert(spark.createDataFrame(pdf), nr_partitions=1, batch_size=64)

        assert (list(next(iter(df))[0]["feature1"].shape) == [64, 10, 2, 2])
        assert (list(next(iter(df))[1].shape) == [64, 10])