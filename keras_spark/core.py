from pyspark.sql import functions as F
from collections.abc import Iterator
from pyspark.sql import SparkSession
from pyspark import SparkFiles
import collections
import shutil
import zipfile
from collections.abc import Iterable
import pandas as pd
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType, FloatType, StringType, DoubleType
import os
from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
import tensorflow as tf
import tempfile
from abc import ABC, abstractmethod

home_path = os.path.expanduser( '~' )

class Reader(ABC):

    def __init__(self,model):

        def unify_input(keras_input):
            if type(keras_input) == list:
                return keras_input
            elif type(keras_input) == dict:
                return keras_input.values()
            else:
                return [keras_input]

        self.model = model
        self.input_names =  [layer.name for layer in unify_input(self.model.input)]
        self.output_names = list(self.model.output_names)
        self.shape_dict,self.type_dict = self.get_dicts()

    def get_dicts(self):
        shape_dict={}
        type_dict={}
        for name in self.input_names:

            if hasattr(self.model.get_layer(name),"_input_tensor"): #keras 3
                shape_dict[name] = self.model.get_layer(name)._input_tensor.shape[1:]
            else:
                shape_dict[name] = self.model.get_layer(name).input.shape[1:]

            type_dict[name] = self.model.get_layer(name).dtype

        for name in self.output_names:
            shape_dict[name] = self.model.get_layer(name).output.shape[1:]
            type_dict[name] = self.model.get_layer(name).output.dtype
        return (shape_dict,type_dict)

    def save(self, spark_df, cache_path, nr_partitions):
        cache_path = cache_path or "file://" + home_path

        temp_name = next(tempfile._get_candidate_names())
        save_path = os.path.join(cache_path,temp_name)

        if nr_partitions is None:
            nr_partitions = spark_df.rdd.getNumPartitions()

        spark_df \
            .repartition(nr_partitions) \
            .select(*(self.input_names + self.output_names)) \
            .write.mode("overwrite").parquet(save_path)

        parquet_files = spark_df.sql_ctx.read.parquet(save_path)\
            .select(F.input_file_name()) \
            .distinct() \
            .rdd.map(lambda row: row[0]) \
            .collect()

        return [f for f in parquet_files]

    @abstractmethod
    def convert(self,spark_df,nr_partitions=1,batch_size=32,nr_workers=10,postpro_fn=None,partition_col="partition_id",cache_path=None):
        pass

class PetaStormReader(Reader):

    def __init__(self,model):
        super(PetaStormReader, self).__init__(model)


    def flatten_arrays_to_1d(self,df):

        #FIXME: fetch a row - and make sure dimensions are aligned with model inputs' dimensions
        def flatten(x):
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                result = []
                for i in x:
                    result.extend(flatten(i))
                return result
            else:
                return [x]
        def get_primitive(type):
            current = type
            while isinstance(current, ArrayType):
                current = current.elementType
            return current.simpleString()

        schema = df.schema
        for field in schema.fields:
            if isinstance(field.dataType, ArrayType):
                primitive_type = get_primitive(field.dataType)
                df = df.withColumn(field.name,F.udf(f"array<{primitive_type}>")(flatten)(F.col(field.name)))
        return df

    def convert(self,spark_df,nr_partitions=1,batch_size=32,nr_workers=10,postpro_fn=None,partition_col="partition_id",cache_path=None):

        files = self.save(self.flatten_arrays_to_1d(spark_df),cache_path=cache_path,nr_partitions=nr_partitions)

        reader = make_batch_reader(files,workers_count=nr_workers)
        dataset = make_petastorm_dataset(reader)
        colmapping = {name:i for i,name in enumerate(dataset.element_spec._fields)}

        if postpro_fn is None:
            postpro_fn = lambda dataset: dataset\
                .prefetch(tf.data.AUTOTUNE)

        dataset = (dataset
                    .apply(tf.data.experimental.unbatch()).batch(batch_size)
                    .map(lambda all: ({name:all[colmapping[name]] for name in self.input_names}
                                     ,{name:all[colmapping[name]] for name in self.output_names}
                            )
                         , num_parallel_calls=tf.data.AUTOTUNE)
                    .map(lambda x, y: ({k: tf.reshape(v, tf.constant([-1]+list(self.shape_dict[k]))) for k, v in x.items()},
                                       {k: tf.reshape(v, tf.constant([-1]+list(self.shape_dict[k]))) for k, v in y.items()}
                                       ), num_parallel_calls=tf.data.AUTOTUNE)
                   .map(lambda x, y: (x, (y[list(y.keys())[0]]) if len(y) == 1 else y), num_parallel_calls=tf.data.AUTOTUNE)
                   )

        return postpro_fn(dataset)


class PlainPythonReader(Reader):
    def __init__(self,model):
        super(PlainPythonReader, self).__init__(model)

    def pandas_to_tensor_dict(self, pandas_df, as_dict=False, only_inputs=False):
        op = {}
        input_names = self.input_names
        output_names = self.output_names
        if only_inputs:
            output_names = []

        def extract_tensor(column_name):
            tensor = tf.constant(pandas_df[column_name].values.tolist())
            reshaped = tf.reshape(tensor, [-1] + list(self.shape_dict[name]))  # Preserve batch size
            return tf.cast(reshaped, self.type_dict[name])

        for name in input_names + output_names:
            assert name in pandas_df.columns, f"column {name} isn't contained in dataset"
            op[name] = extract_tensor(name)

        if as_dict:
            return op
        else:
            return tuple([op[name] for name in input_names + output_names])

    def convert(self,spark_df,nr_partitions=1,batch_size=32,nr_workers=10,postpro_fn=None,partition_col="partition_id",cache_path=None):

        files = self.save(spark_df, cache_path=cache_path, nr_partitions=nr_partitions)

        if postpro_fn is None:
            postpro_fn = lambda ds: ds \
                .repeat(-1)\
                .prefetch(tf.data.AUTOTUNE)

        def get_generator(partition_id):
            pdf = pd.read_parquet(files[partition_id])
            tensor_list =  self.pandas_to_tensor_dict(pdf)
            yield tensor_list

        signature =  tuple(
            [tf.TensorSpec(shape=tf.TensorShape([None]+self.shape_dict[name]), dtype=self.type_dict[name])
                for name in self.input_names + self.output_names]
        )

        dataset = (
            tf.data.Dataset.range(len(files))
            .interleave(
                lambda pid: tf.data.Dataset.from_generator(get_generator,args=(pid,),output_signature=signature)
                .unbatch()
                .map(lambda *flist: ({n: flist[i] for i, n in enumerate(self.input_names)},
                                 {n: flist[len(self.input_names) + i] for i, n in enumerate(self.output_names)}
                                 ), num_parallel_calls=tf.data.AUTOTUNE)
                .map(lambda x, y: ({k: tf.reshape(v, self.shape_dict[k]) for k, v in x.items()},
                               {k: tf.reshape(v, self.shape_dict[k]) for k, v in y.items()}
                               ), num_parallel_calls=tf.data.AUTOTUNE)
                , cycle_length=nr_workers
                , num_parallel_calls=tf.data.AUTOTUNE
                , deterministic=False
            )
            .map(lambda x, y: (x, (y[list(y.keys())[0]]) if len(y) == 1 else y), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            )

        return postpro_fn(dataset)

class KerasOnSparkPredict:

    def __init__(self,use_pyarrow,maxRecordsPerBatch):
        self.use_pyarrow = use_pyarrow
        self. maxRecordsPerBatch=maxRecordsPerBatch

    def infer_output_schema_from_keras(self,model):

        def map_keras_dtype_to_spark(dtype):
            """Maps Keras output data types to corresponding Spark data types."""
            if dtype == tf.int32:
                return IntegerType()
            elif dtype == tf.float32:
                return FloatType()
            elif dtype == tf.float64:
                return DoubleType()
            elif dtype == tf.string:
                return StringType()
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

        output_dict = model.output
        schema_fields = []

        for key, tensor in output_dict.items():
            spark_type = map_keras_dtype_to_spark(tensor.dtype)
            if len(tensor.shape) > 1:
                for _ in range(1,len(tensor.shape)):
                    spark_type = ArrayType(spark_type)
            schema_fields.append(StructField(key, spark_type, nullable=False))

        schema = StructType(schema_fields)
        type_string = f"struct<{','.join([f'{field.name}: {field.dataType.simpleString()}' for field in schema.fields])}>"

        return type_string


    def keras_output_to_dict(self,model):

        output_names = model.output_names

        if type(model.output) == dict:
            return model
        elif type(model.output) == list:
            return tf.keras.models.Model(model.input,{name:output for output,name in zip(model.output,output_names)})
        elif tf.keras.backend.is_keras_tensor(model.output):
            return tf.keras.models.Model(model.input,{name:output for output,name in zip([model.output],output_names)})
        else:
            raise Exception("this keras model has an unknown output type ")

    def predict(self, spark_df, model, use_spark_files=True):
        # Convert Keras model to a dictionary if needed
        model = self.keras_output_to_dict(model)
        spark = SparkSession.builder.getOrCreate()

        # Enable/Disable PyArrow for Spark
        if self.use_pyarrow:
            spark.conf.set("spark.sql.execution.arrow.enabled", "true")
            spark.conf.set('spark.sql.execution.arrow.maxRecordsPerBatch', self.maxRecordsPerBatch)
        else:
            spark.conf.set("spark.sql.execution.arrow.enabled", "false")

        # Generate a temporary file name for saving the model
        temp_file_name = next(tempfile._get_candidate_names())
        mpath = f"{temp_file_name}"
        model.save(mpath)

        # Create a ZIP file for the saved model directory
        zip_model_path = f"{mpath}.zip"
        shutil.make_archive(mpath, 'zip', mpath)

        if use_spark_files:
            # Add the ZIP file to Spark
            spark.sparkContext.addFile(zip_model_path, recursive=True)
        else:
            # Save the model structure and weights in-memory for non-spark-file usage
            mweights = model.get_weights()
            mjson = model.to_json()

        def raw_predict(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            # Unzip the model before loading it
            if use_spark_files:
                zipped_model_path = SparkFiles.get(f"{temp_file_name}.zip")
                with zipfile.ZipFile(zipped_model_path, 'r') as zip_ref:
                    extracted_path = tempfile.mkdtemp()
                    zip_ref.extractall(extracted_path)

                model = tf.keras.models.load_model(extracted_path, custom_objects={'tf': tf})
            else:
                model = tf.keras.models.model_from_json(mjson, custom_objects={'tf': tf})
                model.set_weights(mweights)

            reader = PlainPythonReader(model)

            for pdf_full in iterator:
                td = reader.pandas_to_tensor_dict(pdf_full, as_dict=True, only_inputs=True)
                dict_output = model(td)
                yield pd.DataFrame({k: v.numpy().tolist() for k, v in dict_output.items()})

        # Infer output schema based on the Keras model
        type_str = self.infer_output_schema_from_keras(model)
        reader = PlainPythonReader(model)
        col_struct = F.struct(*reader.input_names)

        # Apply the UDF
        op = spark_df.withColumn("model_output", F.pandas_udf(type_str)(raw_predict)(col_struct)).cache()
        return op
