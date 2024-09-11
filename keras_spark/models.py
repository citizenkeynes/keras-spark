import tensorflow as tf
from keras_spark.core import PetaStormReader,KerasOnSparkPredict,PlainPythonReader
import keras_spark
from pyspark.sql import DataFrame as SparkDataFrame
from typing import Any, Dict
import threading

class KerasSparkModel(tf.keras.Model):
    """
    A custom Keras model that extends the TensorFlow Keras Model to work with Apache Spark DataFrames.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the KerasSparkModel.

        :param args: Positional arguments for the base Keras Model.
        :param kwargs: Keyword arguments for the base Keras Model.
        """
        super(KerasSparkModel, self).__init__(*args, **kwargs)

    def fit(
        self,
        x: Any,
        y: Any = None,
        nr_partitions: int = None,
        nr_workers: int = 10,
        cache_path:str = None,
        postpro_fn=None,
        reader_type="PetaStormReader",
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        Fit the model to the data. If the data is a Spark DataFrame, convert it to a TensorFlow dataset before fitting.

        :param x: Input data. Must be a Sparkx DataFrame or other acceptable type for Keras models.
        :param y: Target data. Not used if x is a Spark DataFrame.
        :param partition_col: Column name used for partitioning the Spark DataFrame.
        :param nr_partitions: Number of partitions for the Spark DataFrame.
        :param num_parallel_calls: Number of parallel calls for data processing.
        :param kwargs: Additional arguments passed to the `fit` method of the Keras Model.
        """
        if isinstance(x, SparkDataFrame):

            dataset_container = []
            # Convert Spark DataFrame to TensorFlow dataset using SparkDsTFDs adapter
            def run_convert():
                adapter = getattr(keras_spark.core, "PetaStormReader")(self)
                dataset = adapter.convert(
                    x,
                    cache_path=cache_path,
                    nr_partitions=nr_partitions or x.rdd.getNumPartitions(),
                    nr_workers=nr_workers,
                    postpro_fn=postpro_fn,
                    batch_size=kwargs['batch_size']
                )
                dataset_container.append(dataset)

            # Create and start the thread
            convert_thread = threading.Thread(target=run_convert)
            convert_thread.start()

            # Optionally, wait for the thread to complete
            convert_thread.join()
            if dataset_container:
                super(KerasSparkModel, self).fit(dataset_container[0], **kwargs)
            else:
                raise Exception("conversion has failed")
        else:
            # Call the base class fit method directly if x is not a Spark DataFrame
            super(KerasSparkModel, self).fit(x, y, **kwargs)

    def predict(self, x: Any,use_pyarrow=True,maxRecordsPerBatch=5000,**kwargs: Dict[str, Any]) -> Any:
        """
        Make predictions using the model. If the input is a Spark DataFrame, use Spark for distributed prediction.

        :param x: Input data for prediction. Must be a Spark DataFrame or other acceptable type for Keras models.
        :param kwargs: Additional arguments passed to the `predict` method of the Keras Model.
        :return: Predictions made by the model.
        """
        if isinstance(x, SparkDataFrame):
            # Use KerasOnSparkPredict to make predictions on Spark DataFrame
            return KerasOnSparkPredict(use_pyarrow=use_pyarrow,maxRecordsPerBatch=maxRecordsPerBatch).predict(x, self)
        else:
            # Call the base class predict method directly if x is not a Spark DataFrame
            return super(KerasSparkModel, self).predict(x, **kwargs)
