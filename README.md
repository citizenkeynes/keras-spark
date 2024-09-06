# Spark-Keras Integration

This package enables seamless integration of PySpark DataFrames with Keras models, allowing users to efficiently train and predict using distributed data.

## Quickstart Guide

### Setting Up the PySpark DataFrame

Start by creating a PySpark DataFrame as shown below:

```python
import pandas as pd
from pyspark.sql import SparkSession
import tensorflow as tf

# Initialize a Spark session
spark = SparkSession.builder.appName("spark_keras").getOrCreate()

# Create a PySpark DataFrame
spark_df = spark.createDataFrame(
    pd.DataFrame({
        "feature1": tf.random.normal([100]).numpy().tolist(),
        "label1": tf.random.normal([100]).numpy().tolist(),
        "partition_id": [0 for _ in range(100)]
    })
)
```

### Training and Predicting with KerasSparkModel

You can fit and predict with `KerasSparkModel` using the standard Keras API:

```python
import tensorflow as tf
from keras_spark.models import KerasSparkModel as Model

# Define the Keras model
input_layer = tf.keras.Input(shape=[1], name="feature1")
output_layer = tf.keras.layers.Dense(1, name="label1")(input_layer)
model = Model(input_layer, output_layer)
model.compile("adam","mean_squared_error")

# Train the model using the PySpark DataFrame
model.fit(spark_df, batch_size=10, epochs=100,partition_col="partition_id")

# Use Spark for distributed scoring on the PySpark DataFrame
predictions = model.predict(spark_df).select("model_output.label1")
```

## Important Considerations

1. **Naming Conventions:**
    - *Input Names*: Each Keras input must have a specified name that corresponds to the respective PySpark DataFrame column.
    - *Output Names*: Output names are inferred from the Keras output layers and must match the PySpark column names if using `.fit()`.

2. **Data Type Compatibility:**
   - Ensure that the data types of Keras inputs and the corresponding PySpark columns are compatible.

3. **Partitioning Requirements:**
   - The PySpark DataFrame must include a `partition_id` column, with values ranging from `0` to `nr_partitions`.
   - Choose `nr_partitions` carefully to ensure that the Spark driver can handle the workload.
   - Parallel processing of partitions is handled using `.interleave()`, with the degree of parallelism set by `num_parallel_calls`.

4. **Prediction Output:**
   - The `.predict()` method generates an additional *struct* column named `model_output`.
   - To access specific outputs, reference them using their keys, e.g., `model_output.label1`.

5. **Keras Version:**
   - This package is compatible with Keras version 3.0 and above. 

This integration empowers users to leverage distributed data processing with PySpark while taking full advantage of Keras's deep learning capabilities.