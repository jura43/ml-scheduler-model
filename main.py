import pandas as pd
import tensorflow as tf
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('ml_scheduler_dataset.csv')
data.dropna(inplace=True)  # Remove records with missing values
data = data[data['frontend_cpu_usage'] <= 1]  # Remove values grater than 1
data = data[data['database_cpu_usage'] <= 1]
data = data[data['backend_cpu_usage'] <= 1]
# Remove outliers
Q1 = data['response_time'].quantile(0.25)
Q3 = data['response_time'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['response_time'] >= lower_bound) &
            (data['response_time'] <= upper_bound)]

print("----------- Checking issing vaules -----------")
print(data.isnull().any())
print("----------- Describe dataset -----------")
print(data.describe())


X = data.drop('response_time', axis=1)
y = data['response_time']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# frontend,backend,database,response_time,frontend_cpu_usage,frontend_memory_usage,frontend_pods,frontend_ssd,backend_cpu_usage,backend_memory_usage,backend_pods,backend_ssd,database_cpu_usage,database_memory_usage,database_pods,database_ssd
ct = make_column_transformer((MinMaxScaler(), ['frontend_cpu_usage', 'frontend_memory_usage', 'frontend_pods', 'backend_cpu_usage', 'backend_memory_usage', 'backend_pods', 'database_cpu_usage', 'database_memory_usage', 'database_pods',]),
                             (OneHotEncoder(handle_unknown="ignore"), ['frontend', 'backend', 'database', 'frontend_ssd', 'backend_ssd', 'database_ssd']))
ct.fit(X_train)
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)
joblib.dump(ct, "ct.save")

tf.random.set_seed(42)

data_model = tf.keras.Sequential([
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
])

data_model.compile(loss=tf.keras.losses.mse,
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.00038421108389847386), metrics=['mse'])

data_model.fit(X_train_normal, y_train, epochs=5000)
data_model.summary()
data_model.save('ml_scheduler_model.keras')
tf.keras.utils.plot_model(model=data_model, show_shapes=True)

# Evaluting model
y_preds = data_model.predict(X_test_normal[:10])
print(y_preds)
