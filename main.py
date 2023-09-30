import pandas as pd
import tensorflow as tf
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import joblib


def build_model(hp):
    model = tf.keras.Sequential()
    for i in range(hp.Int('num_layers', 1, 11)):
        model.add(tf.keras.layers.Dense(units=hp.Int(
            'units', min_value=10, max_value=100, step=10),
            activation=hp.Choice("activation", ["relu", "selu"])))
    model.add(tf.keras.layers.Dense(1, activation=hp.Choice(
        "activation", ["relu", "none"])))
    learning_rate = hp.Float("lr", min_value=1e-4,
                             max_value=1e-0, sampling="log")
    model.compile(loss=hp.Choice('loss', [
                  "mse", "mae"]), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['mse'])

    return model


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

build_model(kt.HyperParameters())
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="mse",
    seed=42,
    max_trials=1000,
    executions_per_trial=1,
    overwrite=True,
    directory="keras_tuner",
    project_name="ml_scheduler",
)

tuner.search(X_train_normal, y_train, epochs=10,
             validation_data=(X_test_normal, y_test))
tuner.search_space_summary()

# Get optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
with open('best_hps.txt', 'w') as f:  # Write results to file
    print('Number of layers: ' + str(best_hps.get('num_layers')) + '\n')
    f.write('Number of layers: ' + str(best_hps.get('num_layers')) + '\n')
    print('Number of neurons: ' + str(best_hps.get('units')) + '\n')
    f.write('Number of neurons: ' + str(best_hps.get('units')) + '\n')
    print('Activation function: ' + best_hps.get('activation') + '\n')
    f.write('Activation function: ' + best_hps.get('activation') + '\n')
    print('Learing rate of optimizer function: ' +
          str(best_hps.get('lr')) + '\n')
    f.write('Learing rate of optimizer function: ' +
            str(best_hps.get('lr')) + '\n')
    print('Loss function: ' + best_hps.get('loss') + '\n')
    f.write('Loss function: ' + best_hps.get('loss'))

# Best model
model = tuner.hypermodel.build(best_hps)

# Finding optimal amount of epochs
history = model.fit(X_train_normal, y_train, epochs=500)
val_mse_per_epoch = history.history['mse']
best_epoch = val_mse_per_epoch.index(min(val_mse_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# Save model
model.save('ml_scheduler_model_kt.keras')
