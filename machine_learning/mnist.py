import keras
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=100))
model.add(keras.layers.Dense(units=10, activation='softmax'))
data = keras.datasets.mnist
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
(x_train,y_train),(x_test,y_test) = data.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model.fit(x_train, y_train, validation_data=(x_test, y_test) ,batch_size=1000,epochs=20)