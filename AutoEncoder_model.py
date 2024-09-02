
input_data2 = keras.Input(shape=(16,))
#encoder2 = (BatchNormalization())(input_data2)
encoder0= (Dropout(0.3))(input_data2)
encoder1 = layers.Dense(12, activation='relu')(encoder0)
encoder2 = layers.Dense(8, activation='relu')(encoder1)

code1 = layers.Dense(2, activation='relu')(encoder2)
decoded1 = layers.Dense(8, activation='relu')(code1)
decoded2 = layers.Dense(12, activation='relu')(decoded1)
output_data2 = layers.Dense(16, activation='sigmoid')(decoded2)

# This model maps an input to its reconstruction
model6 = keras.Model(input_data2, output_data2)


# the optimizer
# the optimizer
sgd = SGD(lr=0.001, momentum=0.99999)
model6.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])