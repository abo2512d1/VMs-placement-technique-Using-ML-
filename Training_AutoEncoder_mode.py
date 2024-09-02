from Auto_Encoder_SVM_model_code import *
import CommonDirectionAndViraibles
from AutoEncoder_model import *
# run the model
# run the model
history6=model6.fit(X_train6, X_train6,
                epochs=500,
                batch_size=64,
                shuffle=True,
                validation_data=(x_validation6, x_validation6))



"""" saving the model """

model6.save('AutoencoderForExtractingFeatureFrom5mintsJob2FeaturesThebest.h5')


"saving the encoder part of the model"


model7 = keras.Model(input_data2, code1)

model7.save('encoderPartOfthemodelThebest8features.h5')

"""loading the model """


model7=load_model('encodersPartOfThemodelTheBestforCode5.h5')