from tensorflow import keras
import tensorflow as tf

class FileModel:
    def __init__(self, file_path):
        # self.model = keras.models.load_model(file_path)
        try:
            self.model = tf.keras.models.load_model(file_path)
        except:
            load_options = tf.saved_model.LoadOptions(experimental_io_device= '/job:localhost')
            self.model = tf.saved_model.load(file_path, options= load_options)
    def fit(self, train_x, train_y, test_x, test_y, epoch =35, batch_size=100, optimizer= tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy',  metrics=['accuracy']):
        self.model.compile(optimizer,
             loss,
             metrics)
        return self.model.fit(train_x, train_y,
          validation_data=(test_x, test_y),
          epochs = epoch,
          batch_size=batch_size
        )
    
    def __predict__(self, x):
        if (len(x)<=1):
            return self.model(x)
        return self.model.predict(x)

    def predict (self, x):
        return self.__predict__(x)

    def evaluate(self, x, y, batch_size=1):
        return self.model.evaluate(x, y, batch_size)