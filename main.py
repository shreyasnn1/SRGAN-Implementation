import utils
import tensorflow as tf
import model
import keras

def main():
    
    gan_model = model.SRGAN()
    epochs = 1000
    batch_size = 10
    input_path = 'coco_small_dataset/val2017'
    model_save_dir = 'model_weights'
    output_dir = 'output'
    gan_model.train(epochs, batch_size,model_save_dir, input_path, output_dir)
    
    print("Training Done!")
    
if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_physical_devices('GPU') # ConfigProto is outdated in TensorFlow v2
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    main()