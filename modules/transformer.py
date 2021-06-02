import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

#Encoder/Transformer functionality code is build upon https://www.tensorflow.org/tutorials/text/transformer#encoder_layer

#simple feed forward network
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])
                
#encoder layer
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1, doMask = False, seed_value = 42):
    super(EncoderLayer, self).__init__()
    self.mha =  tfa.layers.MultiHeadAttention(d_model, num_heads, dropout = rate, return_attn_coef=True)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.num_heads = num_heads

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    
    self.profiling = True
    self.doMask = doMask
    

    
    self.dropout1 = tf.keras.layers.Dropout(rate, seed=seed_value)
    self.dropout2 = tf.keras.layers.Dropout(rate, seed=seed_value)
    self.dropout3 = tf.keras.layers.Dropout(rate, seed=seed_value)
    
    self.lstm = tf.keras.layers.LSTM(dff, return_sequences=False)
    self.flatten = tf.keras.layers.Flatten()
    self.preOut = tf.keras.layers.Dense(dff)
    
    # Shape => [batch, time, features]
    self.out = tf.keras.layers.Dense(1)
    
  def build(self, input_shape):
    print(input_shape)
    
  def call(self, x, training):
    
    #print(x1)
    if self.doMask:
        x1, mask, adminSum = x
        print("aaaaa")
        print(mask)
        attn_output, attention = self.mha([x1, x1, x1], mask = mask)  # (batch_size, input_seq_len, d_model)
    else:
        x1, adminSum = x
        attn_output, attention = self.mha([x1, x1, x1])  # (batch_size, input_seq_len, d_model)
    out1 = self.layernorm1(x1 + attn_output)  # (batch_size, input_seq_len, d_model)

    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1  + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return (out2, adminSum, attention)
        
#class which can represent multiple encoder layers with correct input and output handling
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1, input_vocab_size = 10000, maxLen = None, doMask=False, seed_value=42):
    super(Encoder, self).__init__()

    self.num_heads = num_heads
    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, input_length= maxLen)
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, doMask=doMask, seed_value=seed_value) for z in range(num_layers)]
   
    self.dropout = tf.keras.layers.Dropout(rate, seed=seed_value)
    
    self.doMask = doMask
    
  def build(self, input_shape):
    if self.doMask:
        self.attention = tf.Variable(tf.ones((self.num_heads, input_shape[0][1], input_shape[0][1])), trainable=False, validate_shape=True, name='attentionMat')
    else: 
        self.attention = tf.Variable(tf.ones((self.num_heads, input_shape[1], input_shape[1])), trainable=False, validate_shape=True, name='attentionMat')

    print(input_shape)
    print('#################')
    self.adminSumer =  0
    
  def call(self, xa, training):
    if self.doMask:
        x, mask = xa
    else:
        x = xa

    seq_len = tf.shape(x)[1]
        
    x += self.pos_encoding[:, :seq_len, :]
    
    if self.doMask:
        xF = (x, mask, self.adminSumer)
    else:
        xF = (x, self.adminSumer)
    print(x.shape)
    print(mask.shape)
    fullAttention = []
    for i in range(self.num_layers):
        xF = self.enc_layers[i](xF, training)
        x, self.adminSumer, attention = xF
        xF = x, self.adminSumer
        fullAttention.append(attention)

    attention = tf.math.reduce_mean(fullAttention, axis=0)
    
    return x, attention, fullAttention  # (batch_size, input_seq_len, d_model)

  def initPhase(self):
    for i in range(self.num_layers):
        #self.sumer = self.enc_layers[i].initPhase(self.sumer)
        self.enc_layers[i].profiling = False
        

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

#safes the "best" model to load it later
# safe is based on highest val acc with the smalles val loss
#probably not optimal but doing fine
class SaveBest(tf.keras.callbacks.Callback):
    
    def __init__(self, weightsName):
        self.highestAcc = 0
        self.loss = -1
        self.weightsName = weightsName
        
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_accuracy'] >= self.highestAcc and (self.loss is -1 or logs['val_loss'] < self.loss):
            print('#########++++##########')
            self.highestAcc = logs['val_accuracy']
            self.loss = logs['val_loss']
            self.model.save_weights(self.weightsName, overwrite=True)

#custom scheduler with watp up steps
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=10000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
            
            
