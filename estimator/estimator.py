import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input,Conv2D,Flatten,Dense,Activation,Conv2DTranspose,Reshape,BatchNormalization,Activation
from tensorflow.keras.models import Model



class WGAN_GP(tf.keras.Model):    
    """WGAN-GP model. Creates both generator and critic, and trains them"""
    def __init__(self,z_dim,critic_steps,gp_weight):
        super().__init__()
        self.z_dim = z_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight      
        self.critic = self.create_critic()
        self.generator = self.create_generator()
    
    def compile(self, critic_optimizer, generator_optimizer):
        super().compile()
        self.critic_optimizer = critic_optimizer
        self.generator_optimizer = generator_optimizer       
    
    def create_critic(self):
        critic_input = Input(shape = (48,48,3), name = "input_critic") 
        
        critic_x = Conv2D(512,kernel_size = 3,activation = 'linear', name = "Conv2D_critic_4")(critic_input)
        critic_x = Activation("relu",name = "activation_critic_4")(critic_x)
        
        critic_x = Conv2D(256,kernel_size = 3,activation = 'linear', name = "Conv2D_critic_5")(critic_input)
        critic_x = Activation("relu",name = "activation_critic_5")(critic_x)
        
        critic_x = Conv2D(128,kernel_size = 3,activation = 'linear', name = "Conv2D_critic_6")(critic_x)
        critic_x = Activation("relu",name = "activation_critic_6")(critic_x)
        
        critic_x = Conv2D(64,kernel_size = 3,activation = 'linear', name = "Conv2D_critic_7")(critic_x)
        critic_x = Activation("relu",name = "activation_critic_7")(critic_x)
        
        critic_x = Conv2D(64,kernel_size = 3,activation = 'linear', name = "Conv2D_critic_8")(critic_x)
        critic_x = Activation("relu",name = "activation_critic_8")(critic_x)
        
        critic_x = Flatten(name = "flatten_critic")(critic_x)
        
        critic_x = Dense(150,activation = 'relu',name = "dense_critic_1")(critic_x)   
        critic_x = Dense(50,activation = 'relu',name = "dense_critic_2")(critic_x)     
        critic_output  = Dense(1,activation = 'linear', name = "dense_critic_output")(critic_x)
        
        critic = Model(inputs = [critic_input] , outputs = [critic_output])    
        return critic        
        
    def create_generator(self):    
        random_latent_vectors_input = Input(shape=(self.z_dim,), name = "input_generator")
        generator_x = Dense(576, activation = "linear",name = "dense_generator")(random_latent_vectors_input)
        generator_x = Reshape(target_shape = (3,3,64)) (generator_x)

        generator_x = Conv2DTranspose(128,(3,3),(2,2),padding = "same", name = "Conv2DTranspose_generator_1")(generator_x)
        generator_x = BatchNormalization(name = "batch_norm_generator_1")(generator_x)
        generator_x = Activation("relu",name = "activation_generator_1")(generator_x)

        generator_x = Conv2DTranspose(128,(3,3),(2,2),padding = "same", name = "Conv2DTranspose_generator_2")(generator_x)
        generator_x = BatchNormalization(name = "batch_norm_generator_2")(generator_x)
        generator_x = Activation("relu",name = "activation_generator_2")(generator_x)

        generator_x = Conv2DTranspose(32,(3,3),(2,2),padding = "same", name = "Conv2DTranspose_generator_3")(generator_x)
        generator_x = BatchNormalization(name = "batch_norm_generator_3")(generator_x)
        generator_x = Activation("relu",name = "activation_generator_3")(generator_x)

        generator_x = Conv2DTranspose(4,(3,3),(2,2),padding = "same", name = "Conv2DTranspose_generator_4")(generator_x)
        generator_x = BatchNormalization(name = "batch_norm_generator_4")(generator_x)
        generator_x = Activation("relu",name = "activation_generator_4")(generator_x)

        generator_x = Conv2DTranspose(3,(3,3),(1,1),padding = "same", name = "Conv2DTranspose_generator_output")(generator_x)
        generator_x = BatchNormalization(name = "batch_norm_generator_output")(generator_x)
        generator_output = Activation("sigmoid",name = "activation_generator_output")(generator_x)       

        generator = Model(inputs = [random_latent_vectors_input], outputs = [generator_x], name = "generator")
        return generator     
    
    def gradient_penalty(self,interpolated):  
    
        with tf.GradientTape() as gp_tape:
            
            gp_tape.watch(interpolated)      #tensors arent watch by default, only variables. picture is a tensor
            critic_prediction = self.critic(interpolated)
        
        grads = gp_tape.gradient(critic_prediction, [interpolated])[0]   
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def train_critic(self,batch_input_shape,real_images,fake_images,interpolated_images):
        
        with tf.GradientTape() as c_tape:   

            critic_real_target = tf.ones((batch_input_shape,1), dtype=tf.float32)  
            critic_fake_target = -tf.ones((batch_input_shape,1), dtype=tf.float32)  

            critic_real_pred = self.critic(real_images)
            critic_fake_pred = self.critic(fake_images)

            critic_real_loss = wasserstein(critic_real_target, critic_real_pred)
            critic_fake_loss = wasserstein(critic_fake_target, critic_fake_pred)        
            gp = self.gradient_penalty(interpolated_images)

            critic_loss = critic_real_loss + critic_fake_loss +  self.gp_weight * gp 
       
        critic_gradients = c_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
           
        return  critic_loss,critic_real_loss,critic_fake_loss,gp        
    
    def train_generator(self,batch_input_shape,vector):
        
        generator_target =tf.ones((batch_input_shape,1), dtype=tf.float32)     
        
        with tf.GradientTape() as g_tape:               
            generated_images = self.generator(vector)
            critic_prediction = self.critic(generated_images)
            generator_loss = wasserstein(generator_target, critic_prediction)
        generator_gradients = g_tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        
        return generator_loss
    
    def train_step(self, data):
        
        batch_input_shape = tf.shape(data, name = "WGAN_GP_batch_size")[0]
        interpolation_layer = Interpolate("interpolation_layer")
        
        real_images = data
        random_latent_vectors = tf.random.normal(shape=(batch_input_shape,self.z_dim))        
        fake_images = self.generator(random_latent_vectors)
        intepolated_images = interpolation_layer([real_images,fake_images])  
        
        for i in range(self.critic_steps):           
            critic_loss,critic_real_loss,critic_fake_loss,gp = self.train_critic(batch_input_shape,real_images,fake_images,intepolated_images)        
      
        generator_loss = self.train_generator(batch_input_shape,random_latent_vectors)
        
        return {"generator_loss":generator_loss,"critic_loss":critic_loss,"critic_real_loss":critic_real_loss,"critic_fake_loss":critic_fake_loss,"gp":gp}   
    
class Interpolate(tf.keras.layers.Layer):
    """keras layer thats interpolates two images"""
    def __init__(self,name = "interpolation"):
        super(Interpolate,self).__init__(name = name)        
    def call(self, inputs, **kwargs):
        batch_input_shape = tf.shape(inputs[0], name = "input_shape")[0]
        alpha = tf.random.uniform((batch_input_shape, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])   
    
    
def wasserstein(y_true, y_pred):
    """Keras version of wasserstein loss"""
    return -K.mean(y_true * y_pred)