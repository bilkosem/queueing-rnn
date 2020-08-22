
###############################################################################
# Author: Bilgehan Kösem
# E-mail: bilkos92@gmail.com
# Date created: 07.18.2020
# Date last modified: 07.18.2020
# Python Version: 3.7
###############################################################################


import numpy as np
import copy

class QRNN:
    def __init__(self,
                 layer_list,
                 time_steps = 0,
                 weight_scaler = 1,
                 firing_rate_scaler = 0,
                 learning_rate = 0.1,
                 loss_function = 'mse',
                 optimizer = 'sgd'):
        
        #Assign basic parameters
        self.dh_list = []
        self.learning_rate = learning_rate
        self.shape = layer_list
        self.time_steps = time_steps
        self.loss_func = loss_function
        self.optimizer = optimizer
        n = len(layer_list)
        self.gradient_list=[]
        #Initialize layers
        self.layers = []
        for i in range(n):
            self.layers.append(np.zeros(self.shape[i]))

        #Initialize rates for horizontal and vertical
        self.rate = [] #Horizontal rates : sum of horizontal weights
        for i in range(n):
            self.rate.append(np.zeros((self.shape[i],1)))
        self.rate[-1] = firing_rate_scaler * np.ones(self.rate[-1].shape)
        
        self.rate_h = np.zeros((self.shape[1],1)) #Vertical rates : sum of Vertical weights
        # self.rate.shape = (H,)
        
        #Q and D are initiliazed with zeros
        self.Q_intime = []
        self.D_intime = []
        if self.time_steps != 0: 
            for t in range(self.time_steps+1):
                Q = []
                for i in range(n):
                    Q.append(np.zeros((self.shape[i],1)))
                self.Q_intime.append(copy.deepcopy(Q))
            self.D_intime = copy.deepcopy(self.Q_intime)
            
        # Initialize vertical wieghts: wplus, wminus
        self.wplus = []
        self.wminus = []
        for i in range(n-1):
            self.wplus.append(np.zeros((self.layers[i].size,
                                          self.layers[i+1].size)))
            self.gradient_list.append(np.zeros((self.layers[i].size,
                                          self.layers[i+1].size)))
            self.wminus.append(np.zeros((self.layers[i].size,
                                          self.layers[i+1].size)))
            self.gradient_list.append(np.zeros((self.layers[i].size,
                                          self.layers[i+1].size)))
        
        #Initialize horizonta wieghts: wplus_h, wminus_h
        self.wplus_h = np.zeros((self.layers[1].size,
                                 self.layers[1].size))
        self.gradient_list.append(np.zeros((self.layers[i].size,
                                      self.layers[i].size)))
        self.wminus_h = np.zeros((self.layers[1].size,
                                 self.layers[1].size))
        self.gradient_list.append(np.zeros((self.layers[i].size,
                                      self.layers[i].size)))
        
        #Initialize lambdas : lambda_plus, lambda_minus
        self.lambda_plus = []
        self.lambda_minus = []    
        
        self.global_var2 = []
        self.global_var = []
        self.init_weights(weight_scaler)
        self.gradient_list_2=copy.deepcopy(self.gradient_list)
        self.gradient_list_3=copy.deepcopy(self.gradient_list)
        
        self.bp_counter=0

    def recreate_Q_intime(self,step):
        #Q and D are initiliazed with zeros
        self.Q_intime = []
        self.D_intime = []
        for t in range(step+1):
            Q = []
            for i in range(len(self.shape)):
                Q.append(np.zeros((self.shape[i],1)))
            self.Q_intime.append(copy.deepcopy(Q))
        self.D_intime = copy.deepcopy(self.Q_intime)
        return 1
    
    def init_weights(self,weight_scaler):
        
        for i in range(len(self.wplus)):
            self.wplus[i] = weight_scaler*np.random.rand(self.wplus[i].shape[0],self.wplus[i].shape[1])
            self.wminus[i] = weight_scaler*np.random.rand(self.wminus[i].shape[0],self.wminus[i].shape[1]) 

        self.wplus_h = weight_scaler*np.random.rand(self.wplus_h.shape[0],self.wplus_h.shape[1])
        self.wminus_h = weight_scaler*np.random.rand(self.wminus_h.shape[0],self.wminus_h.shape[1])

        return 1 

    
    def calculate_rate(self):
        #for t in self.timesteps:
        for layer_num in range(len(self.layers)-1): #there is no rate for output layer
            for neuron in range(len(self.layers[layer_num])):
                self.rate[layer_num][neuron] = (self.wplus[layer_num][neuron] + self.wminus[layer_num][neuron]).sum()
        
        for neuron in range(len(self.layers[1])):
            self.rate_h[neuron] = (self.wplus_h[neuron] + self.wminus_h[neuron]).sum()
        
        return 1

    
    def feedforward(self,input_list):
        #self.recreate_Q_intime()
        t=0
        if self.time_steps==0:
            self.recreate_Q_intime(input_list.shape[0])
        for t,input_t in enumerate(input_list,start = 1):
                
            #self.D.clear() #clear list of D matrixes for the next iteration
            self.lambda_plus = np.where(input_t > 0, input_t, 0).reshape(-1,1)
            self.lambda_minus = np.where(input_t < 0, -input_t, 0).reshape(-1,1)
            
            # Input Layer
            self.D_intime[t][0] = self.rate[0]+self.lambda_minus
            self.Q_intime[t][0] = self.lambda_plus/self.D_intime[t][0]
            np.clip(self.Q_intime[t][0],0,1,out=self.Q_intime[t][0])    
            
            # Hidden Layer
        
            T_plus_i = self.wplus[0].transpose() @ self.Q_intime[t][0]
            T_minus_i =  self.wminus[0].transpose() @ self.Q_intime[t][0]
            T_plus_h = self.wplus_h.transpose() @ self.Q_intime[t-1][1]
            T_minus_h =  self.wminus_h.transpose() @ self.Q_intime[t-1][1]
            if t == input_list.shape[0]:
                self.D_intime[t][1] = self.rate[1] + T_minus_i + T_minus_h
            else:
                self.D_intime[t][1] = self.rate_h + T_minus_i + T_minus_h
            self.Q_intime[t][1] = (T_plus_i + T_plus_h)/(self.D_intime[t][1])
            np.clip(self.Q_intime[t][1],0,1,out=self.Q_intime[t][1]) 
            

        # Output Layer
        T_plus = self.wplus[1].transpose() @ self.Q_intime[t][1]
        T_minus = self.wminus[1].transpose() @ self.Q_intime[t][1]
        self.D_intime[t][2] = self.rate[2] + T_minus
        self.Q_intime[t][2] = T_plus / self.D_intime[t][2]
        np.clip(self.Q_intime[t][2] ,0,1,out=self.Q_intime[t][2])

        return copy.deepcopy(self.Q_intime[t][2])
    
    def vertical_gradient(self,t,l): #t:rimestep, l=layer

        vergrad2 = self.wplus[l].transpose()-self.Q_intime[t][l+1].reshape(-1,1)*self.wminus[l].transpose()
        vergrad3 = vergrad2/self.D_intime[t][l+1].reshape(-1,1)
        
        return vergrad3.transpose()
    
    def horizontal_gradient(self,t):        
        hor_grad2 = self.wplus_h.transpose()-self.Q_intime[t][1].reshape(-1,1)*self.wminus_h.transpose()
        hor_grad3 = hor_grad2/self.D_intime[t][1].reshape(-1,1)
        
        return hor_grad3.transpose()
    
    def backpropagation(self,real_output,tmp1=np.zeros((2,1))):
        self.bp_counter=self.bp_counter+1
        
        if self.optimizer=='nag':
            weights=np.asarray([self.wplus[0],self.wminus[0],self.wplus[1],self.wminus[1],self.wplus_h,self.wminus_h])

            x_ahead = weights-np.asarray(self.gradient_list)*0.9
            
            self.wplus[0] = copy.deepcopy(np.clip(x_ahead[0],a_min=0.001,a_max=None))
            self.wminus[0] = copy.deepcopy(np.clip(x_ahead[1],a_min=0.001,a_max=None))
            self.wplus[1] = copy.deepcopy(np.clip(x_ahead[2],a_min=0.001,a_max=None))
            self.wminus[1] = copy.deepcopy(np.clip(x_ahead[3],a_min=0.001,a_max=None))
            self.wplus_h = copy.deepcopy(np.clip(x_ahead[4],a_min=0.001,a_max=None))
            self.wminus_h = copy.deepcopy(np.clip(x_ahead[5],a_min=0.001,a_max=None))
            

                
        d_Who_p = []
        d_Who_m = []
        d_Whh_p = []
        d_Whh_m = []
        d_Wih_p = []
        d_Wih_m = []
        
        d_Who_p.append(((self.Q_intime[-1][1] @ (1/self.D_intime[-1][2]).transpose()).transpose()*tmp1).transpose())
        d_Who_m.append((-(self.Q_intime[-1][1] @ (self.Q_intime[-1][2]/self.D_intime[-1][2]).transpose()).transpose()*tmp1).transpose())        
        
        steps = len(self.Q_intime)-1
        o3_h3 = self.vertical_gradient(steps,1)

        d_hidden_layer = o3_h3

        for t in reversed(range(1,steps+1)):#2 1 0
            if t==steps:
                d_whop = d_hidden_layer*(self.Q_intime[t][1]/self.D_intime[t][1])
                d_whom = copy.deepcopy(d_whop)
                d_Who_p.append((d_whop.transpose()*tmp1).transpose())
                d_Who_m.append((d_whom.transpose()*tmp1).transpose())
                d_hidden_layer = d_hidden_layer@tmp1
            ####################Gradients from hidden layers###################
            #dqh/dNi
            d_hidden_layer_N = d_hidden_layer/self.D_intime[t][1]
            #dqh/dDi
            d_hidden_layer_D = d_hidden_layer*(-self.Q_intime[t][1]/self.D_intime[t][1])
            
            wihm = self.Q_intime[t][0] @ d_hidden_layer_D.transpose()
            wihp = self.Q_intime[t][0] @ d_hidden_layer_N.transpose()
            if t==steps:
                whhm = self.Q_intime[t-1][1] @ d_hidden_layer_D.transpose()
                whhp = self.Q_intime[t-1][1] @ d_hidden_layer_N.transpose()
            else :
                whhm = (1+self.Q_intime[t-1][1]) @ d_hidden_layer_D.transpose()
                whhp_n = self.Q_intime[t-1][1] @ d_hidden_layer_N.transpose()
                whhp_d = d_hidden_layer_D
                whhp = whhp_n + whhp_d
                
            d_Whh_p.append(whhp)
            d_Whh_m.append(whhm)
            d_Wih_p.append(wihp)
            d_Wih_m.append(wihm)
            
            ################### Gradients from input layers ###################
            #dqh/dqi
            d_input_layer = self.vertical_gradient(t,0)
            #dqi/dDi
            d_input_layer_D = d_input_layer*(-(self.Q_intime[t][0]/self.D_intime[t][0]))
            d_Wih_p.append((d_input_layer_D.transpose()*d_hidden_layer_D).transpose())
            d_Wih_m.append((d_input_layer_D.transpose()*d_hidden_layer_D).transpose())

            ######################## New Hidden Layer #########################
            self.dh_list.append(d_hidden_layer)
            d_hidden_layer = self.horizontal_gradient(t) @ d_hidden_layer
            #değiştirebilirsin

        #dh_list.clear()
        #weights=np.asarray([self.wplus[0],self.wminus[0],self.wplus[1],self.wminus[1],self.wplus_h,self.wminus_h])

        if self.optimizer == 'sgd':
            grads=np.asarray([sum(d_Wih_p),sum(d_Wih_m),sum(d_Who_p),sum(d_Who_m),sum(d_Whh_p),sum(d_Whh_m)])*self.learning_rate
            self.update_weigths(grads)

        elif self.optimizer == 'momentum':
            moment = np.asarray(self.gradient_list)*0.9
            grads = moment + np.asarray([sum(d_Wih_p),sum(d_Wih_m),sum(d_Who_p),sum(d_Who_m),sum(d_Whh_p),sum(d_Whh_m)])*self.learning_rate
            self.update_weigths(grads)
            self.gradient_list=copy.deepcopy(grads)
        
        elif self.optimizer == 'nag':
            grads = np.asarray([sum(d_Wih_p),sum(d_Wih_m),sum(d_Who_p),sum(d_Who_m),sum(d_Whh_p),sum(d_Whh_m)])
            self.gradient_list = np.asarray(self.gradient_list)*0.9 + grads*self.learning_rate
            self.update_weigths(self.gradient_list)
            
        elif self.optimizer == 'adagrad':
            grads = np.asarray([sum(d_Wih_p),sum(d_Wih_m),sum(d_Who_p),sum(d_Who_m),sum(d_Whh_p),sum(d_Whh_m)])
            grads_2=np.square(grads)
            self.gradient_list+=copy.deepcopy(grads_2)
            for i in range(len(grads)):
                grads[i] = (self.learning_rate/(np.sqrt(self.gradient_list[i]+0.000001)))*grads[i]
            self.update_weigths(grads)
            #self.gradient_list+=copy.deepcopy(grads_2)
            
        elif self.optimizer == 'adadelta': #gradient_list_2->gt//gradient_list->teta_t

            eps=0.000001;beta=0.90;
            grads=np.asarray([sum(d_Wih_p),sum(d_Wih_m),sum(d_Who_p),sum(d_Who_m),sum(d_Whh_p),sum(d_Whh_m)])

            grads_2 = np.square(grads)
            self.gradient_list_2 = beta*np.asarray(self.gradient_list_2) + (1-beta)*grads_2
            
            delta_teta = copy.deepcopy(self.gradient_list)
            for i in range(len(grads)):
                #delta_teta[i] = (self.learning_rate/(np.sqrt(self.gradient_list_2[i]+0.000001)))*grads[i]
                delta_teta[i] = (np.sqrt(self.gradient_list[i]+0.000001)/(np.sqrt(self.gradient_list_2[i]+0.000001)))*grads[i]
      
            self.gradient_list = beta*np.asarray(self.gradient_list) + (1-beta)*np.square(delta_teta)

            for i in range(len(grads)):
                grads[i] = (np.sqrt(self.gradient_list[i]+0.000001)/(np.sqrt(self.gradient_list_2[i]+0.000001)))*grads[i]
            
            self.update_weigths(grads)
            #self.gradient_list = beta*np.asarray(self.gradient_list) + (1-beta)*np.square(grads)

            
        elif self.optimizer == 'rmsprop':
            eps=0.00000001
            grads=np.asarray([sum(d_Wih_p),sum(d_Wih_m),sum(d_Who_p),sum(d_Who_m),sum(d_Whh_p),sum(d_Whh_m)])
            grads_2 = np.square(grads)
            self.gradient_list = 0.9*np.asarray(self.gradient_list) + 0.1*grads_2

            for i in range(len(grads)):
                grads[i] = (self.learning_rate / np.sqrt(self.gradient_list[i]+eps)) * grads[i]
            self.update_weigths(grads)
            
        elif self.optimizer == 'adam':
            eps=0.000001;beta1=0.9;beta2=0.999;
            
            grads=np.asarray([sum(d_Wih_p),sum(d_Wih_m),sum(d_Who_p),sum(d_Who_m),sum(d_Whh_p),sum(d_Whh_m)])
            grads_2 = np.square(grads)
            self.gradient_list = beta1*np.asarray(self.gradient_list) + (1-beta1)*grads
            self.gradient_list_2 = beta2*np.asarray(self.gradient_list_2) + (1-beta2)*grads_2
            gt = self.gradient_list/(1-beta1**self.bp_counter)
            gt2 = self.gradient_list_2/(1-beta2**self.bp_counter)
            
            for i in range(len(grads)):
                grads[i] = (self.learning_rate / (np.sqrt(gt2[i])+eps)) * gt[i]
            self.update_weigths(grads)
            
        elif self.optimizer == 'adamax':
            eps=0.000001;beta1=0.9;beta2=0.999;
            
            grads=np.asarray([sum(d_Wih_p),sum(d_Wih_m),sum(d_Who_p),sum(d_Who_m),sum(d_Whh_p),sum(d_Whh_m)])
            grads_2 = np.square(grads)
            self.gradient_list = beta1*np.asarray(self.gradient_list) + (1-beta1)*grads
            gt = self.gradient_list/(1-beta1**self.bp_counter)
            
            for i in range(len(self.gradient_list_2)):
                self.gradient_list_2[i] =np.maximum(beta2*np.asarray(self.gradient_list_2)[i],abs(grads)[i])
            #learning rate is 0.002 as an adviced value
            grads = (self.learning_rate / (np.asarray(self.gradient_list_2)+eps)) * gt
            self.update_weigths(grads)

        elif self.optimizer == 'nadam':
            eps=0.000001;beta1=0.9;beta2=0.999;
            
            grads=np.asarray([sum(d_Wih_p),sum(d_Wih_m),sum(d_Who_p),sum(d_Who_m),sum(d_Whh_p),sum(d_Whh_m)])
            grads_2 = np.square(grads)
            self.gradient_list = beta1*np.asarray(self.gradient_list) + (1-beta1)*grads
            self.gradient_list_2 = beta2*np.asarray(self.gradient_list_2) + (1-beta2)*grads_2
            gt = self.gradient_list/(1-beta1**self.bp_counter)
            gt2 = self.gradient_list_2/(1-beta2**self.bp_counter)
            gt = (beta1*gt-((1-beta1)/(1-beta1**self.bp_counter))*gt)
            for i in range(len(grads)):
                grads[i] = (self.learning_rate / (np.sqrt(gt2[i])+eps)) * gt[i]
            self.update_weigths(grads)
        
        elif self.optimizer == 'amsgrad':
            eps=0.000001;beta1=0.9;beta2=0.999;
            
            grads=np.asarray([sum(d_Wih_p),sum(d_Wih_m),sum(d_Who_p),sum(d_Who_m),sum(d_Whh_p),sum(d_Whh_m)])
            grads_2 = np.square(grads)
            self.gradient_list = beta1*np.asarray(self.gradient_list) + (1-beta1)*grads
            self.gradient_list_2 = beta2*np.asarray(self.gradient_list_2) + (1-beta2)*grads_2
            gt = self.gradient_list/(1-beta1**self.bp_counter)
            gt2 = self.gradient_list_2/(1-beta2**self.bp_counter)
            
            for i in range(len(self.gradient_list_2)):
                self.gradient_list_3[i] =np.maximum(np.asarray(self.gradient_list_3)[i],gt2[i])
           
            for i in range(len(grads)):
                grads[i] = (self.learning_rate / (np.sqrt(self.gradient_list_3[i])+eps)) * gt[i]
            self.update_weigths(grads)
        
        else:
            raise Exception('Unknown optimizer : \'{}\''.format(self.optimizer))
        
        return 1
    
    def update_weigths(self,grads):
        self.wplus[0] = copy.deepcopy(np.clip(self.wplus[0] - grads[0],a_min=0.001,a_max=None))
        self.wminus[0] = copy.deepcopy(np.clip(self.wminus[0] - grads[1],a_min=0.001,a_max=None))
        self.wplus[1] = copy.deepcopy(np.clip(self.wplus[1] - grads[2],a_min=0.001,a_max=None))
        self.wminus[1] = copy.deepcopy(np.clip(self.wminus[1] - grads[3],a_min=0.001,a_max=None))
        self.wplus_h = copy.deepcopy(np.clip(self.wplus_h - grads[4],a_min=0.001,a_max=None))
        self.wminus_h = copy.deepcopy(np.clip(self.wminus_h - grads[5],a_min=0.001,a_max=None))
        
    def softmax(self,xs):
      return np.exp(xs) / sum(np.exp(xs))
  
        