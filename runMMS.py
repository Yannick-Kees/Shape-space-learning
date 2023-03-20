from loss_functionals import *

####################
# Settings #########
####################

# Neuronal Network
NUM_TRAINING_SESSIONS = 30
START_LEARNING_RATE = 0.01                        
PATIENCE = 1000
NUM_NODES = 512
FOURIER_FEATUERS = True
SIGMA = 1.3
BATCHSIZE = 100

# LOSS
MONTE_CARLO_SAMPLES = 500
EPSILON = .01   #0.05 MM 0.01 AT b 0_05 more smooth
TAU = EPSILON *10.0
CONSTANT = 14.0 if FOURIER_FEATUERS else 2.5     # 14, Constante höher bei FF
K = 15

####################
# Main #############
####################

v_k = ParkEtAl(2, [NUM_NODES], [], geometric_init=True, FourierFeatures=FOURIER_FEATUERS,num_features = 6, sigma = SIGMA)
v_k.to(device)

v_kplus1 = ParkEtAl(2, [NUM_NODES], [], geometric_init=True, FourierFeatures=FOURIER_FEATUERS, num_features = 6, sigma = SIGMA)
v_kplus1.to(device)
 
pointcloud = Variable( Tensor(produce_circle(1000,.2)) , requires_grad=True).to(device)

for k in range(K):
    print("Step ",k)

    v_kplus1.load_state_dict(v_k.state_dict())
    
    v_k.eval() # Fix NN
    v_kplus1.train() # Fix NN

    optimizer = optim.Adam(v_kplus1.parameters(), START_LEARNING_RATE )
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

    for i in range(NUM_TRAINING_SESSIONS+1):
        # training the network

        loss =  AT_loss(v_kplus1, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, 0, CONSTANT ) + TAU * sobolev(v_k,v_kplus1, MONTE_CARLO_SAMPLES, 0.0)
        if (i%50==0):
            report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().numpy() )

        # backpropagation
            
        v_kplus1.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    v_k.load_state_dict(v_kplus1.state_dict())


    color_plot(v_k, k, True)
    


