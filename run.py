from logger import *

####################
# Settings #########
####################

# Neuronal Network
NUM_TRAINING_SESSIONS = 5000
START_LEARNING_RATE = 0.01                        #  0.01
PATIENCE = 500
NUM_NODES = 512
FOURIER_FEATUERS = True
SIGMA = 1.3
BATCHSIZE = 100

# LOSS
LOSS = "AT"                                         # Either AT or MM
MONTE_CARLO_SAMPLES = 1000
MONTE_CARLO_BALL_SAMPLES = 20
EPSILON = .01 #0.05 MMFalse
CONSTANT = 14.0 if FOURIER_FEATUERS else 2.5 #2.0 if FOURIER_FEATUERS else 2.5 bei eps 0_01
MU = 2.0

# MISC
FILM = False                                        # Makes a movie from the learning process


####################
# Main #############
####################

# Setup Network
network = ParkEtAl(2, [NUM_NODES], [], geometric_init=True, FourierFeatures=FOURIER_FEATUERS, num_features = 6, sigma = SIGMA )
network.to(device)
 
optimizer = optim.Adam(network.parameters(), START_LEARNING_RATE )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

# Get Input Point Cloud
pointcloud = Variable(torch.tensor( produce_circle(1000,.2))  , requires_grad=True).to(device)

# Setup Logger
Reg = Register(os.path.basename(__file__))

for i in range(NUM_TRAINING_SESSIONS+1):
    # training the network
    # feed forward

    if LOSS == "AT":
        loss = AT_loss(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT )
    else:
        #loss = Phase_loss(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT, MU)
        loss = test_MM_GV(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT, False)
    

    # backpropagation  
    network.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    if FILM:
        report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().numpy() )
        color_plot(network, i, True)
        draw_phase_field(network, .5, .5, i, True)
    else:
        if (i%50==0):
            Reg.logging(i, loss, scheduler._last_lr[0])
    
    if scheduler._last_lr[0] < 1.e-5:
        break


draw_point_cloud(pointcloud)
color_plot(network, 2, False)
draw_phase_field_paper(network, .5, .5, i, False)
Reg.finished()