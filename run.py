from loss_functionals import *

####################
# Settings #########
####################

# Neuronal Network
NUM_TRAINING_SESSIONS = 5000
START_LEARNING_RATE = 0.01                        #  0.01
PATIENCE = 1000
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


network = ParkEtAl(2, [NUM_NODES], [], geometric_init=True, FourierFeatures=FOURIER_FEATUERS, num_features = 6, sigma = SIGMA )
#network = small_MLP(128)
network.to(device)
 
optimizer = optim.Adam(network.parameters(), START_LEARNING_RATE )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

pointcloud = Variable(torch.tensor( produce_circle(1000,.2))  , requires_grad=True).to(device)


for i in range(NUM_TRAINING_SESSIONS+1):
    # training the network
    # feed forward
    # Omega = [0,1]^2


    if LOSS == "AT":
        loss = AT_loss(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT )
    else:
        #loss = Phase_loss(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT, MU)
        loss = test_MM_GV(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT, False)
    
    if FILM:
        report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().numpy() )
    else:
        if (i%50==0):
            report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().numpy() )

        # backpropagation
        
    network.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    if FILM:
        color_plot(network, i, True)
        draw_phase_field(network, .5, .5, i, True)

# test_MM_GV(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT, True)
# draw_point_cloud(pointcloud)
color_plot(network, 2, False)
# draw_phase_field(network, .5, .5, i, False)
draw_phase_field_paper_ss(network, .5, .5, i, False)
#draw_height(network)
#torch.save(network.state_dict(), r"C:\Users\Yannick\Desktop\MA\Programming part\models\CUBE.pth")


