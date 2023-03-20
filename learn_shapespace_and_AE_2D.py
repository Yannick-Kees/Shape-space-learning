from logger import *

####################
# Settings #########
####################

# Training Parameters
NUM_TRAINING_SESSIONS = 35000    
START_LEARNING_RATE = 0.01      
PATIENCE = 1500
NUM_NODES = 512         
FOURIER_FEATUERS = True
SIGMA = 1.7
MONTE_CARLO_SAMPLES = 1000
SHAPES_EACH_STEP = 80
EPSILON = .01
CONSTANT = 2. if FOURIER_FEATUERS else 10.0 

# Network Design
FEATURE_DIMENSION = 9
SIZE_POINTCLOUD = 500
TOTAL_SHAPES = 499


####################
# Main #############
####################

#   Load autoencoder
autoencoder = PointNetAutoEncoder(2,SIZE_POINTCLOUD,FEATURE_DIMENSION)
autoencoder.to(device) 

#   Load dataset
dataset = np.load(open(r"dataset/dataset_9DCircleLATENT.npy", "rb"),allow_pickle=True)

#   Setup Shape Space Learning Network
network =  FeatureSpaceNetwork2(2, [NUM_NODES]*7 , [4], FourierFeatures=FOURIER_FEATUERS, num_features = 8, sigma = SIGMA, feature_space=FEATURE_DIMENSION, geometric_init=False )
network.to(device) 

#   Configure Optimizer
all_params = chain(network.parameters(), autoencoder.parameters())
optimizer = torch.optim.Adam(all_params, START_LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

#   Setup Logger
Reg = Register(os.path.basename(__file__))

for i in range(NUM_TRAINING_SESSIONS+1):
    
    network.zero_grad()
    autoencoder.zero_grad()
    loss = 0.0
    shape_batch = np.random.choice(TOTAL_SHAPES, SHAPES_EACH_STEP, replace=False)
    
    for index in shape_batch:

        shape = dataset[index][0]
        pointcloud = Variable( Tensor(shape) , requires_grad=False).to(device)

        cloudT = Tensor( np.array([ np.array(shape).T]))
        pointcloudT = Variable( cloudT , requires_grad=True).to(device)

        rec, latent = autoencoder(pointcloudT)
        latent = torch.ravel(latent)
        loss +=  AT_loss_shapespace2(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES,  CONSTANT, latent ) 
        
    # backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    if (i%10==0):
        Reg.logging(i, loss, scheduler._last_lr[0])
    if scheduler._last_lr[0] < 1.e-5:
        break


torch.save(network.state_dict(), r"models/circle_largenn.pth")
torch.save(autoencoder.state_dict(), r"models/circleQUADRlarge_ae.pth")
Reg.finished()