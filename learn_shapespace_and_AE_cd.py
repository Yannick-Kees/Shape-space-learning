from shapemaker import *

####################
# Settings #########
####################

# Training Parameters
NUM_TRAINING_SESSIONS = 40000    #35000
START_LEARNING_RATE = 0.01      
PATIENCE = 1500
NUM_NODES = 512         
FOURIER_FEATUERS = True
SIGMA = 1.7
MONTE_CARLO_SAMPLES = 1000
SHAPES_EACH_STEP = 80  # 80
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

all_params = chain(network.parameters(), autoencoder.parameters())
optimizer = torch.optim.Adam(all_params, START_LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

# Check if it really trains both networks at the same time | Part 1
#print(autoencoder(Variable( Tensor( np.array([ np.array(dataset[1][0]).T])) , requires_grad=True).to(device)))

for C in [1.0,5.0,10.0,50.0,100.0]:

    for i in range(NUM_TRAINING_SESSIONS+1):
        
        network.zero_grad()
        autoencoder.zero_grad()
        loss = 0.0
        shape_batch = np.random.choice(TOTAL_SHAPES, SHAPES_EACH_STEP, replace=False)
        
        for index in shape_batch:

            shape = dataset[index][0]#[:,:num_points]
            pointcloud = Variable( Tensor(shape) , requires_grad=False).to(device)

            cloudT = Tensor( np.array([ np.array(shape).T]))
            pointcloudT = Variable( cloudT , requires_grad=True).to(device)

            rec, latent = autoencoder(pointcloudT)
            latent = torch.ravel(latent)

            cd = chamfer_distance(torch.reshape(pointcloud, (1,500,2)), torch.transpose(rec, 1, 2))[0]
            loss +=  AT_loss_shapespace2(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES,  CONSTANT, latent ) + C * cd 
            
        if (i%10==0):
            report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().cpu().numpy() )
        
            
            # backpropagation
            
        loss.backward( )
        optimizer.step()
        scheduler.step(loss)

    # Check if it really trains both networks at the same time | Part 2   
    #print(autoencoder(Variable( Tensor( np.array([ np.array(dataset[1][0]).T])) , requires_grad=True).to(device)))

    torch.save(network.state_dict(), r"models/circle_cd"+str(C)+"_nn.pth")
    torch.save(autoencoder.state_dict(), r"models/circle_cd"+str(C)+"_ae.pth")
print("Finished")


