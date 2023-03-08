from shapemaker import *


def test_ellipse(index):
    # Make paraview file from shape of ellipsoid dataset

    #   index:  Index of item in dataset, that should be plotted

    dataset = np.load(open(r"dataset/dataset_ellipsoid.npy", "rb"),allow_pickle=True)

    network =  ParkEtAl(3+3, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3 )
    network.load_state_dict(torch.load(r"models/shape_space_ellipse.pth", map_location=device))
    network.to(device) 

    network.eval()
    x = np.array([dataset[index][1].detach().cpu().numpy()])
    latent = Tensor( x ) 

    

    shape_space_toParaview(network, 127, index, latent)
    return

def test_8D(index):
    # Make paraview file from shape of dataset containing 2 metaballs in 3D (aka 8D)
    # !! OLD NETWORK LAYOUT !!

    #   index:  Index of item in dataset, that should be plotted
 
    dataset = np.load(open(r"dataset/dataset_16D.npy", "rb"),allow_pickle=True)

    network =  ParkEtAl(3+16, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3 )
    network.load_state_dict(torch.load(r"models/shape_space_16D_NoEncoder_AFF.pth", map_location=device))
    network.to(device) 

    network.eval()
    x = np.array([dataset[index][1].detach().cpu().numpy()])
    latent = Tensor( x ) 

    

    shape_space_toParaview(network, 160, index, latent)
    return


def test_shape(index):
    # Make paraview file from shape of dataset containing metaballs
    # Good approach ! 

    #   index:  Index of item in dataset, that should be plotted

    autoencoder = PointNetAutoEncoder(3,2000,16)
    autoencoder.load_state_dict(torch.load(r"models/autoencoder64_16D_AT2.pth", map_location=device))
    autoencoder.to(device) 
    autoencoder.eval()

    dataset = np.load(open(r"dataset/dataset_16D.npy", "rb"),allow_pickle=True)

    network =  FeatureSpaceNetwork2(3, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3, feature_space=16 )
    network.load_state_dict(torch.load(r"models/shape_space_16D_AT2.pth", map_location=device))
    network.to(device) 
    network.eval()


    point = Tensor(point)
    points = [dataset[index][0]]

    # points = points.cuda()
    points = np.array(points)
    points = Variable( Tensor(points) , requires_grad=True).to(device)


    inputs = torch.transpose(points, 1, 2)
    reconstructed_points, global_feat = autoencoder(inputs)

    shape_space_toParaview2(network, 256, index, global_feat)
    return


def test_face(index):

    # Make paraview file from shape of dataset containing the different faces

    #   index:  Index of item in dataset, that should be plotted
    
    autoencoder = PointNetAutoEncoder(3,23725,12)
    autoencoder.load_state_dict(torch.load(r"models/face_ae383.pth", map_location=device))
    autoencoder.to(device) 
    autoencoder.eval()

    dataset = np.load(open(r"dataset/dataset_faces383.npy", "rb"),allow_pickle=True)

    network =  FeatureSpaceNetwork2(3, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3, feature_space=12 )
    network.load_state_dict(torch.load(r"models/face_space383.pth", map_location=device))
    network.to(device) 
    network.eval()

    points = [dataset[index][0]]
    points = np.array(points)
    points = Variable( Tensor(points) , requires_grad=True).to(device)


    inputs = torch.transpose(points, 1, 2)
    reconstructed_points, global_feat = autoencoder(inputs)
    shape_space_toParaview2(network, 512, index, global_feat)
    return


def test_body(index, run):

    # Make paraview file from shape of dataset containing the different faces

    #   index:  Index of item in dataset, that should be plotted
    #   run:    Index of training run (appears in file name of autoencoder and shape space network)
    
    autoencoder = PointNetAutoEncoder(3,12500,12)
    autoencoder.load_state_dict(torch.load(r"models/human_ae"+str(run) + ".pth", map_location=device))
    autoencoder.to(device) 
    autoencoder.eval()

    dataset = np.load(open(r"dataset/dataset_human70.npy", "rb"),allow_pickle=True)

    network =  FeatureSpaceNetwork2(3, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3, feature_space=12 )
    #network =  ParkEtAl(3+16, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3 )
    network.load_state_dict(torch.load(r"models/human_space"+str(run) + ".pth", map_location=device))
    network.to(device) 
    network.eval()

    points = [np.array(dataset[index][0])]

    points = np.array(points)
    points = Variable( Tensor(points) , requires_grad=True).to(device)


    inputs = torch.transpose(points, 1, 2)
    reconstructed_points, global_feat = autoencoder(inputs)
    shape_space_toParaview2(network, 512, run*1000 + index, global_feat)
    return

def test_shape_space_circles(index):

    # Plots phase field of item in the shape space of 2D metacircles

    #   index:  Index of item in dataset, that should be plotted

    feature_dim = 9
    autoencoder = PointNetAutoEncoder(2,500,feature_dim)
    autoencoder.load_state_dict(torch.load(r"models/circle_cd100.0_ae.pth", map_location=device))
    autoencoder.to(device) 
    autoencoder.eval()

    dataset = np.load(open(r"dataset/dataset_9DCircleLATENT.npy", "rb"),allow_pickle=True)

    network =  FeatureSpaceNetwork2(2, [512]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3, feature_space=feature_dim)
    #network =  ParkEtAl(3+16, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3 )
    network.load_state_dict(torch.load(r"models/circle_cd100.0_nn.pth", map_location=device))
    network.to(device) 
    network.eval()
    

    points = [np.array(dataset[index][0])]
    draw_point_cloud(Tensor(points[0]))
    points = np.array(points)
    points = Variable( Tensor(points) , requires_grad=True).to(device)


    inputs = torch.transpose(points, 1, 2)
    reconstructed_points, global_feat = autoencoder(inputs)

    draw_phase_field_paper_is(network, global_feat, .5,.5, "cd100_"+str(index), True)
    return

def test_shape_space_circles_noEncoder(index):


    # Plots phase field of item in the shape space of 2D metacircles
    # In this case, the training was with the real feature vectors, so no autoencoder is needed

    #   index:  Index of item in dataset, that should be plotted

    feature_dim = 9

    dataset = np.load(open(r"dataset/dataset_9DCircleLATENT.npy", "rb"),allow_pickle=True)

    network =  FeatureSpaceNetwork2(2, [1024]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3, feature_space=feature_dim)
    network.load_state_dict(torch.load(r"models/shape_space_9D_NoEncoder_1024_2_.pth", map_location=device))
    network.to(device) 
    network.eval()
    

    points = [np.array(dataset[index][0])]
    #draw_point_cloud(Tensor(points[0]))
    points = np.array(points)
    points = Variable( Tensor(points) , requires_grad=True).to(device)
    global_feat = Tensor([np.array(dataset[index][1])])

    draw_phase_field_paper_is(network, global_feat, .5,.5, "1024_2_"+str(index), True)
    return


def show_latent():

    # Plots latent space
    # Yields a pyramid for 1 circle aka 3d Latent space

    ft = 3
    autoencoder = PointNetAutoEncoder(2,500,ft)
    autoencoder.load_state_dict(torch.load(r"models/circle10_ae.pth", map_location=device))
    autoencoder.to(device) 
    autoencoder.eval()

    dataset = np.load(open(r"dataset/dataset_R2circle500.npy", "rb"),allow_pickle=True)

    points = [np.array(pc[0]) for pc in dataset]

    points = np.array(points)
    points = Variable( Tensor(points) , requires_grad=True).to(device)

    inputs = torch.transpose(points, 1, 2)
    reconstructed_points, global_feat = autoencoder(inputs)

    latent_coords = global_feat.detach().numpy().T
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # For each set of style and range settings, plot n random points in the box

    ax.scatter(latent_coords[0],latent_coords[1],latent_coords[2], marker='o')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.show()

    return

for i in [123,124,125,126,99,101,1,2,3]:
    print(i)
    test_shape_space_circles_noEncoder(i)

