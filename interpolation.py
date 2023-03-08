from networks import *


def interpol_2d():
    # Simply interpolate function values of SDF's
    # Not good
    f = ParkEtAl(2, [128]*3, [], geometric_init=True, FourierFeatures=False, num_features = 6, sigma = 2 )
    f.load_state_dict(torch.load(r"C:\Users\Yannick\Desktop\MA\Programming part\models\circle.pth", map_location=device))


    g = ParkEtAl(2, [128]*3, [], geometric_init=True, FourierFeatures=False, num_features = 6, sigma = 2 )
    g.load_state_dict(torch.load(r"C:\Users\Yannick\Desktop\MA\Programming part\models\square.pth", map_location=device))


    def Interpolate(f, g, t, i):
        color_plot_interpolate(f,g,t, i, True)
        draw_phase_field_interpolate(f, g, t, .5, .5, i, True)

    for i in range(201):
        Interpolate(f, g, i/200, i)
        
    return
        
        
        
        
def interploate_3d(start_shape, end_shape):
    # Interpolate between shapes of shape space
    # Outputs Paraview files of the intermediate shapes
    #
    #   start_shape:    Index of beginning shape in the dataset
    #   end_shape:      Index of final shape in the dataset
    

    #   Load autoencoder
    autoencoder = PCAutoEncoder64(3, 1000)
    autoencoder.load_state_dict(torch.load(r"autoencoder64.pth", map_location=device))
    autoencoder.to(device) 
    autoencoder.eval()
    
    #   Load dataset
    dataset = np.load(open("dataset1k.npy", "rb"))
    
    #   Load shape space network
    network =  FeatureSpaceNetwork(3, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3 )
    network.load_state_dict(torch.load(r"shape_space_64.pth", map_location=device))
    network.to(device) 
    network.eval()


    points = [dataset[start_shape], dataset[end_shape]]

    points = points.cuda()
    points = np.array(points)
    points = Variable( Tensor(points) , requires_grad=True).to(device)


    inputs = torch.transpose(points, 1, 2)
    reconstructed_points, global_feat = autoencoder(inputs)


    for t in [0,.2,.4,.6,.8,1.0]:
        
        feature = Tensor(np.array([t* global_feat[1].detach().cpu().numpy() + (1-t) * global_feat[0].detach().cpu().numpy()]))

        shape_space_toParaview2(network, 128, t * 10, feature)
        
    return
        

def interploate_2d(start_shape, end_shape):
    # Interpolat between shapes of shape space
    #
    #   start_shape:    Index of beginning shape in the dataset
    #   end_shape:      Index of final shape in the dataset
    
    #   Load autoencoder
    feature_dim = 9
    autoencoder = PointNetAutoEncoder(2,500,feature_dim)
    autoencoder.load_state_dict(torch.load(r"models/circle_ae.pth", map_location=device))
    autoencoder.to(device) 
    autoencoder.eval()

    dataset = np.load(open(r"dataset/dataset_circle500.npy", "rb"),allow_pickle=True)

    # Load Shape space network
    network =  FeatureSpaceNetwork2(2, [512]*5 , [3], FourierFeatures=True, num_features = 8, sigma = 3, feature_space=feature_dim)
    network.load_state_dict(torch.load(r"models/circle_nn.pth", map_location=device))
    network.to(device) 
    network.eval()
    

    points = [dataset[start_shape][0], dataset[end_shape][0]]

    points = np.array(points)
    points = Variable( Tensor(points) , requires_grad=True).to(device)


    inputs = torch.transpose(points, 1, 2)
    reconstructed_points, global_feat = autoencoder(inputs)


    for t in [0,.2,.4,.6,.8,1.0]:
        # Interpolate latent vectors
        feature = Tensor(np.array([t* global_feat[1].detach().cpu().numpy() + (1-t) * global_feat[0].detach().cpu().numpy()]))
        draw_phase_field_paper_is(network, feature, .5,.5, 1, False)
        
    return
        

interploate_2d(19,36)