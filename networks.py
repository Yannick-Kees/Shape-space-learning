from visualizing import *
     

class ParkEtAl(nn.Module):
    
    #   !! COPYRIGHT for this network class : https://github.com/amosgropp/IGR/blob/master/code/model/network.py  !!
    #   Added Fourier features

    def __init__(
        self,
        d_in, 
        dims,
        skip_in=(),
        geometric_init=True,
        radius_init=.3,
        beta=100,
        FourierFeatures = False,
        num_features = 0,
        sigma = .5
    ):
        # The network structure, that was proposed in Park et al.
        
        # Parameters:
        #   self:               Neuronal Network
        #   d_in:               dimension of points in point cloud
        #   dims:               array, where dims[i] are the number of neurons in layer i
        #   skip_in:            array containg the layer indices for skipping layers
        #   geometric_init:     Geometric initialisation
        #   radius_init:        Radius for Geometric initialisation
        #   beta:               Value for softmax activation function
        #   FourierFeatures:    Use Fourier Features
        #   num_features:       Number of Fourier Features
        #   sigma:              Sigma value for Frequencies in FF
        
        super().__init__()
        
        self.FourierFeatures = FourierFeatures
        self.d_in = d_in
        
        if FourierFeatures:
            # Use Fourier Features

            self.d_in = d_in * num_features * 2         # Dimension of Fourier Features
            self.original_dim = d_in                    # Original Dimension
            self.FFL = rff.layers.GaussianEncoding(sigma=sigma, input_size=self.original_dim, encoded_size=self.d_in//2) # Fourier Feature layer


        dims = [self.d_in] + dims + [1]     # Number of neurons in each layer
        
        self.num_layers = len(dims)         # Number of layers
        self.skip_in = skip_in              # Skipping layers

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
                
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim) # Affine linear transformation

            
            if geometric_init:
                # if true preform preform geometric initialization
                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin) # Save layer

        if beta > 0:
            self.activation = nn.Softplus(beta=beta) # Softplus activation function
            #self.activation = nn.Sigmoid()

        # vanilla ReLu
        else:
            self.activation = nn.ReLU()

    def forward(self, input):
        # forward pass of the NN

        x = input
        if self.FourierFeatures:
            # Fourier Layer
            x = self.FFL(x)

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                # Skipping layer
                
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x) # Apply layer

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x
    

class FeatureSpaceNetwork(nn.Module):
    
    #   !! COPYRIGHT for this network class : https://github.com/amosgropp/IGR/blob/master/code/model/network.py  !!
    #   Added Fourier features and a feature vector gets concatenated after passing FF-layer

    def __init__(
        self,
        d_in, 
        dims,
        skip_in=(),
        geometric_init=True,
        radius_init=.3,
        beta=100,
        FourierFeatures = False,
        num_features = 0,
        sigma = .5,
        feature_space = 64
    ):
        # The network structure, that was proposed in Park et al.
        
        # Parameters:
        #   self:               Neuronal Network
        #   d_in:               dimension of points in point cloud
        #   dims:               array, where dims[i] are the number of neurons in layer i
        #   skip_in:            array containg the layer indices for skipping layers
        #   geometric_init:     Geometric initialisation
        #   radius_init:        Radius for Geometric initialisation
        #   beta:               Value for softmax activation function
        #   FourierFeatures:    Use Fourier Features
        #   num_features:       Number of Fourier Features
        #   sigma:              Sigma value for Frequencies in FF
        
        super().__init__()
        
        self.FourierFeatures = FourierFeatures
        self.d_in = d_in
        if FourierFeatures:
            # Use Fourier Features

            self.d_in = d_in * num_features * 2         # Dimension of Fourier Features
            self.original_dim = d_in                    # Original Dimension
            self.FFL = rff.layers.GaussianEncoding(sigma=sigma, input_size=self.original_dim, encoded_size=self.d_in//2) # Fourier Feature layer


        dims = [self.d_in] + dims + [1]     # Number of neurons in each layer
        dims[0] += feature_space
        
        self.num_layers = len(dims)         # Number of layers
        self.skip_in = skip_in              # Skipping layers

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim) # Affine linear transformation

            
            if geometric_init:
                # if true preform preform geometric initialization
                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin) # Save layer

        if beta > 0:
            self.activation = nn.Softplus(beta=beta) # Softplus activation function
            #self.activation = nn.Sigmoid()

        # vanilla ReLu
        else:
            self.activation = nn.ReLU()

    def forward(self, input, ft):
        # forward pass of the NN
        #
        # input:    Point in R^3
        # ft:       Feature Vector   

        x = input
        if self.FourierFeatures:
            # Fourier Layer
            x = self.FFL(x)         # Pass Fourier Features
            x = torch.cat((x,ft),1) # Concatenate Feature vector

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                # Skipping layer
                
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x) # Apply layer

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x
    
    
class FeatureSpaceNetwork2(nn.Module):
    
    #   !! COPYRIGHT for this network class : https://github.com/amosgropp/IGR/blob/master/code/model/network.py  !!
    #   Added Fourier features and a feature vector gets concatenated after passing FF-layer

    def __init__(
        self,
        d_in, 
        dims,
        skip_in=(),
        geometric_init=True,
        radius_init=.3,
        beta=100,
        FourierFeatures = False,
        num_features = 0,
        sigma = .5,
        feature_space = 64
    ):
        # The network structure, that was proposed in Park et al.
        
        # Parameters:
        #   self:               Neuronal Network
        #   d_in:               dimension of points in point cloud
        #   dims:               array, where dims[i] are the number of neurons in layer i
        #   skip_in:            array containg the layer indices for skipping layers
        #   geometric_init:     Geometric initialisation
        #   radius_init:        Radius for Geometric initialisation
        #   beta:               Value for softmax activation function
        #   FourierFeatures:    Use Fourier Features
        #   num_features:       Number of Fourier Features
        #   sigma:              Sigma value for Frequencies in FF
        
        super().__init__()
        
        self.FourierFeatures = FourierFeatures
        self.d_in = d_in
        if FourierFeatures:
            # Use Fourier Features

            self.d_in = d_in * num_features * 2         # Dimension of Fourier Features
            self.original_dim = d_in                    # Original Dimension
            self.FFL = rff.layers.GaussianEncoding(sigma=sigma, input_size=self.original_dim, encoded_size=self.d_in//2) # Fourier Feature layer


        dims = [self.d_in] + dims + [1]     # Number of neurons in each layer
        dims[0] += feature_space
        
        
        self.num_layers = len(dims)         # Number of layers
        self.skip_in = skip_in              # Skipping layers

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
        
                out_dim = dims[layer + 1] - dims[0]
   
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim) # Affine linear transformation

            
            if geometric_init:
                # if true preform preform geometric initialization
                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin) # Save layer

        if beta > 0:
            self.activation = nn.Softplus(beta=beta) # Softplus activation function
            #self.activation = nn.Sigmoid()

        # vanilla ReLu
        else:
            self.activation = nn.ReLU()


    def forward(self, input, ft):
        # forward pass of the NN
        #
        # input:    Point in R^3
        # ft:       Feature Vector   

        x = input
        if self.FourierFeatures:
            # Fourier Layer
            x = self.FFL(x)         # Pass Fourier Features
   
            x = torch.cat((x,ft),1) # Concatenate Feature vector
            y = x
        
        
        for layer in range(0, self.num_layers - 1):


            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                # Skipping layer
                
                x = torch.cat([x, y], -1) / np.sqrt(2)

            x = lin(x) # Apply layer

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x
    

    
class PointNetAutoEncoder(nn.Module):
    #  !! COPYRIGHT: https://github.com/dhirajsuvarna/pointnet-autoencoder-pytorch/blob/master/model/model.py !!
    #
    # !! Good Encoder: Only use this one  !! 

    def __init__(self, point_dim, num_points, ft_dimension):
        super(PointNetAutoEncoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=point_dim, out_channels=ft_dimension, kernel_size=1)
        self.fc1 = nn.Linear(in_features=ft_dimension, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_points*point_dim)
        self.ft_dimension = ft_dimension

    
    def forward(self, x):

        batch_size = x.shape[0]
        point_dim = x.shape[1]
        num_points = x.shape[2]

        #encoder
        x = F.relu(self.conv1(x))

        # do max pooling 
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.ft_dimension)
        # get the global embedding
        global_feat = x

        #decoder
        x = F.relu(self.fc1(x))
        reconstructed_points = self.fc3(x)

        #do reshaping
        reconstructed_points = reconstructed_points.reshape(batch_size, point_dim, num_points)

        return reconstructed_points , global_feat
    
    
