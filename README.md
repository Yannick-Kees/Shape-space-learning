# Table of Contents
1. [About](#example)
2. [How to install](#How-to-install)
4. [Files](#Files)
5. [Imports](#imports)
6. [External Packages](#External-packages)



# About
This is an implementation based on my Master Thesis

* [Learning Geometric Phase Field representations](https://drive.google.com/drive/folders/1LKQha7mYWvPzKKS2yC0zf_19FEzRlly8) (Yannick Kees 2022)

The area of computer graphics is a field of mathematics and computer science that deals with creating and manipulating visual content. One central question in computer graphics is the best way to represent three-dimensional data effectively. Most conventional approaches approximate the object’s surface discretely, for example, with meshes. One problem with these approaches is that spatial discretization limits them, just as the display of an image is limited by its number of pixels. Another popular approach is implicit representation, in which the object is written as the level set of an appropriate function. Instead of discretizing the output domain, we can now discretize the space of available functions. In this thesis, we deal with how to find such a function. To this end, we place a particular focus on phase field functions. These functions are, for the most part, constant, with a smooth transition along the surface where the value of the function changes. We distinguish between two different models: the Modica-Mortola approach,in which there are two different phases for an interior and exterior, and the Ambrosio-Tortorelli approach, in which there is only one phase. Calculating these functions is very challenging. Therefore we use deep learning, i.e., neural networks, to approximate the phase fields. The starting points for all our calculations are sets of points sampled from the surfaces of the objects. Learning implicit functions using the Modica-Mortola approach has been introduced in [Phase Transitions, Distance Functions, and Implicit Neural Representations](https://arxiv.org/abs/2106.07689). The new approach Ambrosio-Tortorelli approach in this work based on [Approximation of Functionals Depending on Jumps by Elliptic Functionals via $\Gamma$-Convergence](https://onlinelibrary.wiley.com/doi/pdf/10.1002/cpa.3160430805). Using this new approach, we will also be able to process open surfaces, which was impossible before. To do this, we distinguish two different tasks. The first goal will be to train a network to match the phase field for a single 3D object. In the second step, we will train a network that can
represent the phase fields for several objects at once. Therefore, the network receives additional object-specific input. 

![](images/ezgif-5-ae0dee7d73.gif)*Reconstruction of a square*
![](images/front.png)*Reconstruction of a bunny*
![](images/5.png)*Reconstruction of the nefreteti statue*






# How to install:
1. ssh .... & enter password
2. install conda using wget URL, bash~/Anaconda, conda env list
Then type 
```shell
source ~/anaconda3/bin/activate
conda create -n pytorch3d python=3.10
conda activate pytorch3d
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install matplotlib
pip install random-fourier-features-pytorch 
pip install k3d
git clone https://github.com/paulo-herrera/PyEVTK
cd PyEVTK
python setup.py install
git clone https://github.com/Yannick-Kees/Masterarbeit
cd Masterarbeit
```


Get files from volta using 
```
scp IP_ADRESS:~\Masterarbeit\structured2560.vts C:\Users\Yannick\Desktop
```



# Files
**! DISCLAIMER: This paragraph includes AI generated text !_**

| File | Description |
| --- | --- |
| `dataset.py` | Creates and Visualises Datasets. It can create datasets from generated objects or from a given directory of files |
| `interpolation.py` | Interpolation between to shapes of the shape space. The functions get as input the different shape space indices and interpolate the computed latent coordinates |
| `learn_shapespace_and_AE_2D.py` | Trains Shape space network together with the Encoder on 2D examples |
| `learn_shapespace_and_AE.py` | Trains Shape space network together with the Encoder on 3D examples |
| `learn_shapespace2D.py` | Trains Shape space network on 2D examples. The ground truth feature vectors are known and entered into the network. |
| `logger.py` | Denotes the loss over time in a file and also saves a copy of the current version of the executeable python script  |
| `loss_functionals.py` | Computes Modica-Mortola and Ambrosio-Tortorelli. There are different versions for the shape space learning and the surface reconstruction parts  |
| `misc.py` | Handles import of different file formates, enables CUDA and shows progress on console  |
| `networks.py` | Neural Networks  |
| `packages.py` | All used third party packages |
| `pointclouds.py` | Creates or changes point clouds |
| `run.py` | Solves the 2D reconstruction problem. Can be executed on any computer |
| `runMMS.py` | Deep minimizing movement scheme, for more see [HERE](https://drive.google.com/file/d/1txqmr8siLwQjA0l8lGcqScwr0hRWTuvF/view?usp=share_link) |
| `shapemaker.py` | Programm that can produce random point clouds in 2D or 3D form metaballs |
| `test_autoencoder.py` | Plot inputs and outputs of Autoencoder for differnt shapes of dataset  |
| `test_shape_space.py` | Make plots of elements of shape space after training  |
| `train_autoencoder.py` | Train PointNet - Autoencoder for the different datasets  |
| `visualizing.py` | Handles visualization of input and output data |
| `volta.py` | Solves the 3D reconstruction problem. Should only be executed on high performance computer |


General workflow:
- Create dataset in dataset.py. A dataset is a Nx2xPxd dimensional matrix
- Use dataset in one of the executable files: learn_shapespace_and_AE_2d.py, 
- In these files, you can turn the global parameters at the beginning of the file
- The Neural Network should be 'ParkEtAl' for the single shape learning and 'FeatureSpaceNetwork2' for the shape space learning. (The difference to FeatureSpaceNetwork is that the feature vector is concatenate to the input, after the input passes the Fourier layer. In FeatureSpaceNetwork the feature is directly 
concatenated to the input and the concatenation is passed through the fourier layer)
- Changing the Neural Network: The brackets contain the indices of layers, that get a skipping connection from the input layer
- The true number of fourier features is = num_features * 2 * input_dimension
- Run file on volta
- Copy autoencoder and shapespace network to local pc
- Run test_shape_space file 
- Done :)



## analyse_faces.py

The code is a Python script that shows the implementation of the Chamfer distance function for point clouds from the [PyTorch3D](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html) library. The script imports several libraries such as numpy, matplotlib, and PyTorch, including specific modules from them. The Chamfer distance function is defined as **own_chamfer_distance** with several arguments such as *x*, *y, x_lengths, y_lengths, x_normals, y_normals, weights, batch_reduction, point_reduction, norm*, and *infty*. The function takes two point clouds *x* and *y* and calculates the Chamfer distance between them. The input point clouds can be either a FloatTensor of shape *(N, P, D)* or a Pointclouds object representing a batch of point clouds with at most *P* points in each batch element, batch size *N* and feature dimension *D*.

The function first validates the reduction arguments, *batch_reduction* and *point_reduction*, by checking if they are valid values. Then it checks the input point cloud format and returns the padded points tensor along with the number of points per batch and the padded normals if the input is an instance of Pointclouds. Otherwise, it returns the input points with the number of points per cloud set to the size of the second dimension of points.

The function calculates the Chamfer distance between two point clouds *x* and *y* by calculating the pairwise distance between each point in x and each point in y and taking the minimum distance. The same process is repeated with x and y swapped to obtain two distances, and their sum is returned as the Chamfer distance. The function can also handle point clouds with varying numbers of points by using *x_lengths* and *y_lengths*. The distance metric used is either L1 or L2, depending on the value of norm. The reduction operation used to calculate the distance across the batch and across the points is determined by the values of *batch_reduction* and point_reduction, respectively. The function returns the reduced Chamfer distance and the reduced cosine distance of the normals if provided.

This script also defines a function called **make_color_plot** that takes three arguments: *n, norm*, and *infty*. The purpose of this function is to compute a similarity matrix between different faces and visualize it as a heatmap.

## dataset.py
The script is written in Python and contains functions that create and visualize different datasets of 3D objects using the shapemaker module. The functions include creating datasets of 8D metaballs, 3D metaballs, and ellipsoids, loading and visualizing 3D faces, human models, and chicken models. The functions that load and normalize the point clouds are provided as examples and are not used in the script. The resulting point clouds are saved to binary files. The *draw_point_cloud()* function from the shapemaker module is used to visualize the point clouds.

## interpolation.py
The code defines three functions, **interpol_2d(), interploate_3d(start_shape, end_shape)**, and **interploate_2d(start_shape, end_shape)**. The **interpol_2d()** function loads two trained models, interpolates between them using a loop, and calls two plotting functions, *color_plot_interpolate()* and *draw_phase_field_interpolate()*. The **interploate_3d(start_shape, end_shape)** function loads a dataset, an autoencoder, and a shape space network, interpolates between two shapes using the autoencoder and shape space network, and saves Paraview files of the intermediate shapes. The **interploate_2d(start_shape, end_shape)** function loads a dataset, an autoencoder, and a shape space network, interpolates between two shapes using the autoencoder and shape space network, and calls the *draw_phase_field_paper_is()* function to plot the intermediate shapes. 


## logger

This code defines a class called Register which creates a logging mechanism for an experiment. The *__init__* method takes a file argument, which is used to create a unique identifier for the experiment by combining the name of the file and the current date and time.

The method then creates a directory with this identifier in the logs folder and saves a copy of the script in this directory. It then creates a logger object with the name new_logger, sets its logging level to *DEBUG*, and adds two handlers to it: one that writes logs to a file called *_output.log* in the experiment directory, and one that writes *logs* to the console. The format of the log message includes the current time, the log level, and the log message itself.

The logging method takes three arguments: *i*, *total_loss*, and *stepsize*, and logs the iteration number, total loss, step size, and time elapsed since the experiment started.

The finished method logs a message indicating that the experiment has finished and prints "Finished!" to the console.

Overall, this code provides a convenient way to log the progress of an experiment to both a file and the console, which can be useful for debugging and analyzing the results of the experiment.

## loss_functionals.py

The code consists of different functions that define different loss functions used in the training of a neural network. The loss functions are for the Modica-Mortola part:

- **ModicaMortola**: Calculates the Monte Carlo Integral of $int_{[0,1]^2} W(u(x)) + eps * |Du(x)|^2 dx$.
- **Zero_recontruction_loss_Lip**: Calculates the Monte Carlo Estimation of $C * eps^(1/3) * \frac{1}{|P|} * \sum_{p\in P} |\dashint_{B_\delta}(p) u(s) ds|$.
- **Eikonal_loss**: Calculates the Eikonal loss around the points of point cloud.
- **Phase_loss**: Calculates the PHASE Loss = $e^(-.5)(\int_\Omega W(u) +e|Du|^2 + \frac{Ce(^.3)}{n} sum_{p\in P} \int u ) + \frac{\mu}{n} \sum_{p\in P} |1-|w||$.


And for the Ambrosio-Tortorelli part similar we have 


- **AT_Phasefield function** calculates a Monte Carlo integral of the form $\frac{1}{n} \sum_{i=1}^n W(u(x_i)) + \epsilon \cdot \left| \nabla u(x_i) \right|^2,$ where $W$ is a given double-well potential, $u$ is the target function, $x_i$ are randomly generated points in the domain $[0,1]^2$, and $\epsilon$ is a scalar parameter. The function takes as input the functions $W$ and $u$, the parameter $\epsilon$, the number of samples $n$, and the dimension $d$ of the point cloud. The function generates $n$ random points and their gradients, then returns the mean of the above expression over all the random points.

- **Zero_recontruction_loss_AT** function calculates a Monte Carlo estimate of the form $C \cdot \epsilon^{\frac{1}{3}} \cdot \frac{1}{|X|} \sum_{x \in X} \left| \dashint_{B_\delta(x)} u(s) , ds \right|,$ where $X$ is a given point cloud, $\delta$ is a fixed radius, $u$ is a function, $C$ is a given constant, and $\epsilon$ is a scalar parameter. The function takes as input the function $u$, the point cloud $X$, the parameter $\epsilon$, the constant $C$, the number of samples $m$, and the dimension $d$ of the point cloud. The function returns the mean of the above expression over all the points in the point cloud.

- **AT_loss function** combines the previous functions, by adding up the values.


This code also provides version for all of these function in the setting of shape space learning, where each time the neural network is evaluated in a point, the corresponding latent feature vector is also given into the network.

## misc.py 
This is a Python script that defines several functions for reading and processing point cloud data stored in different file formats. The script begins by importing all the packages defined in a separate Python module called *packages*. The first function defined in the script is **report_progress**, which takes three arguments: *current, total*, and *error*. This function prints out the progress of a training process by writing a message to the standard output (stdout) stream. It also flushes the output buffer to ensure that the message is displayed immediately.

In this file CUDA is enabled, if possible.

The script then defines several functions for reading different file formats of point cloud data, including .off, .obj, and .ply files. Each of these functions takes a file object as an input and returns a matrix of points representing the vertices of the point cloud. The **read_obj_file** function is used for reading large point clouds stored in .obj format. It reads the file line by line, extracts the vertex information, and stores it in a list of points. Finally, the script defines a function called **read_stl_file** that reads point cloud data stored in a binary STL file format, converts the triangle mesh to a set of unique points and returns the point cloud data as a numpy array.

Overall, this script provides a set of utility functions for reading and processing point cloud data in Python.

## networks.py

This files contains implementations of the different types of neural networks:

- **ParkEtAl**: PyTorch module that implements a neural network structure proposed by Park et al. It includes an optional Fourier feature layer and can perform geometric initialization. The class takes as input the dimensionality of the points in a point cloud, an array of integers indicating the number of neurons in each layer, an array of layer indices to skip, and several other parameters that affect the network's behavior, including the use of Fourier features and the number of features to use. The class implements the forward pass of the neural network and returns the output. It applies an affine linear transformation to the input data, followed by an activation function (either softplus or ReLU), for each layer in the network. The output of the final layer has a single neuron. If a Fourier feature layer is used, the input data is first transformed into the Fourier domain. If geometric initialization is used, the weight and bias of the final layer are initialized using a geometric initialization technique.

- **FeatureSpaceNetwork2** This is very similar to the *ParkEtAl* class, with the difference beeing, it is designed to process point clouds with additional feature vectors. In addition to the basic architecture, the class introduces the use of Fourier features to encode the point positions and a feature vector, which is concatenated to the input after passing through the Fourier feature layer.

- **PointNetAutoEncoder** PyTorch module that implements an autoencoder for point clouds. The purpose of this autoencoder is to encode a point cloud into a low-dimensional feature vector and then decode it back into its original point cloud shape. This is achieved through a series of fully connected layers and a 1D convolutional layer. The constructor takes three arguments: point_dim, num_points, and ft_dimension. point_dim is the dimension of each point in the point cloud, num_points is the number of points in the point cloud, and ft_dimension is the dimension of the feature vector to be learned. The module has three layers: *conv1, fc1,* and *fc3*. *conv1* is a 1D convolutional layer that takes in the point cloud as input and outputs a feature vector of dimension ft_dimension. *fc1* is a fully connected layer that takes in the output of *conv1* and outputs a feature vector of dimension 512. *fc3* is another fully connected layer that takes in the output of *fc1* and outputs a vector of size *num_points* * *point_dim*, which is then reshaped into the original point cloud shape. In the forward pass, the input point cloud is first passed through the *conv1* layer with a ReLU activation function. The resulting feature vector is then max-pooled over the points in the point cloud, and then flattened into a 1D tensor. This tensor is saved as the global feature. The flattened feature vector is then passed through the *fc1* layer with a ReLU activation function, and then through *fc3* to obtain the reconstructed point cloud. Finally, the reconstructed point cloud is reshaped into the original point cloud shape and returned, along with the global feature.

## pointclouds.py 

The script defines several functions for generating different types of point clouds, including:

- **produce_circle**: generates a set of points evenly distributed along the circumference of a circle.
- **produce_pan**: generates a set of points evenly distributed along the circumference of a circle and n additional points that are uniformly distributed from 0 to 1/3.
- **produce_spiral**: generates a set of points that follow a spiral pattern around a central point.
- **makeCube**: prints out a list of the coordinates of the corners of a cube with a given size.
- **flat_circle**: generates a set of points that lie on a 2D circle in 3D space.

The script also defines several variables, which are sets of points generated manually. These variables are:

- *k_quadrath*: a square with 4 points
- *m_quadrath*: a square with 8 points
- *g_quadrath*: a square with 24 points
- *triangle*: a triangle with 20 points
- *bow*: an open curve with 16 points
- *eight*: a curve in the shape of an 8 with 33 points


The code also includes three functions for manipulating point clouds.

- The first function, **add_noise(pc)**, adds random noise to every second point in the point cloud *pc.* The noise is generated using the *uniform()* function from the random module, which returns a random *float* in the range [ a, b ).

- The second function, **normalize(pc)**, scales all the points in the point cloud pc to be within the range [-0.3, 0.3]. It first converts the *pc* list to a *numpy* matrix, finds the minimum value in the matrix, and subtracts it from all elements. It then calculates the maximum value of the matrix, scales the matrix by a factor of 0.6, and shifts it by 0.3. Finally, it converts the *numpy* matrix back to a list.

- The third function, **cut_hole(pc)**, removes a circular hole from the point cloud *pc*. It creates a new list *new_pc* and iterates over each point in *pc*. If the point is not inside the specified circular region (determined by a center at [0, 0.15, 0] and a radius of 0.05), the point is appended to the new list *new_pc*. The function prints the length of pc and the length of *new_pc* for diagnostic purposes and returns *new_pc*.


## shapemaker.py

The function **shape_maker1** generates point cloud samples from a randomly generated 2D or 3D object using a Metaball approach. The *d* parameter specifies the dimension of the point cloud, either 2 or 3. The *num_points* parameter specifies the number of points to be sampled. If *save_latent* is set to *True*, the function returns the ground truth latent representation along with the generated point cloud.

If *d* is 2, the function generates a 2D point cloud using a Metacircle approach. The algorithm generates n circles with random centers and radii such that they do not overlap. The zero-level set of the function defined in the $f(x,y)$ function is used to extract the iso-surface of the object, which is then used to generate the point cloud. The normalize function is applied to the generated point cloud, and the function returns the normalized point cloud or the normalized point cloud along with the ground truth latent representation.

If *d* is 3, the function generates a 3D point cloud using a similar approach. It generates n spheres with random centers and radii such that they do not overlap. The zero-level set of the function defined in the f(x,y,z) function is used to extract the iso-surface of the object, which is then used to generate the point cloud. The normalize function is applied to the generated point cloud, and the function returns the normalized point cloud or the normalized point cloud along with the ground truth latent representation.


## visualizing.py

The code consists of functions used for generating visualizations of functions. The most important functions are:

- **draw_phase_field_paper(f, x_, y_, i, film)** This function creates a contour plot of the input function *f* on the rectangular domain [0, x_] x [0, y_]. The parameter *i* is used to name the output file, and *film* is a boolean variable that determines whether or not to save the output as an image file.

- **draw_phase_field_paper_is(f, latent, x_, y_, i, film)** This function is similar to *draw_phase_field_paper*, but it takes an additional input *latent* that is used to evaluate the function *f*. It generates a contour plot of *f* on the same rectangular domain [0, x_] x [0, y_], but uses the *imshow* function for visualization.

- **color_plot(f, y, film)** this function creates a 3D plot of the input function *f* on the unit square [0, 1] x [0, 1]. The parameter *y* is used to name the output file, and *film* is a boolean variable that determines whether or not to save the output as an image file.

- **draw_point_cloud(pc)** This function plots a point cloud *pc* in either two or three dimensions. If the dimension of the point cloud is two, the function plots the points as a scatter plot. If the dimension of the point cloud is three, the function plots the points as a 3D scatter plot.

## Executable files

All of the following files work the same. The code starts by importing the logger module. It then sets some settings, including the number of training sessions, the learning rate, patience, number of nodes, and other parameters. The code then sets up the neural network and loads the dataset. An optimizer is configured, and a logger is set up to record progress. The code then trains the network using a loop that includes backpropagation and a scheduler to adjust the learning rate. Finally, the trained network and autoencoder are saved.

### learn_shapespace_and_AE_2D.py
Shapespace: Trains Encoder and Decoder for 2-dimensional metacircles.

### learn_shapespace_and_AE.py
Shapespace: Trains Encoder and Decoder for 3-dimensional metacircles.

### learn_shapespace2D.py
Shape space: Trains decoder for 2-dimensional metacircles.The correct latent coordinates for each shapes are known and entered into the network.

# run.py
Surface reconstruction: Learn Phase field representation of a single shape in 2D. Runs on any computers, not only on high performance GPU's.

# volta.py
Surface reconstruction: Learn Phase field representation of a single shape in 3DD. Runs only on high performance computers.

# Imports

All packages are connected in series as you can see here:

![](images/import.png)*Imports*

# External packages:
* [Random Fourier Features Pytorch](https://github.com/jmclong/random-fourier-features-pytorch)  
* [K3D Jupyter](https://github.com/K3D-tools/K3D-jupyter)  <- bad renderer, not important
* [EVTK (Export VTK) ](https://github.com/paulo-herrera/PyEVTK) <- exports 3D examples for paraview
* [PointNet Autoencoder](https://github.com/charlesq34/pointnet-autoencoder/tree/cc270113da3f429cebdbe806aa665c1a47ccf0c1) 
* [PyTorch3D](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html) <- Chamfer Distance






