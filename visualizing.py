from pointclouds import *


#############################
# 2D point cloud ############
#############################


def draw_phase_field(f,x_,y_, i, film):
    # Creating Contour plot of f
    
    # Parameters:
    #   f:      Function to plot
    #   x_,y_:  Drawing the function on [0,x_] \times [0,y_]
    #   i:      Index number, for naming the image files
    #   film:   bool, weather to store the image file

    xlist = np.linspace(-x_, x_, 100)
    ylist = np.linspace(-y_, y_, 100)
    X, Y = np.meshgrid(xlist, ylist)
    alpha = np.pi *1./3.
    Z = [[ f(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0]  for j in range(len(X[0]))  ] for i in range(len(X)) ] # Evaluate function in points
    
    fig = plt.figure()                                                      # Draw contour plot
    levels = [-1000.0,-5.0,-.5,0.0,.5,200.0]                                # Specify contours/level set to plot
    contour = plt.contour(X, Y, Z, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    if film:
        plt.savefig("images/mov/pf" + str(i).zfill(5) + '.jpg')
        plt.close(fig)
    else:
        plt.show()
    return


def draw_phase_field_paper(f,x_,y_, i, film):
    # Creating Contour plot of f; visualization for the paper
    
    # Parameters:
    #   f:      Function to plot
    #   x_,y_:  Drawing the function on [0,x_] \times [0,y_]
    #   i:      Index number, for naming the image files
    #   film:   bool, weather to store the image file

    xlist = np.linspace(-x_, x_, 100)
    ylist = np.linspace(-y_, y_, 100)
    X, Y = np.meshgrid(xlist, ylist)
    alpha = np.pi *1./3.
    Z = [[ f(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0]  for j in range(len(X[0]))  ] for i in range(len(X)) ] # Evaluate function in points
    
    fig = plt.figure()                                                      # Draw contour plot
    levels = np.arange(-1,1,.2)       # Specify contours/level set to plot
    contour = plt.contour(X, Y, Z, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    contour_filled = plt.contourf(X,Y,Z, cmap = "cividis")
    plt.colorbar(contour_filled)
    if film:
        plt.savefig("images/mov/pf" + str(i).zfill(5) + '.jpg')
        plt.close(fig)
    else:
        plt.show()
    return

def draw_phase_field_paper_ss(f, latent, x_,y_, i, film):
    # Creating Contour plot of f; visualization shape space objects
    
    # Parameters:
    #   f:      Function to plot
    #   x_,y_:  Drawing the function on [0,x_] \times [0,y_]
    #   i:      Index number, for naming the image files
    #   film:   bool, weather to store the image file

    xlist = np.linspace(-x_, x_, 100)
    ylist = np.linspace(-y_, y_, 100)
    X, Y = np.meshgrid(xlist, ylist)
    alpha = np.pi *1./3.
    Z = np.array([[ f(Tensor([[ X[i][j], Y[i][j] ]] ), latent).detach().numpy()[0][0]  for j in range(len(X[0]))  ] for i in range(len(X)) ]) # Evaluate function in points
   
    fig = plt.figure()                                                      # Draw contour plot
    levels = np.arange(0,1.0,.2)       # Specify contours/level set to plot
    contour = plt.contour(X, Y, Z, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    contour_filled = plt.contourf(X,Y,Z, cmap = "cividis")
    plt.colorbar(contour_filled)# plt.imshow
    if film:
        plt.savefig("images/mov/pf" + str(i).zfill(5) + '.jpg')
        plt.close(fig)
    else:
        plt.show()
    return


def draw_phase_field_paper_is(f, latent, x_,y_, i, film):
    # Creating Contour plot of f; visualization shape space objects
    # Uses imshow for visualization
    
    # Parameters:
    #   f:      Function to plot
    #   x_,y_:  Drawing the function on [0,x_] \times [0,y_]
    #   i:      Index number, for naming the image files
    #   film:   bool, weather to store the image file

    xlist = np.linspace(-x_, x_, 100)
    ylist = np.linspace(-y_, y_, 100)
    X, Y = np.meshgrid(xlist, ylist)

    Z = np.array([[ f(Tensor([[ X[i][j], Y[i][j] ]] ), latent).detach().numpy()[0][0]  for j in range(len(X[0]))  ] for i in range(len(X)) ]) # Evaluate function in points
   
    fig = plt.figure()                                                      # Draw contour plot
    levels = np.arange(0,1.0,.4)       # Specify contours/level set to plot
    #contour = plt.contour(X, Y, Z, levels, colors='k')
    #plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    contour_filled = plt.imshow(Z.reshape(100,100), cmap = "cividis", extent=(-.5,.5,-.5,.5), origin="lower", aspect=1.0)
    plt.colorbar(contour_filled)# plt.imshow
    if film:
        plt.savefig("images/mov/pf" + str(i).zfill(5) + '.jpg')
        plt.close(fig)
    else:
        plt.show()
    return

def draw_phase_field_interpolate(f, g, t, x_,y_, i, film):
    # Creating Contour plot of f
    
    # Parameters:
    #   f:      Function to plot
    #   x_,y_:  Drawing the function on [0,x_] \times [0,y_]
    #   i:      Index number, for naming the image files
    #   film:   bool, weather to store the image file

    xlist = np.linspace(-x_, x_, 100)
    ylist = np.linspace(-y_, y_, 100)
    X, Y = np.meshgrid(xlist, ylist)

    Z = [[ t * f(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0] + (1-t) * g(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0]     for j in range(len(X[0]))  ] for i in range(len(X)) ] # Evaluate function in points
    
    fig = plt.figure()                                                      # Draw contour plot
    levels = [-1000.0,-5.0,-.5,0.0,.5,200.0]                                # Specify contours/level set to plot
    contour = plt.contour(X, Y, Z, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    if film:
        plt.savefig("images/mov/pf" + str(i).zfill(5) + '.jpg')
        plt.close(fig)
    else:
        plt.show()
    return


def draw_height(f):
    # Plot the function values on circle around the origin, radius = 0.3
    # Used for verification of 2D circle
        
    # Parameters:
    #   f:      Function to plot
    
    x = np.linspace(0,2*np.pi,500)
    y = [ f(Tensor([ .3 * np.sin(a), .3 * np.cos(a) ]  )).detach().numpy()[0] for a in x     ]
    plt.xlabel("Angle")
    plt.ylabel("Function value")
    plt.plot(x,y)
    plt.show()
    
def color_plot(f, y, film):
    # Creating 3D plot of f on [0,1]^2
        
    # Parameters:
    #   f:      Function to plot
    #   y:      Index number, for naming the image files
    #   film:   bool, weather to film the learning process or not
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-.5, .5, 0.01)
    Y = np.arange(-.5, .5, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((len(X),len(X[0])))
    alpha = np.pi *1./3.
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i][j]= f(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0]
    # f(Tensor([ X[i][j], Y[i][j] ] ))
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}') # <- This may or may not be out commented, depending on compiler

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if film:
        plt.savefig('images/mov/cp' + str(y).zfill(5) + '.jpg')
        plt.close(fig)
    else:
        plt.show()
    return


def color_plot_interpolate(f, g, t, y, film):
    # Creating 3D plot of f on [0,1]^2
        
    # Parameters:
    #   f:      Function to plot
    #   y:      Index number, for naming the image files
    #   film:   bool, weather to film the learning process or not
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-.5, .5, 0.01)
    Y = np.arange(-.5, .5, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((len(X),len(X[0])))
    alpha = np.pi *1./3.
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i][j]= t *f(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0] + (1-t)* g(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0]
    # f(Tensor([ X[i][j], Y[i][j] ] ))
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}') # <- This may or may not be out commented, depending on compiler

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if film:
        plt.savefig('images/mov/cp' + str(y).zfill(5) + '.jpg')
        plt.close(fig)
    else:
        plt.show()
    return



#############################
# 2/3D point cloud ##########
#############################



def draw_point_cloud(pc):
    # Plotting point cloud
    
    # Parameters:
    #   pc:      Tensor of points

    d = pc.shape[1] # dimension
    
    if (d==2):
        fig1, ax1 = plt.subplots()
        ax1.set_aspect('equal')
        pointcloud = pc.detach().numpy().T 
        plt.plot(pointcloud[0],pointcloud[1], '.')
        plt.xlim(-.5,.5)
        plt.ylim(-.5,.5)
        plt.show()
        return
    if (d==3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pointcloud = pc.detach().numpy().T 
        ax.scatter(pointcloud[0],pointcloud[1],pointcloud[2])

        ax.set_xlim3d(-.3,.3)
        ax.set_ylim3d(-.3,.3)
        ax.set_zlim3d(-.3,.3)
        plt.show()

        
        
#############################
# 3D point cloud ############
#############################       

def plot_implicit(fn, shift=True):
    # Creating 3D contour plot of f on [0,1]^2 using marching cubes. Only works in Jupyter Notebooks (interactive)
        
    # Parameters:
    #   fn:      Function to plot
    #   shift:   for old models that have been created before 02.2022 set shift=False
    
    if shift:
        xa = ya = za= -.3
        xb = yb = zb = .3
    else:
        xa = ya = za= 0.0
        xb = yb = zb = 1.0

    plot = k3d.plot()       # start k3d
    x = np.linspace(xa, xb, 30, dtype=np.float32)
    y = np.linspace(ya, yb, 30, dtype=np.float32)
    z = np.linspace(za,zb, 30, dtype=np.float32)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')   # make mesh grid
    Z = [[[ fn(Tensor([ x[i][j][k], y[i][j][k], z[i][j][k] ] )).detach().numpy()  for k in range(len(x[0][0]))  ] for j in range(len(x[0])) ] for i in range(len(x)) ]# Evaluate function in points
    plt_iso = k3d.marching_cubes(Z, compression_level=5, xmin=xa, xmax=xb,ymin=ya, ymax=yb,  zmin=za, zmax=zb, level=0.0, flat_shading=False)
    plot += plt_iso     # add marching cubes to the file
    plot.display()      # Show plot



def test_f(t):
    # test function for implicit plotting
    return t[0]**2+t[1]**2+t[2]**2 -1 


def toParaview(f, n, l):
    # Makes a File, to visualize the network in ParaView
    # 
    #   f:  Neuronal Network function
    #   n:  Resolution, i.e. the number of function evaluations of NN in each dimension
    #   l:  Number of layers, only for the nameing of the file
    
    nx, ny, nz = n, n, n
    lx, ly, lz = .8, .8, .8
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    ncells = nx * ny * nz 
    npoints = (nx + 1) * (ny + 1) * (nz + 1) 

    # Coordinates 
    # 
    X = np.arange(0, lx + 0.1*dx, dx, dtype='float64') -.4
    Y = np.arange(0, ly + 0.1*dy, dy, dtype='float64')  -.4
    Z = np.arange(0, lz + 0.1*dz, dz, dtype='float64') -.4

    
    yy,xx,zz = np.meshgrid(X,Y,Z)
    all_data = np.array([xx,yy,zz]).T.reshape((n+1)**3,3)
    batch_size = 10000
    num_batches = (n+1)**3 // batch_size
    splitted_data = np.array_split(all_data, num_batches)

    Z = np.concatenate([ f( Variable(Tensor(v)).to(device) ).detach().cpu().reshape(-1).numpy() for v in splitted_data    ])
   
    #Z = f(v).detach().cpu().numpy().reshape(-1)
    #points = np.array([yy,zz,xx]).T
    #print(points) 
    # Variables 

    #Z = np.array( [ f( Variable( Tensor([ xx[i][j][k], yy[i][j][k], zz[i][j][k] ] ), requires_grad=True)).detach().numpy()  for k in range(len(x[0][0]))  for j in range(len(x[0])) for i in range(len(x)) ])

    #pressure = np.random.rand(ncells).reshape( (nx, ny, nz)) 
    #temp = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1)) 
    structuredToVTK("./structured"+str(n)+str(l), xx, yy, zz,  pointData = {"NN" : Z})



def shape_space_toParaview(f, n, l, fv):
    # Makes a File, to visualize the network in ParaView
    # 
    #   f:  Neuronal Network function
    #   n:  Resolution, i.e. the number of function evaluations of NN in each dimension
    #   l:  Number of layers, only for the nameing of the file
    
    nx, ny, nz = n, n, n
    lx, ly, lz = .8, .8, .8
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    ncells = nx * ny * nz 
    npoints = (nx + 1) * (ny + 1) * (nz + 1) 

    # Coordinates 
    # 
    X = np.arange(0, lx + 0.1*dx, dx, dtype='float64') -.4
    Y = np.arange(0, ly + 0.1*dy, dy, dtype='float64')  -.4
    Z = np.arange(0, lz + 0.1*dz, dz, dtype='float64') -.4
    """
    x = np.zeros((nx + 1, ny + 1, nz + 1)) 
    y = np.zeros((nx + 1, ny + 1, nz + 1)) 
    z = np.zeros((nx + 1, ny + 1, nz + 1)) 

    values = np.zeros((nx + 1, ny + 1, nz + 1)) 
    values = np.array([X,Y,Z]).T
    """
    fv = fv.detach().cpu().numpy()
    yy,xx,zz = np.meshgrid(X,Y,Z)
    all_data = np.array([xx,yy,zz]).T.reshape((n+1)**3,3)
    features = fv.repeat(len(all_data),0)
    start_points = np.concatenate((all_data, features), 1)
    batch_size = 10000
    num_batches = (n+1)**3 // batch_size
    splitted_data = np.array_split(start_points, num_batches)

    Z = np.concatenate([ f( Variable(Tensor(v)).to(device) ).detach().cpu().reshape(-1).numpy() for v in splitted_data    ])
   
    #Z = f(v).detach().cpu().numpy().reshape(-1)
    #points = np.array([yy,zz,xx]).T
    #print(points) 
    # Variables 

    #Z = np.array( [ f( Variable( Tensor([ xx[i][j][k], yy[i][j][k], zz[i][j][k] ] ), requires_grad=True)).detach().numpy()  for k in range(len(x[0][0]))  for j in range(len(x[0])) for i in range(len(x)) ])

    #pressure = np.random.rand(ncells).reshape( (nx, ny, nz)) 
    #temp = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1)) 
    structuredToVTK("./structured"+str(n)+str(l), xx, yy, zz,  pointData = {"NN" : Z})
    
    
    
    
def shape_space_toParaview2(f, n, l, fv):
    # Makes a File, to visualize the network in ParaView
    # 
    #   f:  Neuronal Network function
    #   n:  Resolution, i.e. the number of function evaluations of NN in each dimension
    #   l:  Number of layers, only for the nameing of the file
    
    nx, ny, nz = n, n, n
    lx, ly, lz = .8, .8, .8
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    ncells = nx * ny * nz 
    npoints = (nx + 1) * (ny + 1) * (nz + 1) 

    # Coordinates 
    # 
    X = np.arange(0, lx + 0.1*dx, dx, dtype='float64') -.4
    Y = np.arange(0, ly + 0.1*dy, dy, dtype='float64')  -.4
    Z = np.arange(0, lz + 0.1*dz, dz, dtype='float64') -.4
    """
    x = np.zeros((nx + 1, ny + 1, nz + 1)) 
    y = np.zeros((nx + 1, ny + 1, nz + 1)) 
    z = np.zeros((nx + 1, ny + 1, nz + 1)) 

    values = np.zeros((nx + 1, ny + 1, nz + 1)) 
    values = np.array([X,Y,Z]).T
    """
    fv = fv.detach().cpu().numpy()
    yy,xx,zz = np.meshgrid(X,Y,Z)
    all_data = np.array([xx,yy,zz]).T.reshape((n+1)**3,3)
    features = fv.repeat(len(all_data),0)
    
    batch_size = 10000
    num_batches = (n+1)**3 // batch_size
    splitted_data = np.array_split(all_data, num_batches)
    splitted_features = np.array_split(features, num_batches)


    Z = np.concatenate([ f( Variable(Tensor(splitted_data[v])).to(device) ,Variable(Tensor( splitted_features[v])).to(device) ).detach().cpu().reshape(-1).numpy() for v in range(len(splitted_data))    ])
   
    structuredToVTK("./structured"+str(n)+str(l), xx, yy, zz,  pointData = {"NN" : Z})
    return



