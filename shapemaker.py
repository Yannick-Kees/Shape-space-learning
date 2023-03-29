
from loss_functionals import *


#####################################
# Ellipsoid Ansatz  #################
#####################################

def make_ellipse():
    
    # Randomly create parameters
    a = uniform(.1,.3)
    b = uniform(.1,.3)
    c = uniform(.1,.3)
    
    ft= Tensor([a,b,c])     # Feature Vector

    
    def f(x,y,z):
        # Implicit function
        return (1.0/(a**2)) * x**2+ (1.0/(b**2)) *y**2+(1.0/(c**2)) *z**2 -1
    
    
    x_ = y_ = .3       
    num_cells = 30
    x = np.linspace(-x_, x_, num_cells, dtype=np.float32)
    y = np.linspace(-x_, x_, num_cells, dtype=np.float32)
    z = np.linspace(-x_, x_, num_cells, dtype=np.float32)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')   # make mesh grid
    
    Z = [[[ f(x[i][j][k], y[i][j][k], z[i][j][k] )  for k in range(len(x[0][0]))  ] for j in range(len(x[0])) ] for i in range(len(x)) ]# Evaluate function in points

                
    contour = measure.marching_cubes(np.array(Z),0)[0]  # Extract surface using marching cubes

    contour = np.array(contour)
    contour =  (float(x_*2)/(num_cells)  * contour ) - x_
    
    return [contour, ft]



#####################################
# Metaball Ansatz  ##################
#####################################


   
def shape_maker1(d, num_points, save_latent=False):
    # Returns:
    #   Point Cloud samples from a randomly generated 3D Object using Metaball approach
    
    # Parameters:
    #   d:              Dimension of point cloud (d=2 or d=3)
    #   num_points:     How many points are sampled
    #   N:              Number of balls

    if d==2:
        # 2D Point Cloud
        condition = True
        while condition:
            n=  3   # Number of balls
            
            g = 3   # goo factor
            m = []  # center points
            k = len(m)
            s = [ ] # radiusses
            r = .8  # global radius (isosurface)

            nm = 0  # temp center point
            ns = 0  # temp radius

        
        
            def overlap(s1,r1,s2,r2):
                # Returns:
                #   Do two circles overlap
                
                # Parameters:
                #   s1:     Center of first circle
                #   r1:     Radius of first circle
                #   s2:     Center of second circle
                #   r2:     Radius of second circle
                
                return np.linalg.norm(np.array(s1)-np.array(s2)) < abs(r1+r2)+r
            
            while len(s)!= n:
                # Create circles
                nm = [uniform(-1,1),uniform(-1,1)] 
                ns = uniform(0.01,.1)
            
                for i in range(len(s)):
                    
                    if overlap(m[i],s[i],nm,ns):
                        s.append(ns)
                        m.append(nm)
                        break
                    
                if len(s)==0:
                        s.append(ns)
                        m.append(nm)    

            def f(x,y):
                # Returns:
                #   Functions, whos zero-level set is the curve
                
                # Parameters:
                #   x:     x-coordinate
                #   y:     y-coordinate
                
                sum = -r
                
                for i in range(len(m)):
                    
                    if x != m[i][0] and y!= m[i][1]:
                        sum += s[i]/(  np.sqrt( (m[i][0]-x)**2+(m[i][1]-y)**2     )**g    )
                        
                    else:
                        sum+= 0
                        
                return sum
        
        # Extract iso-surface
        
            x_ = y_ = 2       
            xlist = np.linspace(-x_, x_, 700)
            ylist = np.linspace(-y_, y_, 700)
            X, Y = np.meshgrid(xlist, ylist)

            Z = [[ f(X[i][j], Y[i][j])  for j in range(len(X[0]))  ] for i in range(len(X)) ] # Evaluate function in points

            fig = plt.figure(1)                                                      # Draw contour plot
                                    
            contour = plt.contour(X, Y, Z,[0]) # Marching Cubes
            
            #plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)

            #plt.show()

            p = []
            
            for path in contour.collections[0].get_paths():
                
                for pp in path.vertices:
                    p.append(pp)


            plt.close(1)
            print(len(p) )
            condition = len(p) < num_points

            
        choice_indices = np.random.choice(len(p), num_points, replace=False)
        choices = [p[i] for i in choice_indices]
  
        
        if not save_latent:
            return np.array(normalize(choices))
        else:
            return np.array(normalize(choices)),  torch.cat((Tensor(s),torch.ravel(Tensor(m))),0)  # Also save GT latent representation
        
    if d==3:
        # 3D Point Cloud
        
        condition = True
        
        while condition:
            n= 3  # randint(2, 15)
            
            g = 3
            m = []
            k = len(m)
            s = [ ]
            r = .8
            
            def overlap(s1,r1,s2,r2):
                # Returns:
                #   Do two circles overlap
                
                # Parameters:
                #   s1:     Center of first circle
                #   r1:     Radius of first circle
                #   s2:     Center of second circle
                #   r2:     Radius of second circle
            
                return np.linalg.norm(np.array(s1)-np.array(s2)) < abs(r1+r2)+r
            
            while len(s)!= n:
                
                nm = [uniform(-1,1),uniform(-1,1),uniform(-1,1)] 
                ns = uniform(0.01,.1)
            
                for i in range(len(s)):
                    
                    if overlap(m[i],s[i],nm,ns):
                        s.append(ns)
                        m.append(nm)
                        break
                    
                if len(s)==0:
                        s.append(ns)
                        m.append(nm)    

            def f(x,y,z):
                # Returns:
                #   Functions, whos zero-level set is the curve
                
                # Parameters:
                #   x:     x-coordinate
                #   y:     y-coordinate
                #   z:     z-coordinate
            
                sum = -r
                
                for i in range(len(m)):
                    
                    if x != m[i][0] and y!= m[i][1] and z != m[i][2]:
                        sum += s[i]/(  np.sqrt( (m[i][0]-x)**2+(m[i][1]-y)**2 +(m[i][2]-z)**2     )**g    )
                        
                    else:
                        sum+= 0
                        
                return sum
            
            x_ = y_ = 2       
            num_cells = 40
            x = np.linspace(-x_, x_, num_cells, dtype=np.float32)
            y = np.linspace(-x_, x_, num_cells, dtype=np.float32)
            z = np.linspace(-x_, x_, num_cells, dtype=np.float32)
            x, y, z = np.meshgrid(x, y, z, indexing='ij')   # make mesh grid
            Z = [[[ f(x[i][j][k], y[i][j][k], z[i][j][k] )  for k in range(len(x[0][0]))  ] for j in range(len(x[0])) ] for i in range(len(x)) ]# Evaluate function in points

                            
            contour = measure.marching_cubes(np.array(Z),0)[0]


            condition = len(contour) < num_points

            
        choice_indices = np.random.choice(len(contour), num_points, replace=False)
        choices = [contour[i] for i in choice_indices]
            
        
        if not save_latent:
            return np.array(normalize(choices))
        else:
            return np.array(normalize(choices)),  torch.cat((Tensor(s),torch.ravel(Tensor(m))),0)  # Also save GT latent representation
    
    
    
def make_circle():
    # n points sampled from a circle
    # r = .3
    n= 500
    pc = []
    min_rad = .11
    border = .4
    x_0 = np.random.uniform(min_rad-border, border - min_rad)
    y_0 = np.random.uniform(min_rad-border, border - min_rad)
    max_rad = abs(border -  max(abs(x_0),abs(y_0)))
    r = np.random.uniform(min_rad, max_rad)
    
    for t in range(n):
        x   = float(r * np.sin( 2 * t * np.pi /n ) )
        y   = float(r * np.cos( 2 * t * np.pi /n ) )
        pc.append([x,y])
   
    return torch.tensor(pc) + torch.tensor([x_0,y_0])


    
################
# Run  #########
################
    
    
#draw_point_cloud(Tensor(normalize(shape_maker1(2,300))))

