

from audioop import findfactor
from pydoc import doc
from loss_functionals import *
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid

""" 
file = open("3dObjects/bigcube.off")    
pc = read_off(file)
print(pc)
cloud = torch.tensor(normalize(pc) )
draw_point_cloud(cloud)
""" 


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




def shape_maker8D(num_points, N):
    # Returns:
    #   Point Cloud samples from a randomly generated 3D Object using Metaball approach
    
    # Parameters:
    #   N:              Number of points
    #   num_points:     How many points are sampled

    
    # 3D Point Cloud
    
    condition = True
    
    while condition:
        
        n=  N
        g = 3
        m = []
        k = len(m)
        s = [ ]
        r = .8
        
        ft = []
        
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
        num_cells = 90
        x = np.linspace(-x_, x_, num_cells, dtype=np.float32)
        y = np.linspace(-x_, x_, num_cells, dtype=np.float32)
        z = np.linspace(-x_, x_, num_cells, dtype=np.float32)
        x, y, z = np.meshgrid(x, y, z, indexing='ij')   # make mesh grid
        Z = [[[ f(x[i][j][k], y[i][j][k], z[i][j][k] )  for k in range(len(x[0][0]))  ] for j in range(len(x[0])) ] for i in range(len(x)) ]# Evaluate function in points

                        
        contour = measure.marching_cubes(np.array(Z),0)[0]

        print(len(contour))
        condition = len(contour) < num_points

        
        if not condition:
            # Save Parameters for feature vector
            # Only works for 8D; Maybe change later
            
            ft.extend(m[0])
            ft.append(s[0])
            ft.extend(m[1])
            ft.append(s[1])

        
    choice_indices = np.random.choice(len(contour), num_points, replace=False)  # Choose some final point (point clouds need to be of same size for network)
    choices = [contour[i] for i in choice_indices]
        
    
    return (np.array(normalize(choices)) , Tensor(ft))
    

   
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
            n=  randint(2, 15)
            
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
            
        
        return np.array(normalize(choices)) 
    
    
    
    
def shape_maker1_contour():
    # Returns:
    #   Plot of of metaball in 3d

    n=  randint(2, 15)
    print(n)
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

                    
    verts, faces, normals, values = measure.marching_cubes(np.array(Z),0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlim(10, 35)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(10, 35)  # b = 10
    ax.set_zlim(10, 35)


    plt.show()

    return


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




#####################################
# Bezier curve Ansatz  ##############
#####################################


# !! COPYRIGHT:  https://stackoverflow.com/a/50751932 !!
# This approach was not used in the Thesis, since it is not possible (or too difficult) to scale it to work in higher dimensions

from scipy.special import binom



bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k) # Bernstein Polynomial

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)

def shape_maker2(n):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    rad = 0.2
    edgy = 0.05



    a = get_random_points(n=n, scale=1) 
    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
    plt.plot(x,y)

    plt.show()
    
    
    
################
# Run  #########
################
    
    
#shape_maker1(3)
#shape_maker2(6)
#shape_maker1_contour(3)
# make_ellipse()
# x = shape_maker8D(2000)
