from math import prod
from misc import *

#############################
# Test-pointclouds ##########
#############################

# square with 4 points
k_quadrath = torch.tensor([ [ .2,.2 ],[ .2,.8 ],[ .8,.2 ],[ .8,.8 ]])
# square with 8 points
m_quadrath = torch.tensor([ [ .2,.2 ],[ .2,.8 ],[ .8,.2 ],[ .8,.8 ],[.2,.5],[.5,.2],[.8,.5],[.5,.8] ])
# square with 24 points
g_quadrath = torch.tensor([ [.2,.2],[.2,.3],[.2,.4],[.2,.5],[.2,.6],[.2,.7],[.2,.8],[.8,.2],[.8,.3],[.8,.4],[.8,.5],[.8,.6],[.8,.7],[.8,.8] ,[.3,.2] ,[.3,.8] ,[.4,.2] ,[.4,.8] ,[.5,.2] ,[.5,.8],[.6,.2] ,[.6,.8],[.7,.2] ,[.7,.8]     ]    )
# triangle
triangle = torch.tensor([ [.29,.41],[.22,.27],[.49,.85],[.79,.42],[.32,.49],[.25,.33],[.38,.62],[.44,.74],[.53,.8],[.58,.72],[.62,.66],[.66,.6],[.71,.54],[.74,.41],[.64,.38],[.54,.35],[.38,.31],[.46,.33],[.69,.39],[.27,.29]  ] )
# open curve
bow = torch.tensor([ [.24,.26],[.24,.3],[.28,.35],[.32,.38],[.37,.4],[.42,.43],[.47,.47],[.48,.53],[.44,.58],[.39,.62],[.35,.65],[.33,.68],[.3,.7],[.29,.77],[.26,.22],[.32,.19]   ])
# curve in 8 shape
eight = torch.tensor([ [.31,.34],[.32,.27],[.35,.22],[.4,.2],[.47,.18],[.51,.19],[.58,.21],[.63,.24],[.64,.3],[.64,.38],[.62,.43],[.58,.48],[.54,.52],[.48,.53],[.42,.53],[.36,.52],[.33,.45],[.43,.59],[.37,.64],[.34,.69],[.31,75],[.32,.79],[.35,.84],[.37,.87],[.45,.89],[.54,.87],[.58,.83],[.6,.8],[.61,.73],[.57,.67],[.51,.62],[.47,.6] ,[.45,.56]  ])

def produce_circle(n, r=.3):
    # n points sampled from a circle
    # r = .3
    pc = []
    
    for t in range(n):
        x   = float(r * np.sin( 2 * t * np.pi /n ) )
        y   = float(r * np.cos( 2 * t * np.pi /n ) )
        pc.append([x,y])
        
    return torch.tensor(pc)




def produce_pan(n, r=.3):
    
    # n points sampled from a circle, and additional n point uniformly distrubuted from 0 to 1/3
    # Remark: Not uniform distributed, because of using polar coordinates
    
    pc = []

    
    for t in range(n):
        x   = float(r * np.sin( 2 * t * np.pi /n )  )
        y   = float(r * np.cos( 2 * t * np.pi /n )  )
        pc.append([x,y])
        
    for t in range(n//3):
        pc.append([t/(n),t/(n)])
    return torch.tensor(pc)

def produce_spiral(n):
    # n points sampled from a circle
    r = .3
    pc = []
    
    for t in range(n):
        x   = float(r *t  * np.sin( 2 * t * np.pi /n ) + .5)
        y   = float(r * np.cos( 2 * t * np.pi /n ) + .5)
        pc.append([x,y])
        
    return torch.tensor( normalize(pc) )

def makeCube(size):
    #
    # Make a 3D Model of a cube
    
    l =[]
    counter  = 0

    for x in range(0,size+1):
        for y in range(0,size+1):
            for z in range(0,size+1):
                
                if x ==size or x==0 or y==size or y==0 or z==size or z==0:
                    counter += 1
                    print(float("{0:.2f}".format(x)), float("{0:.2f}".format(y)), float("{0:.2f}".format(z)))

    print(l)
    print(counter)


def flat_circle(n):
    # Make 2d circle in 3d space
    pc = []
    
    for _ in range(n):
        r =  np.random.uniform(0,.45)
        alpha = np.random.uniform(0,2.0*np.pi)
        x   = float(r * np.sin( alpha )  )
        y   = float(r * np.cos( alpha )  )
        pc.append([x,y, y])
        
    return pc


def signal(frequency):

    # define the range of x-values
    x_range = np.linspace(-5, 5, num=1000)

    # define the function for the curve
    def f_frequency(x):
        return np.sin(frequency*x) + np.cos(frequency*x)

    # evaluate the y-values of the curve over the given range
    y_values = f_frequency(x_range)

    # create 2D points by combining x and y values
    points = np.stack((x_range, y_values), axis=1)

    return Tensor(normalize(Tensor(points)))

def dual_slice(n):
    # Create slice on top of circle
    pc = []
    
    for _ in range(n):
        x = np.random.uniform(-1,1)
        y = np.random.uniform(0,1)
        
        if np.linalg.norm([x,y,0])<1:
            pc.append([x,y,0])
            pc.append([x,-y,0])
            
        if np.linalg.norm([x,0,y])<1:
            pc.append([x,0,-y])
            

    return pc


def triple_slice(n):
    # Create three slices touching at 120 degree angle
    pc = []
    
    for _ in range(n):
        x = np.random.uniform(-1,1)
        y = np.random.uniform(0,1)
        
        tan = np.tan((2.*np.pi)/12.0)
        tan = 1.0
        
        if np.linalg.norm([x,y,tan *y])<1:
            pc.append([x,y,tan * y])
            pc.append([x,-y,tan *y])
            
        if np.linalg.norm([x,0,y])<1:
            pc.append([x,0,-y])
            

    return pc


def sliced_sphere(n):
    
    pc = []
    for _ in range(n):
        x = np.random.uniform(-1,1)
        y = np.random.uniform(-1,1)
        z = np.random.uniform(-1,1)
         
        pc.append( 1.0/(np.linalg.norm([x,y,z])) * np.array([x,y,z]))
        
        pc.append([x,y,0])
        
    return pc
        

#############################
# change pointclouds ########
#############################

def add_noise(pc):
    # Adding noise to every second point in pointcloud
    
    for i in range(len(pc)//2):
        pc[2*i] +=  uniform(-0.02,.02)
        
    return pc


def normalize(pc):
    # Scale all point of the point cloud in such a way, that every coordinate is betweeen -0.3 and 0.3 
    # That is how they are still away enough from the boundary
        
    pc = np.matrix(pc)
    
    if np.amin(pc)<0:
        pc = pc - np.amin(pc)
    
    norm = pc - np.amin(pc)
    norm = .6 * (1.0/np.amax(norm) *  norm  ) - .3
    return norm.tolist()


def cut_hole(pc):
    # Cut a hole in point cloud
    
    new_pc = []
    
    for p in pc:
        
        if (p[0])**2+(p[2]-.15)**2>.005:
            new_pc.append(p)
            
    print(len(pc), len(new_pc))
    return new_pc


