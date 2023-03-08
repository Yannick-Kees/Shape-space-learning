from shapemaker import *


####################
# Metaballs ########
####################

def create_8D():
    # Create dataset of metaballs from 2 balls
    # 2 . 3 coordinates + 2 radii = 8 dimensions
    
    points = []

    f = open(r"dataset/dataset_8D.npy", "wb")
    
    for i in range(50):
        points.append(shape_maker8D(2000, 4))
        print(i)
        
        
    np.save(f, points)
    return
    
def eval_8D(i):
    # Plot metaballs of 2 balls from the dataset
    
    f = np.load(open(r"dataset/dataset_16D.npy", "rb"),allow_pickle=True)
    draw_point_cloud(Tensor(f[i][0]))
    print(f[i][1])


def create():
    # Create dataset of metaballs
    
    points = []

    f = open(r"dataset/dataset5k.npy", "wb")
    
    for i in range(5000):
        points.append(shape_maker1(2,1000))
        print(i)
        
    np.save(f, points)
    return
    
def eval(i):
    # visualize items from the metaball dataset
    
    f = np.load(open(r"dataset/dataset1k.npy", "rb"))
    print(f.shape)
    draw_point_cloud(Tensor(f[i]))
    
    


####################
# Ellipsoid ########
####################    
    
    
def create_ell():
    # Create dataset of ellipsoids
    points = []

    f = open(r"dataset/dataset_ellipsoid.npy", "wb")
    
    for i in range(50):
        points.append(make_ellipse())
        print(i)
        
        
    np.save(f, points)
    return
    
def eval_ell(i):
    # Plot ellipsoids from the dataset
    
    f = np.load(open(r"dataset/dataset_ellipsoid.npy", "rb"),allow_pickle=True)
    draw_point_cloud(Tensor(f[i][0]))
    
    
####################
# Faces   ##########
####################  

# 383 faces in total

#import open3d as o3d

def load_face(index):
    # Load face 'index' from .ply file
    # Need to enable the import open3d part for this
    # But this does not work on volta... so only enable on local pc!!

    pcd = o3d.io.read_point_cloud("faces/face_" + str(index) + ".ply")
    x = np.asarray(pcd.points)
    x = Tensor(normalize(x))
    x = x - np.array([-0.22,-0.22,0.24])
    x = Tensor(normalize(x))
    x = x + np.array([0.0,.15,-0.15])
    x = np.array(normalize(x))
    
    if x.shape[0] != 23725:
        print("NOT THE RIGHT SIZE!!")
        print(index)
        
    return x

def make_face_dataset():
    # Convert all .ply files into one numpy array and save it
    points = []

    f = open(r"dataset/dataset_faces.npy", "wb")
    
    for i in range(383):
        print(i)
        x = (load_face(i),0)
        points.append(x)
        
    np.save(f, points)
    return



def show_face(i):
    # Plot face from the dataset

    f = np.load(open(r"dataset/dataset_faces383.npy", "rb"),allow_pickle=True)
    draw_point_cloud(Tensor(f[i][0]))     




def load_human(index):
    # Load face 'index' from .ply file
    # Need to enable the import open3d part for this
    file = open("/home/data/sassen/Projects/reducedNRIC/SCAPE/data/inputSCAPE" + str(index) + ".ply")
    pcd = read_ply_file_human(file)

    x = .8* Tensor(pcd)
    x = x - np.array([+0.8,0.0,0.0])
    
    x = Tensor(normalize(x))
    """
    x = x + np.array([0.0,.15,-0.15])
    x = np.array(normalize(x))
    """
    
    if x.shape[0] != 12500:
        print("NOT THE RIGHT SIZE!!")
        print(index)
        
    return x

def make_human_dataset():
    # Convert all .ply files into one numpy array and save it
    points = []

    f = open(r"dataset/dataset_human70.npy", "wb")
    
    for i in range(1,70):
        print(i)
        x = (load_human(i),0)
        points.append(x)
        
    np.save(f, points)
    return

def show_human(i):
    # Plot face from the dataset

    f = np.load(open(r"dataset/dataset_human70.npy", "rb"),allow_pickle=True)
    draw_point_cloud(Tensor(f[i][0]))   




####################
# Human Moves ######
####################  


def load_chicken(index):
    # Load face 'index' from .ply file
    # Need to enable the import open3d part for this
    file = open("/home/data/sassen/Data/DFAUST/meshes_m/50009_chicken_wings_" + str(index) + ".ply")
    pcd = read_ply_file_human(file)
    x= pcd
    #x = .8* Tensor(pcd)
    x = x - np.array([0.0,0.3,-0.3])
    
    x = Tensor(normalize(x))
    """
    x = x + np.array([0.0,.15,-0.15])
    x = np.array(normalize(x))
    """

    if x.shape[0] != 6890:
        print("NOT THE RIGHT SIZE!!")
        print(index)
        
    return x


def make_chicken_dataset():
    # Convert all .ply files into one numpy array and save it
    points = []

    f = open(r"dataset/dataset_chicken.npy", "wb")
    
    for i in range(1461,1672):
        print(i)
        x = (load_chicken(i),0)
        points.append(x)
        
    np.save(f, points)
    return

def show_chicken(i):
    # Plot face from the dataset

    f = np.load(open(r"dataset/dataset_chicken.npy", "rb"),allow_pickle=True)
    draw_point_cloud(Tensor(f[i][0]))  


####################
# MetaCircle   #####
####################  

def make_circle_dataset():
    # Make dataset of metaballs and save latent coordinates
    points = []

    f = open(r"dataset/dataset_9DCircleLATENT.npy", "wb")
    
    for i in range(0,500):
        print(i)
        x = shape_maker1(2,500, save_latent = True)
        points.append(x)
        
    np.save(f, points)
    return


def show_circle(i):
    # Plot face from the dataset

    f = np.load(open(r"dataset/dataset_9DCircleLATENT.npy", "rb"),allow_pickle=True)
    print(Tensor(f[i][0]).shape)
    print(Tensor(f[i][1]))
    draw_point_cloud(Tensor(f[i][0]))
    

def make_circle_dataset_new():
    # Make dataset of metaballs without latent coordinates
    points = []

    f = open(r"dataset/dataset_circle500.npy", "wb")
    
    for i in range(0,500):
        print(i)
        x = (make_circle().numpy(),0)
        points.append(x)
        
    np.save(f, points)
    return


def show_circle_new(i):
    # Plot face from the dataset

    f = np.load(open(r"dataset/dataset_circle500.npy", "rb"),allow_pickle=True)
    draw_point_cloud(Tensor(f[i][0]))



show_circle(145)
