from networks import *

def gradient(inputs, outputs):
    
    #  !! COPYRIGHT for this function : https://github.com/amosgropp/IGR/blob/master/code/model/network.py  !!

    # Returns:
    #   Pointwise gradient estimation [ Df(x_i) ]
    
    # Parameters:
    #   inputs:     [ x_i ]
    #   outputs:    [ f(x_i) ] 
    
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    # No idea why this works but it does 
    points_grad = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=d_points, create_graph=True, retain_graph=True, only_inputs=True)[0][:,-3:]
    return points_grad

#############################
# PHASE - Loss ##############
#############################

# double well potential
#W = lambda s: s**2 - 2.0*torch.abs(s) + torch.tensor([1.0]).to(device) 
W = lambda s: (9.0/16.0) *  (s**2 -torch.tensor([1.0]).to(device)   )**2


def ModicaMortola(f, eps, n, d):
    # Returns:
    #   Monte Carlo Integral of int_{[0,1]^2} W(u(x)) + eps * |Du(x)|^2 dx
    
    # Parameters:
    #   f:      Function to evaluate
    #   eps:    Epsilon
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
     
    start_points = Variable(torch.rand(n, d), requires_grad =True)-torch.full(size=(n,d), fill_value=.5)   # Create random points [ x_i ]
    start_points = start_points.to(device)                          # Move points to GPU if possible
    gradients = gradient(start_points, f(start_points))             # Calculate their gradients [ Dx_i ]
    norms = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    return (W(f(start_points))+eps*norms).mean()                    # returns 1/n * sum_{i=1}^n W(u(x_i)) + eps * |Du(x_i)|^2


def Zero_recontruction_loss_Lip(f, pc, eps, m, c, d):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
    
    n = len(pc)
    
    matrix = pc.repeat(m,1).to(device) 
    matrix = torch.reshape(matrix, (m,n,d))             # 3D Matrix containing the points
    variation  = torch.normal(mean = torch.full(size=( n*m *d,1), fill_value=0.0) , std= torch.full(size=(m*n*d,1), fill_value=.001) ).to(device) 
    error = torch.reshape( variation, (m,n, d) )        # 3D Matrix containing normal distribution
    matrix += error
    matrix = matrix.reshape(m*n,d)
    matrix = f(matrix)                                  # Apply network to targets
    matrix = torch.reshape( matrix, (m,n) ).mean(0)
    matrix = torch.abs(matrix).mean()

    return  c*eps**(1.0/3.0) *  matrix                  # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} u(x) dx|


def Eikonal_loss(f, pc, eps, d):
    # Returns:
    #   Eikonal loss around the points of point cloud
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X = [ x_i ]
    #   eps:    Epsilon
    
    gradients = gradient( pc, f(pc) )                                                                       # calculates [ Du(x) ]
    norms = np.sqrt(eps) *  gradients.norm(2,dim=-1)                                                        # calculates [ |Dw(x)| ] = [ sqrt(eps) * |Du(x)| ]
    eikonal = torch.abs(torch.full(size=(len(pc) ,1), fill_value=1.0).to(device)-norms.to(device))**2       # calculates [ | 1-|Dw(x)| |^2 ] 
    
    return eikonal.mean()                                                                                   # return \sum_{x\in X} |1-|Dw(u) | |^2                

def Phase_loss(f, pointcloud, eps, n, m, c, mu):
    # Returns:
    #   PHASE Loss = e^(-.5)(\int_\Omega W(u) +e|Du|^2 + Ce(^.3)/(n) sum_{p\in P} \dashint u ) + \mu/n \sum_{p\in P} |1-|w||
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X = [ x_i ]
    #   eps:    Epsilon
    #   n:      Number of Sample for Monte-Carlo in int_\Omega
    #   m:      Number of Sample for Monte-Carlo in int_{B_\delta}
    #   c:      Constant C, contribution of Zero recontruction loss
    #   mu:     Constant \mu, contribution of Eikonal equation
    
    d = pointcloud.shape[1] # dimension of point cloud
    
    return eps**(-.5)*(ModicaMortola(f, eps, n, d) +  Zero_recontruction_loss_Lip(f, pointcloud, eps, m, c, d))+mu * Eikonal_loss(f, pointcloud, eps, d )


def test_MM_GV(f, pc, eps, n, m, c, p):
    # Compute different contributions from the three MM terms to the total loss
    d = pc.shape[1]
    start_points = Variable(torch.rand(n, d), requires_grad =True)-torch.full(size=(n,d), fill_value=.5)   # Create random points [ x_i ]
    start_points = start_points.to(device)                          # Move points to GPU if possible
    gradients = gradient(start_points, f(start_points))             # Calculate their gradients [ Dx_i ]
    norms = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    
    (W(f(start_points))+eps*norms).mean()   
    EINS = 1.0/(eps) * W(f(start_points)).mean()
    ZWEI  = (eps * norms).mean()
    n = len(pc)
    
    matrix = pc.repeat(m,1).to(device) 
    matrix = torch.reshape(matrix, (m,n,d))         # 3D Matrix containing the points
    variation  = torch.normal(mean = torch.full(size=( n*m *d,1), fill_value=0.0) , std= torch.full(size=(m*n*d,1), fill_value=.001) ).to(device) 
    error = torch.reshape( variation, (m,n, d) )    # 3D Matrix containing normal distribution
    matrix += error
    matrix = matrix.reshape(m*n,d)
    matrix = f(matrix)                              # Apply network to targets
    matrix = torch.reshape( matrix, (m,n) ).mean(0)
    DREI = c * eps**(-1.0/3.0) * torch.abs(matrix).mean()
    if p:
        print("1: ",EINS,"2: ",ZWEI,"3: ",DREI)
    return  EINS+ZWEI+DREI


#############################
# Ambrosio Tortorelli #######
#############################

# One well potential
U = lambda s: (s- torch.tensor([1.0]).to(device))**2

# Shifting function
g = lambda s: s**2


def AT_Phasefield(f, eps, n, d):
    # Returns:
    #   Monte Carlo Integral of int_{[0,1]^2} W(u(x)) + eps * |Du(x)|^2 dx
    
    # Parameters:
    #   f:      Function to evaluate
    #   eps:    Epsilon
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
     
    start_points = Variable(torch.rand(n, d), requires_grad =True)-torch.full(size=(n,d), fill_value=.5)   # Create random points [ x_i ]
    start_points = start_points.to(device)                          # Move points to GPU if possible
    gradients = gradient(start_points, f(start_points))             # Calculate their gradients [ Dx_i ]
    norms = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    return ( (1.0/(4*eps))  * U(f(start_points))+eps*norms).mean()                    # returns 1/n * sum_{i=1}^n W(u(x_i)) + eps * |Du(x_i)|^2



def Zero_recontruction_loss_AT(f, pc, eps, m, c, d):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    #   m:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
    

    return  c*eps**(-1.0/3.0) *  ( torch.abs(f(pc)).mean() )            # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} g( u(x) ) dx|


def Zero_recontruction_loss_AT_Shift(f, pc, eps, m, c, d):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
    
    n = len(pc)
    
    matrix = pc.repeat(m,1)
    matrix = torch.reshape(matrix, (m,n,d))         # 3D Matrix containing the points
    variation  = torch.normal(mean = torch.full(size=( n*m *d,1), fill_value=0.0) , std= torch.full(size=(m*n*d,1), fill_value=.001) )
    error = torch.reshape( variation, (m,n, d) )    # 3D Matrix containing normal distribution
    matrix += error
    matrix = matrix.reshape(m*n,d)
    matrix = g(f(matrix))                           # Apply network to targets and shift values
    matrix = torch.reshape( matrix, (m,n) ).mean(0)
    matrix = torch.abs(matrix).mean()

    return  c*eps**(-1.0/3.0) *  matrix             # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} g( u(x) ) dx|



def AT_loss(f, pointcloud, eps, n, m, c):
    # Returns:
    #   PHASE Loss = e^(-.5)(\int_\Omega W(u) +e|Du|^2 + Ce(^.3)/(n) sum_{p\in P} \dashint u ) + \mu/n \sum_{p\in P} |1-|w||
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X = [ x_i ]
    #   eps:    Epsilon
    #   n:      Number of Sample for Monte-Carlo in int_\Omega
    #   m:      Number of Sample for Monte-Carlo in int_{B_\delta}
    #   c:      Constant C, contribution of Zero recontruction loss
    
    d = pointcloud.shape[1] # dimension of point cloud
    
    return AT_Phasefield(f, eps, n, d) +  Zero_recontruction_loss_AT(f, pointcloud, eps, m, c, d)



#############################
# Loss on L^2 ###############
#############################


def L2_Loss(f, input, Batch):
    
    # Returns:
    #   Integral over manifold \int_S f(x) dx
    
    # Parameters:
    #   f:      Function to evaluate
    #   input:  Points on the manifold
    #   Batch:  Number of integral evaluations

    indices = np.random.choice(len(input), Batch, False)
    x = Variable( Tensor(input[indices])).to(device)

    return f(x).mean()


def sobolev(f,g, MCS, Tau):
    start_points = Variable(torch.rand(MCS, 2), requires_grad =True)-torch.full(size=(MCS,2), fill_value=.5)   # Create random points [ x_i ]
    start_points = start_points.to(device)                          # Move points to GPU if possible
    L2 = (f(start_points) - g(start_points))**2
  
    gradients_f = gradient(start_points, f(start_points))             
    gradients_g = gradient(start_points, g(start_points))
    gradients = gradients_f - gradients_g
    H1 = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    return L2.mean() + Tau * H1.mean()




#####################################
# Shapespace Learning ###############
#####################################


#
#       Enter feature vector into the Fourier Features
#


def AT_Phasefield_shapespace(f, eps, n, d, fv):
    # Returns:
    #   Monte Carlo Integral of int_{[0,1]^2} W(u(x)) + eps * |Du(x)|^2 dx
    
    # Parameters:
    #   f:      Function to evaluate
    #   eps:    Epsilon
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
    #   fv:     Feature Vector
     
    start_points = Variable(torch.rand(n, d), requires_grad =True).to(device)-torch.full(size=(n,d), fill_value=.5).to(device)    # Create random points [ x_i ]
    features = fv.repeat(n,1)
    start_points = torch.cat((start_points, features), 1)
    start_points = start_points.to(device)                          # Move points to GPU if possible
    gradients = gradient(start_points, f(start_points))             # Calculate their gradients [ Dx_i ]
    norms = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    return ( (1.0/(4*eps))  * U(f(start_points))+eps*norms).mean()                    # returns 1/n * sum_{i=1}^n W(u(x_i)) + eps * |Du(x_i)|^2



def Zero_recontruction_loss_AT_shapespace(f, pc, eps, c, fv):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    #   fv:     Feature Vector
    
    n = pc.shape[0]
    features = fv.repeat(n,1)    
    pc = torch.cat((pc, features), 1)
    points = torch.abs(f(pc))**2

    return  c*eps**(-1.0/3.0) *  ( points.mean() )            # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} g( u(x) ) dx|


def AT_loss_shapespace(f, pointcloud, eps, n, c, fv):
    # Returns:
    #   PHASE Loss = e^(-.5)(\int_\Omega W(u) +e|Du|^2 + Ce(^.3)/(n) sum_{p\in P} \dashint u ) + \mu/n \sum_{p\in P} |1-|w||
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X = [ x_i ]
    #   eps:    Epsilon
    #   n:      Number of Sample for Monte-Carlo in int_\Omega
    #   m:      Number of Sample for Monte-Carlo in int_{B_\delta}
    #   c:      Constant C, contribution of Zero recontruction loss
    #   fv:     Feature Vector
    
    d = pointcloud.shape[1] # dimension of point cloud
    
    return AT_Phasefield_shapespace(f, eps, n, d, fv) +  Zero_recontruction_loss_AT_shapespace(f, pointcloud, eps, c, fv)


#
#       Concatenate Feature vector after Fourier Features
#



def AT_Phasefield_shapespace2(f, eps, n, d, fv):
    # Returns:
    #   Monte Carlo Integral of int_{[0,1]^2} W(u(x)) + eps * |Du(x)|^2 dx
    
    # Parameters:
    #   f:      Function to evaluate
    #   eps:    Epsilon
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
    #   fv:     Feature Vector
     
    start_points = Variable(torch.rand(n, d), requires_grad =True).to(device)-torch.full(size=(n,d), fill_value=.5).to(device)  # Create random points [ x_i ]
    features = fv.repeat(n,1)                                       # Duplicate feature vector
    
    start_points = start_points.to(device)                          # Move points to GPU if possible
    gradients = gradient(start_points, f(start_points, features))   # Calculate their gradients [ Dx_i ]
    norms = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    return ( (1.0/(4*eps))  * U(f(start_points, features))+eps*norms).mean()            # returns 1/n * sum_{i=1}^n W(u(x_i)) + eps * |Du(x_i)|^2



def Zero_recontruction_loss_AT_shapespace2(f, pc, eps, c, fv):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    #   fv:     Feature Vector
    
    n = pc.shape[0]
    features = fv.repeat(n,1)    
    points = torch.abs(f(pc, features))**2

    return  c*eps**(-1.0/3.0) *  ( points.mean() )            # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} g( u(x) ) dx|


def AT_loss_shapespace2(f, pointcloud, eps, n, c, fv):
    # Returns:
    #   PHASE Loss = e^(-.5)(\int_\Omega W(u) +e|Du|^2 + Ce(^.3)/(n) sum_{p\in P} \dashint u ) + \mu/n \sum_{p\in P} |1-|w||
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X = [ x_i ]
    #   eps:    Epsilon
    #   n:      Number of Sample for Monte-Carlo in int_\Omega
    #   m:      Number of Sample for Monte-Carlo in int_{B_\delta}
    #   c:      Constant C, contribution of Zero recontruction loss
    #   fv:     Feature Vector
    
    d = pointcloud.shape[1] # dimension of point cloud
    
    return AT_Phasefield_shapespace2(f, eps, n, d, fv) +  Zero_recontruction_loss_AT_shapespace2(f, pointcloud, eps, c, fv)
