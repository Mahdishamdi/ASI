import numpy as np
from scipy.spatial.distance import cdist
import random
import math
import time



def lattice_points(N,s):
    
    l = ((N+1)/2-0.5)*s

    lattice_points = []

    for x in np.arange(-l+0.5,l+0.5,s):
        for y in np.arange(l-0.5,-l,-s):    
            lattice_points.append([x,y,0]) 

    xs = [i[0] for i in lattice_points]
    ys = [i[1] for i in lattice_points]

    return xs,ys,lattice_points

def Spin_config(N, N_input):

    rotation_angle = 0.0
    xComp = []
    yComp = []
    angle = 45
    
    for i in range(N):
        if i % 2 == 0:
            for j in range(N):
                if j % 2 == 0:
                    xComp.append(math.cos(math.radians(angle - rotation_angle)))
                    yComp.append(-math.sin(math.radians(angle - rotation_angle)))
                else:
                    xComp.append(math.cos(math.radians(angle + rotation_angle)))
                    yComp.append(math.sin(math.radians(angle + rotation_angle)))
        else:
            for j in range(N):
                if j % 2 == 0:
                    xComp.append(-math.cos(math.radians(angle + rotation_angle)))
                    yComp.append(-math.sin(math.radians(angle + rotation_angle)))
                else:
                    xComp.append(-math.cos(math.radians(angle - rotation_angle)))
                    yComp.append(math.sin(math.radians(angle - rotation_angle)))
    
    polarity_vector = [[] for i in range(N * N + N_input)]
    
    for i in range(N * N):
        q = random.random()
        if q <= 0.5:
            polarity_vector[i].extend([xComp[i], yComp[i], 0.0])
        else:
            polarity_vector[i].extend([-xComp[i], -yComp[i], 0.0])
    
    return polarity_vector

def distance_position_vector(N, lattice_points, N_input):
    lattice_points = np.array(lattice_points)
    vector_rij = np.empty((N*N+N_input, N*N+N_input, 3))
    for i in range(N*N+N_input):
        for j in range(N*N+N_input):
            vector_rij[i,j,:] = lattice_points[j] - lattice_points[i]   
    dists = cdist(lattice_points, lattice_points)
    
    return vector_rij, dists



def Energy_initial(N,N_input, temp, polarity_vector, dists, vector_rij): 
    const =1/temp
    Energy = 0.0
    polarity_vector=np.array(polarity_vector)
    for i in range(N*N+N_input):
        for j in range(N*N+N_input):
            if i != j:
                Energy += 0.5*(const*dists[i][j]**(-3)*np.dot(polarity_vector[i],polarity_vector[j]))-0.5*const*dists[i][j]**(-5)*3*np.dot(polarity_vector[i],vector_rij[i][j])*np.dot(polarity_vector[j],vector_rij[i][j])

    return Energy



def energy_cost(N,N_input,random_spin, temp, polarity_vector, dists, vector_rij):
    
    const = np.array(1 / temp)
    E_eff = np.zeros(3)

    polarity_vector = np.array(polarity_vector)
    vector_rij = np.array(vector_rij)
    for j in range(N*N+N_input):
        if random_spin != j:
            E_eff = E_eff + const * dists[random_spin][j]**(-3) * polarity_vector[j] - const * dists[random_spin][j]**(-5) * 3 * vector_rij[random_spin][j]* np.dot(polarity_vector[j],vector_rij[random_spin][j])
    
    delta_E = -2 * np.dot(E_eff, polarity_vector[random_spin])
    
    return delta_E


def MC(N,N_input, temp, Energy, polarity_vector, dists, vector_rij):
    k=1
    for q in range(N*N):
        random_spin = int(np.random.rand() * len(polarity_vector))
        delta_E = energy_cost(N,N_input,random_spin, temp, polarity_vector, dists, vector_rij)
        r = np.random.rand()
        #if delta_E < 0 or r < np.exp(-delta_E):
        if delta_E < 0 or r <np.exp(-delta_E):
            polarity_vector[random_spin] = [-x for x in polarity_vector[random_spin]]
            #Energy += delta_E

    return polarity_vector


def magnetization(polarity_vector):
    
    mag = np.sum(polarity_vector)
    
    return mag


def magnetostatic_field(r,temp, m, point):
    
    B = np.zeros(3)
    for i in range(len(r)):
        rvec = point - r[i]
        rnorm = np.linalg.norm(rvec)
        if rnorm == 0:
            continue
     
        B += (3*rvec*np.dot(m[i], rvec)/rnorm**5) - (m[i]/rnorm**3)
    B *= 1/temp
    return B


def vortex(N):
    vortex_points = []

    for j in range(int(N/2-1), -(int(N/2-1))-1, -1):
        if j % 2 == 0:
            for i in range(-(int(N/2-1)), int(N/2-1)+1, 2):
                vortex_points.append([i, j, 0.5])
        elif j % 2 == 1 or j % 2 == -1:
            for i in range(-(int(N/2-2)), int(N/2-2)+1, 2):
                vortex_points.append([i, j, 0.5])

    return vortex_points



N=6
N_input=0

temp=3.5
steps =60000
s=1

start_time = time.time()


polarity_vector=Spin_config(N, N_input)

xs,ys,lattice_points=lattice_points(N,s)

vector_rij, dists=distance_position_vector(N, lattice_points, N_input)

Energy=Energy_initial(N,N_input, temp, polarity_vector, dists, vector_rij)



vortex_points=np.array(vortex(N))



e=[]
Data=[]
B=[]
mx=[]
my=[]

for i in range(steps):
    M=[]   
    xcomp=[]
    ycomp=[]
    D=[]    
    polarity_vector_new  = MC(N,N_input, temp, Energy, polarity_vector, dists, vector_rij)
    Energy=Energy_initial(N,N_input, temp, polarity_vector_new, dists, vector_rij)

    
    for j in range(len(polarity_vector_new)):
        D.append(polarity_vector_new[j])
#        M.append(polarity_vector_new[j])
#        xcomp.append(polarity_vector_new[j][0])
#        ycomp.append(polarity_vector_new[j][1])
    vb=0
#    for k in range(len(vortex_points)):
        
#        vb+=(magnetostatic_field(lattice_points,temp, M, vortex_points[k]))
    if i>=15000:
        e.append(Energy)
#        B.append(vb)
#        mx.append(magnetization(xcomp))
#        my.append(magnetization(ycomp))
        Data.append(D)

end_time = time.time()
TIME = end_time - start_time
print(TIME)


#Bz = [b[2] for b in B]
E  = np.array(e)
#Mx = np.array(mx)
#My = np.array(my)
#Hz = np.array(Bz)
np.save('pv_6lat_T=3.5_60k_2',Data)
#np.save("lattice_points",lattice_points)
np.save('E_6lat_T=3.5_60k_2',E)
#np.save('Mx_6lat_T=3.5_60k_1',Mx)
#np.save('My_6lat_T=3.5_60k_1',My)
#np.save('Hz_6lat_T=3.5_60k_1',Hz)
