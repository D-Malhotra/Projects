"""
Project 4 code
CID: Add your CID here
"""

import numpy as np
import matplotlib.pyplot as plt
#use scipy as needed


def load_image(normalize=True,display=False):
    """"
    Load and return test image as numpy array
    """
    import scipy
    from scipy.datasets import face
    A = face()
    if normalize:
        A = A.astype(float)/255
    if display:
        plt.figure()
        plt.imshow(A)
    return A


#---------------------------
# Code for Part 1
#---------------------------
def decompose1(A,eps):
    """
    Implementation of Algorithm 3 from KCSM
    Input:
    A: tensor stored as numpy array 
    eps: accuracy parameter
    Output:
    Glist: list containing core matrices [G1,G2,...]
    """
    #Initialise parameters
    R = 1
    N = len(np.shape(A))
    delta = eps * np.linalg.norm(A)/(np.sqrt(N-1))
    Ashape = A.shape

    #Unfold A into A(1)
    Z = np.moveaxis(A, 0, 0).reshape((A.shape[0], -1))

    #Calculate core tensors - we have A x G1 x B in the 3 dimsensional tensor case
    Glist = []
    for n in range(0, N-1):
        #Compute SVD decomp
        U, S, V = np.linalg.svd(Z, full_matrices=False)
        #Determine desired number of components
        cum_var = np.cumsum(S)/np.sum(S)
        num_comp = np.argmax(cum_var > 1-delta)
        #Define S with the number of components we want 
        U = U[:, :num_comp]
        S = np.diag(S[:num_comp])
        V = V[:num_comp, :]
        #Estimate of n-th TT rank
        Rn = np.shape(U)[1]
        #Compute G tensor (calculates A on first run)
        G = U.reshape(R, Ashape[n], Rn)
        Z = (S@V).reshape(Rn*Ashape[n+1], int(np.product(Ashape[n+2:])))
        R = Rn
        Glist.append(G)
    
    #Compute final tensor
    B = Z.reshape(R, Ashape[-1])
    Glist.append(B)
    #Reshape first tensor 
    Glist[0] = Glist[0][0, :, :]

    return Glist

def reconstruct(Glist):
    """
    Reconstruction of tensor from TT decomposition core matrices
    Input:
    Glist: list containing core matrices [G1,G2,...]
    Output:
    Anew: reconstructed tensor stored as numpy array
    """
    Anew = np.tensordot(Glist[0], Glist[1], axes =([-1], [0]))
    for g in Glist[2:]:
        Anew = np.tensordot(Anew, g, axes=([-1], [0]))
    
    return Anew

def decompose2(A,Rlist):
    """
    Implementation of modified Algorithm 3 from KCSM with rank provided as input
    Input:
    A: tensor stored as numpy array 
    Rlist: list of values for rank, [R1,R2,...]
    Output:
    Glist: list containing core matrices [G1,G2,...]
    """
    #Initialise parameters
    R = 1
    N = len(np.shape(A))
    Ashape = A.shape

    #Unfold A into A(1)
    Z = np.moveaxis(A, 0, 0).reshape((A.shape[0], -1))

    #Calculate core tensors - we have A x G1 x B in the 3 dimsensional tensor case
    Glist = []
    for n in range(0, N-1):
        #Compute SVD decomp
        U, S, V = np.linalg.svd(Z, full_matrices=False)
        #Desired number of components
        num_comp = Rlist[n]
        #Define S with the number of components we want 
        U = U[:, :num_comp]
        S = np.diag(S[:num_comp])
        V = V[:num_comp, :]
        #Estimate of n-th TT rank
        Rn = np.shape(U)[1]
        #Compute G tensor (calculates A on first run)
        G = U.reshape(R, Ashape[n], Rn)
        Z = (S@V).reshape(Rn*Ashape[n+1], int(np.product(Ashape[n+2:])))
        R = Rn
        Glist.append(G)
    
    #Compute final tensor
    B = Z.reshape(R, Ashape[-1])
    Glist.append(B)
    #Reshape first tensor 
    Glist[0] = Glist[0][0, :, :]
    return Glist




def part1():
    """
    Add code here for part 1, question 2 if needed
    """

    #Add code here

    return None #modify as needed


#-------------------------
# Code for Part 2
#-------------------------
from hottbox.core import Tensor


def part2(A, Rlist):
    def hosvd():
        """
        #HOSVD ALGORITHM
        Inputs
        A - Numpy array of data 
        Rlist - List of ranks for HOSVD algorithm

        Output
        recon - Reconstructed data
        """
        #make data tensor form
        tensor = Tensor(A)
        #Decompose X in each dimension 
        G = tensor
        Ulist = []
        for i in range(len(Rlist)):
            X = tensor.unfold(mode=i, inplace = False)
            U, S, V = np.linalg.svd(X.data, full_matrices=False)
            #Define U with the number of components we want 
            num_comp = Rlist[i]
            U = U[:, :num_comp]
            Ulist.append(U)
            #Reconstruct G
            G = G.mode_n_product(U.T, mode = i)

        #Reconstructed Image 
        recon = G
        for i in range(len(Ulist)):
            recon = recon.mode_n_product(Ulist[i], mode = i)

        return recon.data
    
    def lr(A, n):
        #Low-rank algorithm
        Anew = np.zeros(A.shape)
        for i in range(A.shape[2]):
            U, S, V = np.linalg.svd(A[:, :, i], full_matrices=False)
            Anew[:, :, i] = U[:, :n] @ np.diag(S[:n]) @ V[:n, :]
        return Anew

    def lr_vid(A, n):
        #Low-rank algorithm for videos 
        vid_reconstruct = np.zeros(A.shape)
        for frame in range(A.shape[0]):
            vid_reconstruct[frame, :, :, :] = lr(A[frame, :, :, :], n)
        return vid_reconstruct
    
    ##Image Reconstruction 
    # Image reconstructions 
    r = 100

    # Create a figure and a 3x1 subplot
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # HOSVD 
    hosvd_recon = hosvd(A, [r,r, 3])
    axs[0].imshow(hosvd_recon)
    axs[0].set_title('HOSVD')
    axs[0].axis("off")

    error0 = np.linalg.norm(A - hosvd_recon)

    # Test of code
    Rlist = [r,3]
    Glist = decompose2(A, Rlist)

    ttsvd_recon = reconstruct(Glist)
    axs[1].imshow(ttsvd_recon)
    axs[1].set_title('TTSVD')
    axs[1].axis("off")

    error1 = np.linalg.norm(A - ttsvd_recon)

    lr_recon = lr(A, 19)
    error2 = np.linalg.norm(A - lr_recon)
    axs[2].imshow(lr_recon)
    axs[2].set_title('Low-Rank')
    # Display the figure with subplots
    axs[2].axis("off")
    axs[0].text(0.5, -0.1, f"Error: {error0}", size=12, ha="center", 
                    transform=axs[0].transAxes)
    axs[1].text(0.5, -0.1, f"Error: {error1}", size=12, ha="center", 
                    transform=axs[1].transAxes)
    axs[2].text(0.5, -0.1, f"Error: {error2}", size=12, ha="center", 
                    transform=axs[2].transAxes)
    plt.tight_layout()
    plt.savefig("Image_recon.png")
    plt.show()

    import time 

    times = []
    for r in range(50, 300, 50):
        hts = time.time()
        hosvd_recon = hosvd(vid[:r, :, :, :], [100, 100, 3])
        hte = time.time()
        Glist = decompose2(vid[:r, :, :, :], [100, 100, 3])
        ttsvd_recon = reconstruct(Glist)
        tte= time.time()
        lr_recon = lr_vid(vid[:r, :, :, :], 100)
        lre = time.time()
        times.append([hte-hts, tte-hte, lre-tte])

    times = np.array(times)
    plt.plot([i for i in range(50, 300, 50)], times[:, 0], label = "HOSVD")
    plt.plot([i for i in range(50, 300, 50)], times[:, 1], label = "TTSVD")
    plt.plot([i for i in range(50, 300, 50)], times[:, 2], label = "LR")
    plt.xlabel("Frames")
    plt.ylabel("Time")
    plt.title("The effect of Video length on Time")
    plt.legend()
    plt.savefig("time_algos")
    plt.show()




def video2numpy(fname='project4.mp4'):
    """
    Convert mp4 video with filename fname into numpy array
    """
    import cv2
    cap = cv2.VideoCapture(fname)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    A = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, A[fc] = cap.read()
        fc += 1

    cap.release()
    
    return A.astype(float)/255 #Scales A to contain values between 0 and 1

def numpy2video(output_fname, A, fps=30):
    """
    Convert numpy array A into mp4 video and save as output_fname
    fps: frames per second.
    """
    import cv2
    video_array = A*255 #assumes A contains values between 0 and 1
    video_array  = video_array.astype('uint8')
    height, width, _ = video_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_fname, fourcc, fps, (width, height))

    for frame in video_array:
        out.write(frame)

    out.release()

    return None

#----------------------
if __name__=='__main__':
    pass
    #out = part2() Uncomment and modify as needed after completing part2
