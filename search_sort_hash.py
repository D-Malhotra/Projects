"""
Code for Scientific Computation Project 1
Please add college id here
CID:01857554
"""
import matplotlib.pyplot as plt
import numpy as np
import time

#===== Code for Part 1=====#
def part1(Xin,istar):
    """
    Sort list of integers in non-decreasing order
    Input:
    Xin: List of N integers
    istar: integer between 0 and N-1 (0<=istar<=N-1)
    Output:
    X: Sorted list
    """
    X = Xin.copy() 
    #i is the index of the value x in the list ignoring the first value 
    for i,x in enumerate(X[1:],1):
        #i less than some index istar
        if i<=istar:
            #ind is the new location to place the value x 
            ind = 0
            #start, stop, step 
            for j in range(i-1,-1,-1):
                #if x is GOE any value before it save the location it should be placed at(ind)
                if x>=X[j]:
                    ind = j+1
                    break                   
        else:
        #for values below istar
            a = 0
            b = i-1
            while a <= b:
                c = (a+b) // 2
                #find a point halfway in the list to compare x with
                #if x is larger than the halway value we split that half into another half 
                if X[c] < x:
                    a = c + 1
                else:
                    #otherwise we split the lower half into half by redefining b 
                    b = c - 1
            #the final index value we want will be a 
            ind = a
        #set the list from ind + 1 to i (i+1 not inclusive) equal to the section of the list before it 
        X[ind+1:i+1] = X[ind:i]
        #we get a repeat value in the location where we want x, so we set that value = x
        X[ind] = x

    return X


def part1_time():
    """Examine dependence of walltimes of part1 function on N and istar
        You may modify the input/output as needed.
    """
    time_list = [[], [], []]
    #considering values of istar  
    for N in range(1, 1000):
        istar_list = [0, N//2, N-1]
        #Here we can pick how we want to define Xin to compare best/worst case costs
        #Xin = np.random.choice(N, N, replace= False)
        Xin = np.linspace(1, N, N)
        for i in range(3):
            t_1 = time.time()
            part1(Xin, istar_list[i])
            t_2 = time.time()
            time_taken = t_2 - t_1 
            time_list[i].append(time_taken)
    #The case where 
    for i in range(3):
        plt.plot(np.linspace(1, 999, 999), time_list[i])
        np.polyfit(np.linspace(1, 999, 999), time_list[i], 1)
        plt.legend(['istar = 0', 'istar = N/2', 'istar = N-1'])
        plt.title('Comparing values of istar for N= 1 to 1000')
        plt.xlabel('N')
        plt.ylabel('Time')
        plt.grid(True)
        #plt.savefig('best_time_complexity.png')
        
    return  #Modify if needed

#length n list
time_list = part1_time()

#Time for varying istar with fixed N 
N = 1000
istar = np.arange(0, N)
time_list = []
Xin = np.random.choice(N, N, replace= False)
for i in istar:
    t_1 = time.time()
    part1(Xin, i)
    t_2 = time.time()
    time_taken = t_2 - t_1 
    time_list.append(time_taken)
plt.plot(istar, time_list)
plt.title('Time taken for varying istar with fixed N')
plt.xlabel('istar')
plt.ylabel('Time taken')
plt.grid(True)
#plt.savefig("Optimal_istar.png")
print(time_list.index(min(time_list)))
#===== Code for Part 2=====#

def char2base4(S):
    """Convert gene test_sequence
    string to list of ints
    """
    #initalise dictionary
    c2b = {}

    #Conversions for each gene letter
    c2b['A']=0
    c2b['C']=1
    c2b['G']=2
    c2b['T']=3
    L=[]
    #Loop over string
    for s in S:
        L.append(c2b[s])
    return L

def heval(L,Base,Prime):
    """Convert list L to base-10 number mod Prime
    where Base specifies the base of L
    """
    #Converts Hash to a base 10 integer
    a = 0
    for l in L[:-1]:
        a = Base*(l+a)
    b = (a + (L[-1])) % Prime
    return b

def part2(S, T, m):
    """Find locations in S of all length-m sequences in T
    Input:
    S,T: length-n and length-l gene sequences provided as strings

    Output:
    L: A list of lists where L[i] is a list containing all locations 
    in S where the length-m sequence starting at T[i] can be found.
   """
    # Size parameters
    n = len(S) 
    l = len(T) 
    base = 4
    # Initialise list of lists L
    L = [[] for i in range(l - m + 1)]

    # Convert gene sequence string to list of ints
    X = char2base4(S)
    Y = char2base4(T)
    # Large prime number taken from Google
    prime = 393050634124102232869567034555427371542904833
    # Store base for rolling hash
    bm = (base ** m) % prime
    #Store hash value as the key and the indices of the string as the value
    Y_hash_dict = {}
    # Find hash value for the first m-length substring
    hash_val = heval(Y[:m],base,prime) 
    # Add to dictionary with index
    Y_hash_dict[hash_val] = [0]
    # Use rolling hash to fill rest of the indices
    for i in range(l - m):
        hash_val = (4 * hash_val - int(Y[i]) * bm + int(Y[i + m])) % prime
        # If hash values are repeated then append all indices for that hash
        if hash_val in Y_hash_dict:
            Y_hash_dict[hash_val].append(i+1)
        else:
            Y_hash_dict[hash_val] = [i+1]
    # Compute hash for first m-length substring of X
    h = heval(X[:m],base,prime)
    if h in Y_hash_dict:
        # If the hash is in the dictionary we check its index with Y_hash_dict[hi]
        # We must also check if it is not a hash collision by comparing lists
        for j in Y_hash_dict[h]:
                if S[:m] == T[j:j + m]:
                    # If the strings match in the index, append to the correct list in L
                    L[j].append(0)
                
    #We use rolling hash for rest of the comparisons
    for i in range(n-m):
    #Update rolling hash
        h = (4*h - int(X[i])*bm + int(X[i+m])) % prime
        if h in Y_hash_dict: 
            # If hashes match, check if strings match
            # If two or more strings are assigned the same has, then check all such strings
            for j in Y_hash_dict[h]:
                if S[i+1:1+i+m]== T[j:j+m]:
                    # Append if there is a match
                    L[j].append(i+1)
                
    return L
       


if __name__=='__main__':
    #Small example for part 2
    S = 'ATCGTACTAGTTATCGT'
    T = 'ATCGT'
    m = 3
    out = part2(S,T,m)
    print(out)
    #Large gene sequence from which S and T test sequences can be constructed
    infile = open("words.txt") #file from lab 3
    sequence = infile.read()
    infile.close()
