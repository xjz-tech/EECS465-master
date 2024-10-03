import matplotlib.pyplot as plt
import numpy as np

#this function plots the hyperplanes
def DrawHyperplanes(hyperplanes):
    linescale = 2
    plt.figure(1)
    numplanes = hyperplanes.shape[0]
    x0 = np.asmatrix(np.zeros((2,numplanes)))
    for i in range(0,numplanes):
        a = hyperplanes[i][:,0:2].T
        b = hyperplanes[i,2]
        x0[:,i] = (np.linalg.pinv(a)*b).T #basically the same as x0[:,i] = inv(a'*a)*a'*b
        plt.plot([x0[0,i]],[x0[1,i]],'bo')
        #plt.plot([x0[0,i], (x0[0,i]+a[0])],[x0[1,i], (x0[1,i]+a[1])],'b') #arrow without a tip
        plt.arrow(x0[0,i], x0[1,i], a[0,0], a[1,0], head_width=0.05, head_length=0.1, fc='b', ec='b') #arrow with tip
        plt.plot([x0[0,i], (x0[0,i]-linescale*a[1,0])],[x0[1,i],(x0[1,i]+linescale*a[0,0])],'r')
        plt.plot([x0[0,i], (x0[0,i]+linescale*a[1,0])],[x0[1,i],(x0[1,i]-linescale*a[0,0])],'r')
        
    plt.axis('equal')
    

#this is the main function where this script begins to execute
if __name__ == "__main__":
    #define parameters to be used in the solver
    t = 0.1 #this is the "force" with which the optimization function "pulls" on the current point
    mu = 1.5 #this is how much to scale t at each outerloop iteration
    epsilon = 0.001 #this is the desired precision of the outer loop
    newton_epsilon = 0.001 #this is the desired precision of Newton's method (inner loop)
    alpha = 0.1 #for back-tracking line search
    beta = 0.6 #for back-tracking line search

    #these are defined as [a b]
    hyperplanes = np.asmatrix([[0.7071,    0.7071, 1.5],
                    [-0.7071,    0.7071, 1.5],
                    [0.7071,    -0.7071, 1],
                    [-0.7071,    -0.7071, 1]])


    #number of hyperplanes in this problem
    numplanes = hyperplanes.shape[0]

    #the optimization function c:
    c = np.asmatrix([2, 1]).T

    #pick a starting point (this can be done autonomously but we'll do it by hand)
    x = np.asmatrix([0, 0]).T

    #now draw the constraining hyperplanes
    DrawHyperplanes(hyperplanes)

    #let's break down the data into variables we are familiar with
    a = hyperplanes[:][:,0:2].T # each column is the "a" part of a hyperplane
    b = hyperplanes[:,2] # each row is the "b" part of a hyperplane (only one element in each row)

    #plot the starting point
    plt.plot(x[0,0],x[1,0], 'gx')

    num_outer_iterations = 0
    num_inner_iterations = 0
    

    ###############Start outer loop (Barrier Method)#####################
    while 1:
        num_outer_iterations = num_outer_iterations + 1

        ###############Start inner loop (Newton's Method)#####################
        num_inner_iterations = 0
        while 1:
            num_inner_iterations = num_inner_iterations + 1

            #now start computing f' (fprime), which is the sum of the optimization force 
            #and the forces pushing away from the barriers
            #you will also need to compute the second derivative f'' (fprimeprime) in the same way

            #compute fprime for just the optimization force first
            fprime = t * c###YOUR CODE HERE###
            
            #compute fprimeprime for just the optimization force first
            fprimeprime = 0 ###YOUR CODE HERE###

            #compute the first and second derivatives from each hyperplane and aggregate
            for j in range(0,numplanes):
                fprime_for_plane_j = a[:, j] / (a[:, j].T @ x - b[j])###YOUR CODE HERE###

                fprimeprime_for_plane_j = (a[:, j] @ a[:, j].T) / (b[j] - a[:, j].T @ x) ** 2###YOUR CODE HERE###

                fprime = fprime - fprime_for_plane_j # put in the contribution of hyperplane j to fprime
                fprimeprime = fprimeprime + fprimeprime_for_plane_j # put in the contribution of hyperplane j to fprimeprime

            #you might want to print fprime and fprimeprime here to debug (but it will slow things down)

            #the step according to Newton's method (in terms of fprime and fprimeprime)
            step =  -np.linalg.pinv(fprimeprime) * fprime###YOUR CODE HERE###

            #compute the Newton decrement squared (in terms of step and fprimeprime)
            lambda2 =  fprime.T @ (-step)###YOUR CODE HERE###

            #check if we've reached the Newton's method stopping condition
            #if so, break out of Newton's method
            if(lambda2 / 2 <= newton_epsilon):
                break
    
            #now we have a direction to move the point x (i.e. the Newton step) but we don't 
            #know how far to move in that direction
            #so we look along the direction for the biggest step we can take which doesn't jump 
            #over a barrier or move to a higher-cost location
            #the method we use here is called back-tracking line search

            #back-tracking line search

            k = 1 #this is how much to scale the step, start with the original magnitude
            f = t*c.T*x
            for j in range(0,numplanes):
                f = f - np.log(-a[:,j].T*x + b[j])

            while 1:
                xnew = x + k*step
                fnew = t*c.T*xnew
                pastboundary = 0
                #check if we've jumped over a boundary
                for j in range(0,numplanes):
                    dist = -a[:,j].T*xnew + b[j]
                    if (dist < 0):
                        pastboundary = 1
                        break
                    fnew = fnew - np.log(dist)

                #use alpha and beta to generate new guess for how much to move along the step direction
                if(pastboundary or fnew > f + alpha * k * fprime.T * step):  #put in the check for terminating backtracking line search
                    #if we're not done
                    k = k * beta###YOUR CODE HERE###
                else:
                    break
            
            #now we have k, the amount to scale the step
            x = x + k*step
            #plot the new point for this Newton iteration in green
            plt.plot(x[0,0],x[1,0], 'gx')

        ###############End inner loop (Newton's Method)#####################
        #plot the new point for this outer loop iteration in red
        plt.plot(x[0,0],x[1,0], 'rx')

        print('OUTER loop iteration %d: Number of INNER loop iterations: %d\n'%(num_outer_iterations, num_inner_iterations))


        #compute the duality gap (in terms of numplanes and t)
        duality_gap = numplanes / t

        #If the duality gap is below our error tolerance (epsilon), we're done!
        if duality_gap < epsilon:
            break
    
        #now that we've figured out the optimal point for this amount of optimization "force," increase the optimization force to a larger value
        #compute the new optimization force magnitude
        t = t*mu

    ###############End outer loop (Barrier Method)#####################

    print("The optimal point: (%f, %f)\n"%(x[0,0], x[1,0]))
    print('Total number of outer loop iterations: %d\n'%(num_outer_iterations))

plt.show()      
    
