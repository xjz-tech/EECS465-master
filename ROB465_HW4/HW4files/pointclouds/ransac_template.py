#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###

###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')


    ###YOUR CODE HERE###
    # Show the input point cloud
    utils.view_pc([pc])

    #Fit a plane to the data using ransac



    #Show the resulting point cloud

    #Draw the fitted plane


    ###YOUR CODE HERE###
    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
