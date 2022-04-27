# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 23:03:49 2022

@author: TD
"""

# Variational Auto Encoder (VAE) for 2D datasets

This project will consider the simple problem of a polynomial curve in R^2.  
Thus, the input would have shape (num_points, 2).  Where the columns 
will hold the (x, y) coordinate values and each row is for each point in the 
curve.

Since this is a VAE that is built around 2-dimensional input data, the idea 
is to supply examples of n-dim surfaces.  The input matrix will have shape 
(num_points, num_dim).