cmake_minimum_required(VERSION 3.5)
project(libcerf)

add_library(libcerf STATIC  erfcx.c  err_fcts.c  im_w_of_x.c  w_of_z.c)