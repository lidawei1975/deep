cmake_minimum_required(VERSION 3.5)

project(deep_picker)



add_library(spectrum OBJECT spectrum_io.cpp spectrum.cpp dnn_picker.cpp clique.cpp ann_data.c)
add_library(spectrum_d OBJECT spectrum_io.cpp spectrum.cpp dnn_picker.cpp clique.cpp ann_data.c)
target_compile_definitions(spectrum_d PUBLIC LDW_DEBUG)
add_library(commandline OBJECT commandline.cpp )



# 2D peak picking
add_executable(deep_picker main_peak_picking.cpp $<TARGET_OBJECTS:spectrum> $<TARGET_OBJECTS:commandline> )


# 2D peak picking with debug output
add_executable(deep_picker_d main_peak_picking.cpp $<TARGET_OBJECTS:spectrum_d> $<TARGET_OBJECTS:commandline> )


# 1d peak picking
add_executable(deep_picker_1d main_1d_picking.cpp  $<TARGET_OBJECTS:spectrum> $<TARGET_OBJECTS:commandline> )

