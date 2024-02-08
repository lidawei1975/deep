# Deep Picker Suite

A bunch of programs that belong to the Deep Picker suite.  Dawei can fill this in with something useful.

# Build

For local building of this application, please utilize the ccic/spindocker project and the associated 
image at `code.osu.edu:5000/ccic/spindocker/spindev:latest`.

# Deployment

As this project is updated, built binaries of the program will be deployed to `/srv/www/orapps/exec/deep-picker/build/`.
Merge requests can be tested prior to approval by adding a `-#` to the deep-picker part of the path, with # being the
merge request ID.  For example, merge request #2 would have the path of `/srv/www/orapps/exec/deep-picker-2/build/`.
Merge build directories will be removed as they are merged into the master branch.

Only files that are specified in the artifacts section of the `.gitlab-ci.yml` file in this project will be deployed
to the server.  By default, this is the build directory, but you may include any other files that are required for this
program to run as well.