# Slow Flow: Generating Optical Flow Reference Data

This code is based on the paper [Slow Flow: Exploiting High-Speed Cameras for
Accurate and Diverse Optical Flow Reference Data](http://www.cvlibs.net/publications/Janai2017CVPR.pdf).

The high speed flow code (slow_flow) is based on EpicFlow v1.00. We extended it to reason over multiple frames and occlusions. The extended code is included and for details on the original code we refer to the paper by Revaud et al. [EpicFlow: Edge-Preserving Interpolation of
Correspondences for Optical Flow](https://hal.inria.fr/hal-01142656/document) and the project webpage https://thoth.inrialpes.fr/src/epicflow/.

We provide [two teaser sequences](http://www.cvlibs.net/projects/slow_flow/slow_flow_teaser.zip) to run our code. We are working on publishing the complete high speed datasets used in the project.

## Compiling
#### The following libraries are necessary to compile and use our code:

	Eigen3, Boost, Atlas, Blas, Lapack, Flann, GSL, PNG and JPEG
	sudo apt-get install libeigen3-dev libboost-all-dev libatlas-base-dev libblas-dev liblapack-dev libflann-dev libgsl-dev libpng-dev libjpeg-dev
	 
	Download and install opencv2.4
	http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html
	
	Download SED for edges
	https://www.microsoft.com/en-us/download/details.aspx?id=52370&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F389109f6-b4e8-404c-84bf-239f7cbf4e3d%2F

	Download Piotr Dollar's toolbox
	http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html

	Download Deep Matching 
	(Optional: using course-to-fine by setting ‘deep_matching’ to 0 and ‘slow_flow_layers’  larger than 1)
	http://lear.inrialpes.fr/people/revaud
	
	Download flow-code from Middlebury OF dataset and compile ImageLib
	http://vision.middlebury.edu/flow/code/flow-code.zip
	
	Download gco-v3.0 library
	http://vision.csd.uwo.ca/code/
	
	(Optional) Download Gunturk-Altunbasak-Mersereau Alternating Projections Image Demosaicking 
	By setting ‘raw_demosaicing’ to 1 in the configuration file and uncommenting line 17 and 38 in configuration.h
	http://www.ipol.im/pub/art/2011/g_gapd/

#### The paths to libraries need to be specified in the following files:
	configuration.h
	configuration_epic.h
	matlab/detect_edges.m
	CMakeLists.txt 

#### Run cmake and make to compile the code
	mkdir build
	cd build
	cmake ../
	make

## Run Pipeline ###
#### 1. Run epic flow on low resolution to use adaptive frame rates for slow flow
	./adaptiveFR -path [path] -folder [folder]

	Examples for our teaser_sequences:
	./adaptiveFR -path '[path to teaser]/sequence/' -folder 'sheeps' -raw
	./adaptiveFR -path '[path to teaser]/sequence/' -folder 'ambush_2' -format 'out_%i_%03i.png' -start 491 -sintel

#### 2. Run slow_flow for flow estimations of all consecutive high speed frames
	./slow_flow [slow flow cfg file]

	An example configuration file is provided in "cfgs".
	
	Optional: 
		-jet		compute the flow for one specified high speed pair
		-fr		using adaptive frame rate compute flow for 0: high frame rate, 1: low frame rate 
		-resume		resume processing configuration file
		-deep_settings	specify settings for deep matching

#### 3. Run dense_tracking using the output of slow flow 
	./dense_tracking [dense tracking cfg file]

	Optional: 
		-select		compute the flow for one specific final image pair
		-resume		resume processing configuration file

	An example configuration file is provided in "cfgs".

## Citation

If you use our code, please cite our paper:
<br><br>
@INPROCEEDINGS{<a href="http://www.cvlibs.net/publications/Janai2017CVPR.pdf">Janai2017CVPR</a>,<br>
&nbsp; author = {<a href="https://avg.is.tue.mpg.de/person/jjanai" target="blank">Joel Janai</a> and <a href="http://ps.is.tuebingen.mpg.de/person/g%C3%BCney" target="blank">Fatma Güney</a> and <a href="https://ps.is.tuebingen.mpg.de/person/jwulff" target="blank">Jonas Wulff</a> and <a href="http://ps.is.tuebingen.mpg.de/person/black" target="blank">Michael Black</a> and <a href="http://www.cvlibs.net" target="blank">Andreas Geiger</a>},<br>
&nbsp; title = {Slow Flow: Exploiting High-Speed Cameras for Accurate and Diverse Optical Flow Reference Data},<br>&nbsp; booktitle = {Conference on Computer Vision and Pattern	Recognition (CVPR)},<br>
&nbsp; year = {2017}<br>
}
