function detect_edges( img1, output_edges )
	% ############################# PLEASE SPECIFY ############################# 
	addpath('[SPECIFY PATH TO STRUCTURED EDGE DETECTION TOOLBOX]');
	addpath(genpath('[SPECIFY PATH TO PIOTR TOOLBOX]')); 
	load('[SPECIFY PATH TO SLOW FLOW RELEASE]/epic_flow_extended/modelFinal.mat'); 
	% ############################# PLEASE SPECIFY ############################# 
    
	% load first image, convert it to 8 bit unsigned and scale it
	I = imread(img1); 
    
	% detect edges
	edges = edgesDetect(I, model);
    
	% write edges to file
	fid=fopen(output_edges,'wb'); 
	fwrite(fid,transpose(edges),'single');
	 fclose(fid); 
end

