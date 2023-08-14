# SGC-counting-and-3D-mapping
This is the code supporting the paper published in JARO 'Neural degeneration in normal-aging human cochleas: machine-learning counts and 3D mapping in archival sections'
The code has four parts, Step1, Step2, and Step3, F1-score the function of each will be explained below. 

Step 1 
This step is to take the eslides (.svs files) from the MEE Temporal Bone library and transform the images into a 3d tiff file. The coordinates of the landmarks are exported and stored locally for use in step 2. 

Step2
This step is to crop the region of interest from the eslide, i.e. the cochlea, into 'samples' that have the size of 3600x2400 pixels. Then pass the samples sequentially to the two models (red model and blue model), and superimpose the masks from these two models together to generate the final mask where the SGCs are labeled with 1, the background labeled with 0. The x, y coordinates of the SGCs were then derived from this mask with connected component analysis. 

Step3 
This step is to align the serial sections of the human temporal bone by rotating and translating the Eslides. The parameter of this operation was then applied to the x,y coordinates of the SGCs (step 2) to generate the final x,y,z coordinates using 3D reconstruction. 

CalculateF1score.m
This file was coded in MATLAB. The F1 - Score for the model and human-level-performance F1 score is coded separately as Neurolucida data files were involved in the human-level-performance F1 score. 

SGCsMLFigures.m
This file was coded in MATLAB. This includes the analysis for all the data presented in this paper. 





