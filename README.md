# FRF ARGUS c1 Segementation App

This is a deployed app to segement ARGUS imagery from FRF camera 1 (facing north). 

There are 4 classes: 
1. Water + buildings + background
2. Vegetation
3. Sand
4. Coarse sediment/Lag deposits

https://huggingface.co/spaces/ebgoldstein/FRFArgus

this repo is missing the actual model (which is up on huggingface).

The model was built with [Segmentation Zoo](https://github.com/dbuscombe-usgs/segmentation_zoo), with images labeled with [Doodler](https://github.com/dbuscombe-usgs/dash_doodler)
