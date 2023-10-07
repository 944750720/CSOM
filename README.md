# CSOM
Based on https://github.com/JustGlowing/minisom.
In order to classify goals by unsupervised-learning, I am using SOM (Self organizing maps neural network) to process MIMO-FMCW radar. 
Here is the setting of experiment and original images.

![exp setting](https://imgur.com/9iDxGbb.png "experiment setting")
![full amp image](https://imgur.com/pdghVF3.jpg "full ampitude image")
![full pha image](https://imgur.com/uq0WNyE.jpg "full phase image")
We can see the reflector is the field be circled on the left side of radar heatmap. 
![classified result](https://imgur.com/NTUQQdF.jpg "SOM result")
And I am coding the complex-value SOM to make better use of the amplitude and phase data of radar.
