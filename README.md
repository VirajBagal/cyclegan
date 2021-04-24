# Cyclegan

![alt text](https://github.com/VirajBagal/cyclegan/tree/main/Images/cyclegan.png?raw=true)

In this repository, I train a CycleGAN to convert photos to Monet-like paintings and Monet paintings to photos. 

# Generated Output


![alt text](https://github.com/VirajBagal/cyclegan/tree/main/Images/sample1.png?raw=true)
![alt text](https://github.com/VirajBagal/cyclegan/tree/main/Images/sample2.png?raw=true)
![alt text](https://github.com/VirajBagal/cyclegan/tree/main/Images/sample3.png?raw=true)

# Train on Custom data

To train on custom data, setup the images from two sources in two different directories. Clone this repository. Then run the following command.

```
python main.py --photos_path one_dir --paintings_path second_dir --run_name give_some_name
```

The evaluation curves and sample output images will be generated and automatically uploaded in real time to your WandB dashboard. Please create a WandB account to log the results successfully.
