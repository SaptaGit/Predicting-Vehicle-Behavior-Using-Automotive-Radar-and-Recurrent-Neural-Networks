# Predicting-Vehicle-Behavior-Using-Automotive-Radar-and-Recurrent-Neural-Networks

We present a Long Short Term Memory (LSTM) encoder-decoder architecture to anticipate the future positions of vehicles in a road network given several seconds of historical observations and associated map features. Unlike existing architectures, the proposed method incorporates and updates the surrounding vehicle information in both the encoder and decoder, making use of dynamically predicted new data for accurate prediction in longer time horizons. It seamlessly performs four tasks: the first task encodes a feature given the past observations, the second task estimates future maneuvers given the encoded state, the third task predicts the future motion given the estimated maneuvers and the initially encoded states, and the fourth task estimates future trajectory given the encoded state and the predicted maneuvers and motions. Experiments on a public benchmark and a new, publicly available radar dataset demonstrate that our approach can equal or surpass the state-of-the-art for long term trajectory prediction.


**Steps to run the code:**

1. Run the EnocderDecoderV29 with the **processOrRead** flag  set to **processStr** to generate the data in the provided folder. Once done this will create all the relevant folders inside the parent folder needed for training the model. 

2. The again run the EnocderDecoderV29, but this time set the **processOrRead** flag  set to **readStr** and this will read the data from the previously created folder, perform the model the training, save the model weights, perfrom the trajectory prediction task and finally show the RMSE error on the terminal as well as write them in the provided file path. This code will execute the **GT surroudning technique** mentioned in this work. This means during the entire future prediction horizon it will will use the predicted positions for the target vehicle but the ground truth positions for the surrounding vehicles.  

3. Run the ModifiedDecoderV15 to do the **retrain technique** mentioned in this work. This is a more realistic setting. For the first round training provide the data folder created by the previous steps. This code will use that data to train the model and save the model weights in the provided locations. For the first time it will use the ground truth future values of the surrounding vehicles. Once the first round is done it will go though all the vehicle IDs to do the intermediate prediction stage using the currently trained model. Then it will use those predicted positions of both the target and surrounding vehicles to update the decoder input. Once done it will then convert it into a fixed shape array and use that instead of the gournd truth decoder input to fine tune the previously trained model. This additional retraining process was done to remove the discripency between the ground truth and actually predicted decoder inputs which will be used during a real testing. The retraining process will be done multiple times depending on a predecided value set through the **numberOfTrainingLoop** variable. This file executes the **retrain technique** proposed in this work, which can be think of as a variation of the classical schedule sampling technique where the decoder input is dynamic.

**If you are using this code please cite the following paper:**

S. Mukherjee, A. M. Wallace and S. Wang, "Predicting Vehicle Behavior Using Automotive Radar and Recurrent Neural Networks," in IEEE Open Journal of Intelligent Transportation Systems, vol. 2, pp. 254-268, 2021, doi: 10.1109/OJITS.2021.3105920.
