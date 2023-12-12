Contrastive Language-Image Pretraining (CLIP) Demo

clip_eval.ipynb follows https://github.com/openai/CLIP demo but applied to the Intel classification dataset. This demo includes two parts, Zero-Shot Prediction (ZSP), and Linear Probe Evaluation (LPE).

The assignment says, to perform ZSP and list the top 10 classes that an image is most likely to be, but unfortunately, the Intel dataset only has six classes. Therefore only the top six were listed in descending order. The results can be found in the notebook under the section entitled Zero Shot Prediction, and are as follows:

Top predictions:
          street: 72.27%
       buildings: 27.44%
        mountain: 0.25%
             sea: 0.02%
          forest: 0.01%
         glacier: 0.00%

Note: I tried at least 20 different images, and and the network was right every time.

The LPE exercise is located in the notebook section called Linear Probe Evaluation. Results from training and inferencing came out to 95.2% accuracy.