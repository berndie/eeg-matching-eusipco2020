MODELING THE RELATIONSHIP BETWEEN ACOUSTIC STIMULUS AND EEG WITH A DILATED CONVOLUTIONAL NEURAL NETWORK (EUSIPCO2020)
===
### Authors
Bernd Accou, Mohammad Jalilpour Monesi, Jair Montoya Martinez, Hugo Van hamme, Tom Francart


### Models

Code for the baseline decoder model, baseline convolutional model and the dilated convolutional network can be found in `baseline_decoder_model.py`, `baseline_convolutional_model.py` and `dilated_model.py` respectively.

Models are constructed using the Keras API of tensorflow and can be trained by calling the `fit` method:

#### Linear decoder baseline model

![Linear baseline model](images/baseline_decoder_model.svg)

Training the linear baseline is equivalent to training the linear decoder
```
from baseline_model import linear_decoder, linear_decoder_baseline

decoder = linear_decoder()
decoder.fit(eeg_train, good_env_train)

baseline = linear_decoder_baseline(decoder)
results = baseline.evaluate([eeg_test, env1_test, env2_test])
```


#### Convolutional baseline model

![Convolutional baseline model](images/baseline_convolutional_model.svg)

Training/Evaluating the simple convolutional model is as described in the Keras documentation
```
from baseline_convolutional_model import simple_convolutional_model

conv_model = simple_convolutional_model()
conv_model.fit([eeg_train, env1_train, env2_train], labels)

results = conv_model.evaluate([eeg_test, env1_test, env2_test], labels)
```

#### Dilated model

![Dilated model](images/dilation_network.svg)

Training/Evaluating the dilated model is as described in the Keras documentation
```
from dilated_model import dilated_model

dilated = dilated_model()
dilated.fit([eeg_train, env1_train, env2_train], labels)

results = dilated.evaluate([eeg_test, env1_test, env2_test], labels)
```
