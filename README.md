# A Python Version of Psychoacoustic Model-I (PAM-I)

### [1] The psychoacoustic model is based on  
https://www.iso.org/standard/22412.html
### [2] The implementation is derived from 
https://www.petitcolas.net/fabien/software/mpeg/
### [3] It supports both 44100 Hz and 32000 Hz. The corresponding MPEG tables are from 
https://books.google.com/books?id=tb41_XyeF4MC&pg=PA607&lpg=PA607&dq=19982.81+frequency&source=bl&ots=W-PAtFj9K_&sig=ACfU3U3AUtD71CM_jo7UrGGPpMLeTknYqw&hl=en&sa=X&ved=2ahUKEwinkZ6-gafmAhURPa0KHZWrBOwQ6AEwAHoECAoQAQ#v=onepage&q&f=false

## How to run:
### python2 calculate_global_masking_threshold.py --sample_rate 44100 --input_data './data/test_data_44.npy'

![alt text](https://github.com/cocosci/pam-nac/blob/master/pam/output/pam-1-test.png)

## To train a model:

### python main.py --learning_rate 0.0002 --epoch 150  --loss_coeff '50 5 5 5 10 0.0'  --training_mode 3

You will see something like the following. STOI (and PESQ) scores do not matter for audio signals.
```
Epoch   0: SNR: 19.09991 dB    STOI: 0.84400    _quan_loss: 0.03533  tau: -0.01500   fully_entropy: 2.34234
Epoch   1: SNR: 22.10162 dB    STOI: 0.90347    _quan_loss: 0.03252  tau: -0.03000   fully_entropy: 2.75364
Epoch   2: SNR: 22.73904 dB    STOI: 0.90252    _quan_loss: 0.02893  tau: -0.04500   fully_entropy: 2.75894
Epoch   3: SNR: 22.54639 dB    STOI: 0.91948    _quan_loss: 0.02741  tau: -0.06000   fully_entropy: 2.85497
Epoch   4: SNR: 22.60652 dB    STOI: 0.91150    _quan_loss: 0.02608  tau: -0.07500   fully_entropy: 2.82074
Epoch   5: SNR: 22.35036 dB    STOI: 0.92016    _quan_loss: 0.02660  tau: -0.09000   fully_entropy: 2.89799
Epoch   6: SNR: 22.69647 dB    STOI: 0.91279    _quan_loss: 0.02655  tau: -0.10500   fully_entropy: 2.85986
Epoch   7: SNR: 22.90094 dB    STOI: 0.92637    _quan_loss: 0.02734  tau: -0.12000   fully_entropy: 2.96242
Epoch   8: SNR: 22.54134 dB    STOI: 0.92840    _quan_loss: 0.02699  tau: -0.13500   fully_entropy: 2.94525
Epoch   9: SNR: 22.19404 dB    STOI: 0.93287    _quan_loss: 0.02898  tau: -0.15000   fully_entropy: 3.10054
Epoch  10: SNR: 22.69657 dB    STOI: 0.93189    _quan_loss: 0.02893  tau: -0.16500   fully_entropy: 3.09248
Epoch  11: SNR: 22.82722 dB    STOI: 0.94572    _quan_loss: 0.02979  tau: -0.18000   fully_entropy: 3.18963
Epoch  12: SNR: 22.41368 dB    STOI: 0.94167    _quan_loss: 0.02932  tau: -0.19500   fully_entropy: 3.10602
```