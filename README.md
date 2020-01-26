# A Python Version of Psychoacoustic Model-I (PAM-I)

## Claims:
### [1] The psychoacoustic model is based on  
https://www.iso.org/standard/22412.html
### [2] The implementation is derived from 
https://www.petitcolas.net/fabien/software/mpeg/
### [3] It supports both 44100 Hz and 32000 Hz. The corresponding MPEG tables are from 
https://books.google.com/books?id=tb41_XyeF4MC&pg=PA607&lpg=PA607&dq=19982.81+frequency&source=bl&ots=W-PAtFj9K_&sig=ACfU3U3AUtD71CM_jo7UrGGPpMLeTknYqw&hl=en&sa=X&ved=2ahUKEwinkZ6-gafmAhURPa0KHZWrBOwQ6AEwAHoECAoQAQ#v=onepage&q&f=false

## How to run:
### python2 calculate_global_masking_threshold.py --sample_rate 44100 --input_data './data/test_data_44.npy'

![alt text](https://github.com/cocosci/pam-i-python/blob/master/pam/output/pam-1-test.png)