THEANO_FLAGS="floatX=float32,lib.cnmem=0.85,optimizer=fast_compile" platoon-launcher lm gpu0 gpu1 gpu2 gpu3 -w="--proto=prototype_med_lm --platoon"
