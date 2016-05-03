THEANO_FLAGS="floatX=float32,lib.cnmem=0.85,optimizer=fast_compile" platoon-launcher lm gpu0 gpu1 -w="--proto=prototype_lm --platoon"
