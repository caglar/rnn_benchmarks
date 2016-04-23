THEANO_FLAGS="floatX=float32,device=gpu0,lib.cnmem=0.85,force_device=True,optimizer=fast_compile" python train_lm.py --proto prototype_lm
