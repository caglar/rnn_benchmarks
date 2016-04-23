
def prototype_lm():
    # Model related
    config = {}
    config['dim'] = 200
    config['dim_word'] = 200
    config['encoder'] = 'lstm'
    config['maxlen'] = 20
    config['batch_size'] = 20
    config['max_epochs'] = 6
    config['use_dropout'] = True
    config['lrate'] = 0.1
    config['data_path'] = '/home/benchmark/code/rnn_exps/torch/ptb/lstm/data' 
    config['n_words'] = 10000
    return config

def prototype_med_lm():
    config = prototype_lm()
    config['dim'] = 650
    config['maxlen'] = 40
    return config

def prototype_large_lm():
    config = prototype_lm()
    config['dim'] = 650
    config['maxlen'] = 50
    config['nlayers'] = 2
    return config

