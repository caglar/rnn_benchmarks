'''
Build a simple neural language model using GRU units
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import ipdb
import numpy

import warnings
import sys
import time

from collections import OrderedDict

import reader

import pprint


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng=None):
    if trng is None:
        trng = RandomStreams(1234)

    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype) * 2,
        state_before)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        if vv.shape != pp[kk].shape:
            raise ValueError(
                'Shape mismatch: {} - {} for {}'.format(
                    pp[kk].shape, vv.shape, kk))
        print '...loading parameter {:15}: {}'.format(vv.shape, kk)
        params[kk] = pp[kk]

    return params


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'gru': ('param_init_gru', 'gru_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# orthogonal initialization for weights
# see Saxe et al. ICLR'14
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return 0.05 * (u.dot(v)).astype('float32')


# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')
    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# LSTM layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None):
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U
    params[_p(prefix, 'b')] = numpy.zeros((4 * dim,)).astype('float32')

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm',
               one_step=False,
               init_state=None, init_memory=None,
               trng=None, use_noise=None,
               **kwargs):

    nsteps = state_below.shape[0]
    if trng is None:
        trng = RandomStreams(1234)

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'U')].shape[0]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(x_, h_, c_, U, b, p=None,
              use_noise=None, Wemb=None):

        preact = tensor.dot(h_, U)
        preact += x_
        preact += b
        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        h = o * tensor.tanh(c)

        return h, c

    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]

    updates = OrderedDict({})
    non_seqs = [tparams[_p(prefix, 'U')], tparams[_p(prefix, 'b')]]
    if one_step:
        updates = OrderedDict({})
        h, c = _step(state_below, init_state, init_memory, *non_seqs)
        rval = [h, c]
    else:
        rval, updates = theano.scan(_step,
                                    sequences=[state_below],
                                    outputs_info=[init_state,
                                                  init_memory],
                                    non_sequences=non_seqs,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    strict=True)

    return rval, updates


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None,
                   **kwargs):

    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru',
              mask=None, one_step=False, init_state=None, **kwargs):
    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = state_below.shape[0]

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    # set initial state to all zeros
    if init_state is None:
        init_state = tensor.unbroadcast(tensor.alloc(0., n_samples, dim), 0)

    if one_step:  # sampling
        rval = _step(*(seqs + [init_state] + shared_vars))
    else:  # training
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state],
                                    non_sequences=shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    strict=True)
    rval = [rval]
    return rval


# initialize all parameters
def init_params(options):
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    for i in xrange(options['nlayers']):
        if i == 0:
            nin = options['dim_word']
        else:
            nin = options['dim']
        params = get_layer(options['encoder'])[0](
            options, params, prefix="encoder_l" + str(i),
            nin=nin,
            dim=options['dim'])

    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_logit',
                                nin=options['dim'],
                                nout=options['n_words'])

    return params


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # input
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    if options['use_dropout']:
        emb = dropout_layer(emb, use_noise=use_noise)

    opt_ret['emb'] = emb

    # pass through LSTM layer, recurrence here
    updates = {}
    ht = emb
    for l in xrange(options['nlayers']):
        proj, ups = get_layer(options['encoder'])[1](tparams, ht, options,
                                                     prefix='encoder_l' + str(l),
                                                     trng=trng,
                                                     use_noise=use_noise)
        ht = proj[0]
        updates.update(ups)

        if options['use_dropout']:
            ht = dropout_layer(ht, use_noise=use_noise)

    logit = get_layer('ff')[1](tparams,
                               ht,
                               options,
                               prefix='ff_logit',
                               activ='linear')

    logit_shp = logit.shape
    probs = tensor.nnet.softmax(
        logit.reshape([logit_shp[0] * logit_shp[1],
                       logit_shp[2]]))

    # cost
    x_flat = x.flatten()
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost = -tensor.log(tensor.maximum(probs.flatten()[x_flat_idx], 1e-8))
    cost = cost.reshape([x.shape[0], x.shape[1]])
    opt_ret['cost_per_sample'] = cost
    cost = (cost).sum(0)
    return trng, use_noise, x, opt_ret, cost, updates


# calculate the log probablities on a given corpus using language model
def pred_probs(f_log_probs, options, iterator, verbose=True):
    probs = []
    n_done = 0
    for x, y in iterator:
        n_done += len(x)

        pprobs = f_log_probs(x)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


def perplexity(f_cost, lines, worddict, options, verbose=False):
    cost = 0.
    n_words = 0.

    for i, line in enumerate(lines):
        # get array from line
        seq = []
        for w in line.strip().split():
            if w in worddict:
                seq.append(worddict[w])
            else:
                seq.append(1)  # unknown
        seq = [s if s < options['n_words'] else 1 for s in seq]
        n_words += len(seq) + 1
        x = numpy.array(seq + [0]).astype('int64').reshape([len(seq) + 1, 1])
        x_mask = numpy.ones((len(seq) + 1, 1)).astype('float32')

        # note, f_cost returns negative log-prob summed over the sequence
        cost += f_cost(x, x_mask)

    cost = cost / n_words
    return cost


def sgd(lr, tparams, grads, inps, cost, max_grad_norm):
    if max_grad_norm > 0:
        gnorm = tensor.sqrt(sum((g**2).sum() for g in grads))
        alpha = tensor.switch(gnorm > max_grad_norm, max_grad_norm / gnorm, 1.)
    else:
        alpha = 1.
    # allocate gradients and set them all to zero
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]

    # create gradient copying list,
    # from grads (tensor variable) to gshared (shared variable)
    gsup = [(gs, alpha * g) for gs, g in zip(gshared, grads)]

    # compile theano function to compute cost and copy gradients
    f_grad_shared = theano.function(inps, cost, updates=gsup)

    # define the update step rule
    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]

    # compile a function for update
    f_update = theano.function([lr], [], updates=pup)

    return f_grad_shared, f_update


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of GRU units
          encoder='gru',
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 weight decay penalty
          lrate=0.01,
          n_words=100000,  # vocabulary size
          maxlen=100,  # maximum length of the description
          batch_size=16,
          valid_batch_size=16,
          max_grad_norm=5,
          nlayers=1,
          data_path=None,
          use_dropout=False,
          platoon=False,
	  name=""):

    # Model options
    model_options = locals().copy()

    print 'Loading data'

    raw_data = reader.ptb_raw_data(data_path)
    train_data, valid_data, test_data, _ = raw_data
    pprint.pprint(model_options)
    print 'Building model'
    params = init_params(model_options)

    # create shared variables for parameters
    tparams = init_tparams(params)

    if platoon:
        print "PLATOON: Init ...",
        from platoon.channel import Worker
        from platoon.param_sync import ASGD
        worker = Worker(control_port=5567)
        print "DONE"

        print "PLATOON: Initializing shared params ...",
        worker.init_shared_params(tparams.values(), param_sync_rule=ASGD())
        print "DONE"
	worker.send_req({"type": name})

    # build the symbolic computational graph
    trng, use_noise, \
        x, \
        opt_ret, \
        cost, ups = \
        build_model(tparams, model_options)
    inps = [x]

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, updates=ups)
    print 'Done'

    # before any regularizer - will be used to compute ppl
    print 'Building f_cost...',
    cost_sum = cost.sum()
    f_cost = theano.function(inps, cost_sum, updates=ups)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = sgd(lr, tparams, grads, inps, cost, max_grad_norm)
    print 'Done'

    print 'Optimization'

    history_errs = []
    history_ppls = []
    wpss = []

    best_p = None

    # Training loop
    uidx = 0
    estop = False
    bad_counter = 0
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0
            tlen = 0
            start_time = time.time()
            for x, y in reader.ptb_iterator(train_data, batch_size, maxlen):
                if platoon:
                    #print "PLATOON: Copying data from master ...",
                    worker.copy_to_local()
                    #print "DONE"

                n_samples += len(x)
                uidx += 1
                use_noise.set_value(1.)
                tlen += (x.shape[0] * x.shape[1])
                # pad batch and create mask
                if x is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    uidx -= 1
                    continue

                ud_start = time.time()

                # compute cost, grads and copy grads to shared variables
                cost = f_grad_shared(x)

                # do the update on parameters
                f_update(lrate)

                ud = time.time() - ud_start

                if platoon:
                    #print "PLATOON: Syncing with master ...",
                    worker.sync_params(synchronous=True)
                    #print "DONE"

                # check for bad numbers
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1.

                # verbose
                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

                # finish after this many updates
                if uidx >= finish_after:
                    print 'Finishing after %d iterations!' % uidx
                    estop = True
                    break
            current_time = time.time()
            wps = int(tlen // (current_time - start_time))
            print "Current wps", wps
            wpss.append(wps)
            print 'Seen %d samples' % n_samples
            if platoon:
                print "PLATOON: Sending wps to controller ...",
                worker.send_req({'wps': wps, 'epoch': eidx})
                print "DONE"

        print "Avg wps, ", numpy.mean(wpss)
        print "Std avgs,", numpy.std(wpss)

        use_noise.set_value(0.)
    finally:
        if platoon:
            print "PLATOON: Closing worker ...",
            worker.send_req('done')
            worker.close()
            print "DONE"
    return 0


if __name__ == '__main__':
    pass
