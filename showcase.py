import rnn
import theano
import numpy as np
import pickle

from keras.datasets import reuters

from model import Model
from theano import tensor as T
from opt_algs import OptAlgs

class LSTMEncoderDecoder(Model):

        def __init__(self, nws, nwt, nes, net, nh, rnn_factory, **params):

                super(LSTMEncoderDecoder, self).__init__(**params)

                #word embeddings used for input & output
                self.add_parameters((nws, nes), "Es")
                self.add_parameters((nwt, net), "Et")

                #encdec
                self.encoder = rnn_factory(nes, nh, self, prefix="encoder")
                self.decoder = rnn_factory(nes, nh, self, include_h0 = False, prefix="decoder")

                # output params
                self.add_parameters((nwt, nh), "Woh")
                self.add_parameters(nwt, "bo")

                #inputs, shapes are (w,)
                self.src = T.ivector("source")
                self.tgt = T.ivector("target")

                self.inputs = [self.src, self.tgt]

                #phrase 1 encode
                xs = self.Es[self.src]

                encoded, _ = theano.scan(fn=self.encoder.recurrence, sequences=xs[:-1],  #skip the EOS
                outputs_info=self.encoder.initial_hidden(), n_steps=xs.shape[0]-1)

                #phrase 2, delivery outputed hidden of encoder to decoder, LSTM reture format [ht,ct]

                encoding = [encoded[-1]]

                self.encoding = theano.function(inputs=[self.src], outputs=encoding)

                #phrase 3, decode
                def recurrence(x_t, *h_tm1):
                    hiddens_t = self.decoder.recurrence(x_t, *h_tm1)
                    y_t = T.nnet.softmax(T.dot(self.Woh, hiddens_t[0]) + self.bo)  # shape of y is (w,)

                    return hiddens_t+ [y_t]
 
                xt = self.Et[self.src]   #invert the target input in decoder

                outputs, _= theano.scan(fn=recurrence, sequences=xt[::-1],
                        outputs_info=encoding+[None], n_steps=xt.shape[0])             # output [[h,c,y] * n]

                prob = outputs[-1][:,0,:]

                #objective function
                self.objective = -T.mean(T.log(prob)[T.arange(self.tgt.shape[0]), self.tgt])


        def decode(self, seq) :
                h_prev = []
                for i in range(len(self.decoder.initial_hidden())):
                    h_prev.append(T.vector('hidden_%d' % i))

                x_prev = T.iscalar('word')

                h_t = self.decoder.recurrence(self.Et[x_prev], *h_prev)
                y_t = T.nnet.softmax(T.dot(self.Woh, h_t[0]) + self.bo)

                decode_step = theano.function(inputs=[x_prev] + h_prev, outputs=[y_t] + h_t)

                hidden = self.encoding(seq)

                tokens = [4001]

                while len(tokens) < 13:
                    output_y = decode_step(tokens[-1], *hidden)
                    word = np.argmax(output_y[0])

                    tokens.append(word)

                    if(word == 4001):
                        return tokens[1:]

                    hidden = output_y[1:]

                return tokens[1:]

if __name__ == "__main__":

    theano.config.exception_verbosity = 'high'
    theano.config.optimizer = 'None'
    theano.config.optimizer = 'fast_run' # also 'fast_run' or 'None' for debugging
    theano.config.linker = 'py'
    theano.config.floatX = 'float32'

    print 'initialising...'

    V = 1001
    E = 12
    total_trainset = 10000
    total_iterations = 9000
    train_x_entropy = 0

    (X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.pkl", \
                nb_words=None, skip_top=0, maxlen=None, test_split=0.1, seed=10086)

    word_map_tmp = reuters.get_word_index(path="reuters_word_index.pkl")
    word_dict = dict((v, k) for k, v in word_map_tmp.iteritems())
    word_dict[0] = "<UNK>"

    def real_words(l, eos):
        sent = []
        for word in l:
            if word == eos :
                sent.append("<EOS>")
            elif word > eos:
                sent.append(word_dict[0])
            else:
                sent.append(word_dict[word])

        return sent

    model = LSTMEncoderDecoder(nws=V+1, nwt=V+1, nes=E, net=E, nh=96, rnn_factory=rnn.RNN)
    train = model.trainer(opt_alg='ADAgrad', lr=0.01)
    valid = model.validater()

    print 'training...'

    for i in range(total_iterations):
        content = []

        for word in X_train[i][:10]:
            if word > 4000 :
                content.append(0)
            else :
                content.append(word)

        seq = np.asarray(content+[4001]).astype("int32")
        target_seq = np.asarray(content[::-1]+[4001]).astype("int32")

        x_entropy = train(seq, target_seq)
        train_x_entropy += x_entropy

            print "trainging tuple ", i

        if (i+1) % 5 == 0:
            print 'training batch cross-entropy', (train_x_entropy / 5)
            train_x_entropy = 0

        if (i+1) % 10 == 0:
            for j in range(10):
                pickone = np.random.randint(0, len(X_test))

                testcase = []
                for word in X_test[pickone][:10]:
                    if word > 4000  :
                        testcase.append(0)
                    else :
                        testcase.append(word)

                test_seq = np.asarray(testcase + [4001]).astype("int32")
                decoded = model.decode(test_seq)

                print 'decoding', test_seq, '->', decoded

                if (i+1) % 30 == 0:
                    print  real_words(test_seq), "======>",  real_words(decoded)



