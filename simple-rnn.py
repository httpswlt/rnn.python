# coding: utf-8
import numpy as np


def data_process():
    # data I/O
    data = open('input.txt', 'r').read()  # should be simple plain text file

    # use set() to count the vacab size
    chars = list(set(data))
    chars.sort()
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))

    return data, chars


class RNN:
    def __init__(self):
        # data parameter
        self.data, self.chars = data_process()
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab = len(self.chars)

        # network hyper_parameters
        self.h_dim = 100  # size of hidden layer of neurons
        self.T = 25  # number of steps to unroll the RNN for
        self.lr = 0.1
        self.u = np.random.randn(self.h_dim, self.vocab) * 0.01
        self.w = np.random.randn(self.h_dim, self.h_dim) * 0.01
        self.v = np.random.randn(self.vocab, self.h_dim) * 0.01
        self.w_b = np.zeros((self.h_dim, 1))  # hidden bias
        self.v_b = np.zeros((self.vocab, 1))  # output bias

        # auto-adjust the lr by Adagrad
        self.m_u, self.m_w, self.m_v = np.zeros_like(self.u), np.zeros_like(self.w), np.zeros_like(self.v)
        self.m_w_b, self.m_v_b = np.zeros_like(self.w_b), np.zeros_like(self.v_b)  # memory variables for Adagrad

    def step(self, x, pre_h):
        """

        :param x:
        :param pre_h:
        :return:
        """
        h = np.tanh(np.dot(self.u, x) + np.dot(self.w, pre_h) + self.w_b)
        y = np.dot(self.v, h) + self.v_b
        return y, h

    def forward(self, x, pre_h):
        xs, hs, ys = {}, {}, {}
        # record each hidden state of
        hs[-1] = np.copy(pre_h)
        # forward pass for each training data point
        for t in range(self.T):
            # input
            xs[t] = np.zeros((self.vocab, 1))  # encode in 1-of-k representation
            xs[t][x[t]] = 1

            # run step
            y, h = self.step(xs[t], hs[t - 1])

            # update state
            ys[t] = np.exp(y) / np.sum(np.exp(y))
            hs[t] = h
        return xs, ys, hs

    def compute_loss(self, x, y):
        loss = 0
        for t, output in x.items():
            # softmax (cross-entropy loss)
            loss += -np.log(output[y[t], 0])
        return loss

    def bptt(self, xs, ys, hs, targets):
        # compute loss
        loss = self.compute_loss(ys, targets)

        # backward pass: compute gradients going backwards
        d_u, d_w, d_v = np.zeros_like(self.u), np.zeros_like(self.w), np.zeros_like(self.v)
        d_w_b, d_v_b = np.zeros_like(self.w_b), np.zeros_like(self.v_b)
        d_h_next = np.zeros_like(hs[0])
        for t in reversed(range(self.T)):
            # compute derivative of error w.r.t the output probabilites
            # dE/dy[j] = y[j] - t[j]
            dy = np.copy(ys[t])
            dy[targets[t]] -= 1  # backprop into y

            # output layer doesnot use activation function,
            # the weight between hidden layer and output layer.
            # dE/dy[j]*dy[j]/dWhy[j,k] = dE/dy[j] * h[k]
            d_v += np.dot(dy, hs[t].T)
            d_v_b += dy

            # backprop into h
            d_h = np.dot(self.v.T, dy) + d_h_next
            # backprop through tanh nonlinearity
            # dtanh(x)/dx = 1 - tanh(x) * tanh(x)
            d_h_raw = (1 - hs[t] * hs[t]) * d_h
            d_w_b += d_h_raw

            # derivative of the error with regard to the weight between input layer and hidden layer
            d_u += np.dot(d_h_raw, xs[t].T)
            d_w += np.dot(d_h_raw, hs[t - 1].T)
            # derivative of the error with regard to H(t+1)
            # or derivative of the error of H(t-1) with regard to H(t)
            d_h_next = np.dot(self.w.T, d_h_raw)

        for d_param in [d_u, d_w, d_v, d_w_b, d_v_b]:
            np.clip(d_param, -5, 5, out=d_param)  # clip to mitigate exploding gradients

        for param, d_param, mem in zip([self.u, self.w, self.v, self.w_b, self.v_b],
                                       [d_u, d_w, d_v, d_w_b, d_v_b],
                                       [self.m_u, self.m_w, self.m_v, self.m_w_b, self.m_v_b]):
            mem += d_param * d_param
            # learning_rate is adjusted by mem, if mem is getting bigger, then learning_rate will be small
            # gradient descent of Adagrad
            param += -self.lr * d_param / np.sqrt(mem + 1e-8)  # adagrad update
        return loss

    def sample(self, h, seed_ix, n):
        """
            # given a hidden RNN state, and a input char id, predict the coming n chars

        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """

        # a one-hot vector
        x = np.zeros((self.vocab, 1))
        x[seed_ix] = 1

        ixes = []
        for t in range(n):
            # self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
            h = np.tanh(np.dot(self.u, x) + np.dot(self.w, h) + self.w_b)
            # y = np.dot(self.W_hy, self.h)
            y = np.dot(self.v, h) + self.v_b
            # softmax
            p = np.exp(y) / np.sum(np.exp(y))
            # sample according to probability distribution
            ix = np.random.choice(range(self.vocab), p=p.ravel())

            # update input x
            # use the new sampled result as last input, then predict next char again.
            x = np.zeros((self.vocab, 1))
            x[ix] = 1

            ixes.append(ix)

        return ixes

    def train(self):
        # iterator counter
        n = 0
        # data pointer
        p = 0

        # main loop
        while True:
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + self.T + 1 >= len(self.data) or n == 0:
                # reset RNN memory
                # h_prev is the hidden state of RNN
                h_prev = np.zeros((self.h_dim, 1))
                # go from start of data
                p = 0

            inputs = [self.char_to_ix[ch] for ch in self.data[p: p + self.T]]
            targets = [self.char_to_ix[ch] for ch in self.data[p + 1: p + self.T + 1]]

            xs, ys, hs = self.forward(inputs, h_prev)
            loss = self.bptt(xs, ys, hs, targets)

            if n % 100 == 0:
                print('iter %d, loss: %f' % (n, loss))  # print progress
                sample_ix = self.sample(h_prev, inputs[0], 100)
                txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                print('---- sample -----')
                print('----\n input: {} \n output: {}{} \n----'.format(self.ix_to_char[inputs[0]],
                                                                       self.ix_to_char[inputs[0]], txt))

            p += self.T  # move data pointer
            n += 1  # iteration counter


def main():
    rnn = RNN()
    rnn.train()


if __name__ == '__main__':
    main()
