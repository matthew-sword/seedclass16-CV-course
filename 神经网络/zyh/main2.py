#! -*-coding:utf8-*-
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

reg_lambda = 0.001
epsilon = 0.000001
num_examples = 0
num_passes = 7000
gamma = 0.2


def fetch_data():
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

    # pprint(newsgroups_train.data[0])

    num_train = len(newsgroups_train.data)
    num_test = len(newsgroups_test.data)
    print(num_train, num_test)

    vectorizer = TfidfVectorizer(max_features=20)

    x = vectorizer.fit_transform(newsgroups_train.data + newsgroups_test.data)
    x_train = x[0:num_train, :]
    x_test = x[num_train:num_train + num_test, :]

    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test


def update_learn_rate(rate, method="exp", iters=0):
    if method == "step":
        rate *= np.power(gamma, (iters/num_passes))
    elif method == "exp":
        rate *= np.power(gamma, iters)
    elif method == "inv":
        pass
        # rate *= np.power((1 + gamma * iters), (-power))
    else:
        pass


class init_m(object):
    def __init__(self, m_x):
        self.a = m_x


class end_m(object):
    def __init__(self, e_x):
        self.a = e_x


class softmax(object):
    def __init__(self, in_dim, out_dim):
        self.w = np.random.randn(in_dim, out_dim) / np.sqrt(in_dim)
        self.b = np.zeros((1, out_dim))
        self.dw = None
        self.db = None
        self.delta = None
        self.z = None
        self.a = None

    def forward(self, f_x):
        self.z = f_x.dot(self.w) + self.b
        exp_scores = np.exp(self.z)
        self.a = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.a

    def backprop(self, font_layer, back_layer):
        self.delta = self.a
        self.delta[range(num_examples), back_layer.a] -= 1
        self.dw = font_layer.a.T.dot(self.delta)
        self.db = np.sum(self.delta, axis=0, keepdims=True)

        self.dw += reg_lambda * self.w
        self.w += -epsilon * self.dw
        self.b += -epsilon * self.db


class relu(object):
    def __init__(self, in_dim, out_dim):
        self.w = np.random.randn(in_dim, out_dim) / np.sqrt(in_dim)
        self.b = np.zeros((1, out_dim))
        self.dw = None
        self.db = None
        self.delta = None
        self.z = None
        self.a = None
        self.grad = None

    def forward(self, f_x):
        self.z = f_x.dot(self.w) + self.b
        self.a = np.where(self.z < 0, 0, self.z)
        self.grad = np.where(self.a < 0, 0, 1)
        return self.a

    def backprop(self, font_layer, back_layer):
        self.delta = back_layer.delta.dot(back_layer.w.T) * self.grad
        self.dw = font_layer.a.T.dot(self.delta)
        self.db = np.sum(self.delta, axis=0, keepdims=True)

        self.dw += reg_lambda * self.w
        self.w += -epsilon * self.dw
        self.b += -epsilon * self.db


class l_tanh(object):
    def __init__(self, in_dim, out_dim):
        self.w = np.random.randn(in_dim, out_dim) / np.sqrt(in_dim)
        self.b = np.zeros((1, out_dim))
        self.dw = None
        self.db = None
        self.delta = None
        self.z = None
        self.a = None
        self.grad = None

    def forward(self, f_x):
        self.z = f_x.dot(self.w) + self.b
        self.a = np.tanh(self.z)
        self.grad = 1 - np.power(self.a, 2)
        return self.a

    def backprop(self, font_layer, back_layer):
        self.delta = back_layer.delta.dot(back_layer.w.T) * self.grad
        self.dw = font_layer.a.T.dot(self.delta)
        self.db = np.sum(self.delta, axis=0, keepdims=True)

        self.dw += reg_lambda * self.w
        self.w += -epsilon * self.dw
        self.b += -epsilon * self.db


class MLP(object):

    def __init__(self):
        self.layers = [relu(20, 4),
                       relu(4, 4),
                       relu(4, 4),
                       relu(4, 4),
                       softmax(4, 20)]
        self._loss = 10000

    def forward(self, f_x):
        tmp = f_x
        for layer in self.layers:
            tmp = layer.forward(tmp)
        return tmp

    def loss(self, l_x, y):
        l_x = self.forward(l_x)

        # Calculating the loss
        corect_logprobs = -np.log(l_x[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        w_sum = 0
        for layer in self.layers:
            w_sum += np.sum(np.square(layer.w))

        data_loss += reg_lambda / 2 * w_sum
        self._loss = 1. / num_examples * data_loss

    def train(self, t_x, y):

        for passes in range(0, num_passes):
            self.loss(t_x, y)
            if passes % 1000 == 0:
                print("Loss after iteration %i: %f" % (passes, self._loss))

            for i in range(len(self.layers)):
                if i == 0:
                    self.layers[-i - 1].backprop(self.layers[-i - 2], end_m(y))
                elif i == len(self.layers) - 1:
                    self.layers[-i - 1].backprop(init_m(t_x), self.layers[-i])
                else:
                    self.layers[-i - 1].backprop(self.layers[-i - 2], self.layers[-i])
            update_learn_rate(epsilon, "fixed", passes)

    def predict(self, _x, _y):
        n_correct = 0
        n_test = _x.shape[0]
        for n in range(n_test):
            xp = _x[n, :]
            yp = np.argmax(self.forward(xp), axis=1)
            if yp == _y[n]:
                n_correct += 1.0

        print('Accuracy %f = %d / %d' % (n_correct / n_test, int(n_correct), n_test))


def save_result():
    pass

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = fetch_data()
    num_examples, input_dim = X_train.shape
    np.random.seed(0)

    print(num_examples)
    # laye = [relu(20, 4), relu(4, 4), relu(4, 4), relu(4, 4), softmax(4, 20)]
    Model = MLP()
    # Model.layers = laye
    Model.train(X_train, Y_train)
    Model.predict(X_train, Y_train)
    Model.predict(X_test, Y_test)
