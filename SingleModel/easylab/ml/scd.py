import numpy as np
from time import time
from multiprocessing import Pool


class SCD(object):
    """
    Stochastic coordinate descent for 01 loss optimization
    Numpy version Proto Type
    """

    def __init__(self, nrows, nfeatures, w_inc=100, tol=0.001, local_iter=10,
                 num_iters=100, interval=10, round=100, seed=2018, n_jobs=4):
        """

        :param nrows: ratio of training data in each iteration
        :param nfeatures: ratio of features in each iteration
        :param w_inc: w increment
        :param tol: stop threshold
        :param local_iter: the maximum number of iterations of updating all
                            columns
        :param num_iters: the number of iterations in each RR
        :param interval: interval in bias search if best index given
        :param round: number of round, RR
        :param seed: random seed
        :param n_jobs: number of process
        """
        self.nrows = nrows  #
        self.nfeatures = nfeatures  #
        self.w_inc = w_inc  #
        self.tol = tol  #
        self.num_iters = num_iters
        self.local_iter = local_iter
        self.round = round
        self.w = []
        self.b = []
        self.best_w = None
        self.best_b = None
        self.best_acc = None
        self.w_index_order = None
        self.obj = []
        self.orig_plus = 0
        self.orig_minus = 0
        self.plus_row_index = []
        self.minus_row_index = []
        self.yp = None
        self.interval = interval
        self.w_inc = w_inc * np.array([1, 100, 10000,
                                       -1, -100, -10000,  ],
                                      dtype=np.float32)
        self.n_jobs = n_jobs
        np.random.seed(seed)

    def train(self, train_data, train_labels):
        """

        :param train_data: training data, 2d array, float 32
        :param train_labels: training labels, 1d array, int8 (0,1)
        :return:
        """
        # initialize variable

        orig_cols = train_data.shape[1]
        train_labels = train_labels.astype(np.int8)

        # counting class and get their index
        for idx, value in enumerate(train_labels):
            if value == 1:
                self.orig_plus += 1
                self.plus_row_index.append(idx)
            else:
                self.orig_minus += 1
                self.minus_row_index.append(idx)

        # balanced pick rows and cols
        plus = max(2, int(self.orig_plus * self.nrows))
        minus = max(2, int(self.orig_minus * self.nrows))
        num_cols = max(min(5, orig_cols), int(self.nfeatures * orig_cols))
        # initialize up triangle matrix
        self.yp = np.triu(np.ones(plus + minus, dtype=np.int8), -1)

        # multi-process
        pool = Pool(self.n_jobs)
        results = []
        for r in range(self.round):
            # temp_w, temp_b, temp_obj = self.single_run(train_data, train_labels,
            #                                            plus, minus,)
            results.append(pool.apply_async(self.single_run,
                                            args=(
                                            train_data, train_labels, plus,
                                            minus)))
        pool.close()
        pool.join()
        for result in results:
            temp_w, temp_b, temp_obj = result.get()
            self.w.append(temp_w.reshape((-1, 1)))
            self.b.append(temp_b.reshape((1, 1)))
            self.obj.append(temp_obj)

        self.w = np.concatenate(self.w, axis=1)
        self.b = np.concatenate(self.b, axis=1)
        self.obj = np.array(self.obj)
        best_index = self.obj.argmax()
        self.best_acc = self.obj[best_index]
        self.best_w = self.w[:, best_index]
        self.best_b = self.b[:, best_index]

    def single_run(self, data, labels, plus, minus):
        """

        :param data: subset of training data
        :param labels: subset of training labels
        :param plus: number of +1 data points
        :param minus: number of -1 data points
        :return: best w, b, and acc we searched in this subset and evaluate on
                the full training set
        """

        best_acc = 0

        w = np.random.uniform(-1, 1, size=(data.shape[1],)).astype(np.float32)
        for i in range(self.num_iters):
            # pick rows randomly
            row_index = np.hstack([
                np.random.choice(self.plus_row_index, plus, replace=False),
                np.random.choice(self.minus_row_index, minus, replace=False)
            ])
            temp_w, temp_b = self.single_iteration(w, data[row_index],
                                                   labels[row_index], plus,
                                                   minus)

            temp_acc = self.eval(data, labels, temp_w, temp_b,
                                 batch_size=None)
            # if temp_acc > best_acc:
            w = temp_w
            b = temp_b
            best_acc = temp_acc

        # for result in results:
        #     temp_w, temp_b = result.get()
        #     w_matrix.append(temp_w.reshape((-1, 1)))
        #     b_matrix.append(temp_b)
        #
        # # build w matrix, each column of w_matrix is a set of w
        # w_matrix = np.concatenate(w_matrix, axis=1)
        # b_matrix = np.array(b_matrix).reshape((1, -1))
        #
        # # evaluate each pair of w and bias on full data set given
        # # and return the best pairs and loss
        # w, b, obj = self.eval(data, labels, w_matrix, b_matrix, batch_size=None)

        return w, b, best_acc

    def single_iteration(self, w, data, labels, plus, minus):
        """

        :param data: subset of training data
        :param labels: subset of training labels
        :param plus: number of +1 data points
        :param minus: number of -1 data points
        :return: temporary w and b we find in this subset
        """
        # initialize w followed uniform distribution (-1 ~ +1)
        # w = np.random.uniform(-1, 1, size=(data.shape[1],)).astype(np.float32)
        # L2 normalization
        # w = w / np.linalg.norm(w)
        temp_best_objective = 0
        best_w, best_b = self.get_best_w_and_b(0, data, labels, w,
                                               plus, minus,
                                               temp_best_objective)

        return best_w, best_b

    def argsort(self, x, kind='quicksort'):

        return np.argsort(x, kind=kind)

    def get_best_w_and_b(self, iter, data, labels, w, plus, minus,
                         best_objective):
        """

        :param iter: ignore
        :param data:
        :param labels:
        :param w:
        :param plus:
        :param minus:
        :param best_objective:
        :return:
        """
        projection = self.obtain_projection(data, w)
        raw_index_sorted = self.argsort(projection)
        projection = projection[raw_index_sorted]
        best_acc, b, best_index = self.get_best_b(labels, projection, None,
                                                  raw_index_sorted, None, plus,
                                                  minus)

        localit = 0
        while best_acc - best_objective > self.tol and localit < self.local_iter:
            updation_order = np.random.permutation(w.shape[0])
            for update_index in updation_order:
                sign = 0
                best_w_inc = 0
                best_b = b
                temp_projection = projection
                best_temp_index = best_index
                localit_ = 0

                for w_inc in self.w_inc:
                    projection_new = projection + data[:, update_index] * w_inc
                    raw_index_sorted_new = self.argsort(projection_new)
                    projection_new = projection_new[raw_index_sorted_new]
                    temp_acc, temp_b, temp_index = self.get_best_b(
                        labels, projection_new, best_index,
                        raw_index_sorted_new,
                        self.interval, plus, minus
                    )
                    if temp_acc > best_acc:
                        best_objective = best_acc
                        best_acc = temp_acc
                        best_w_inc = w_inc
                        sign = np.sign(w_inc)
                        best_b = temp_b
                        best_temp_index = temp_index
                        temp_projection = projection_new

                # update w by searched best w_inc
                projection = temp_projection
                b = best_b
                w[update_index] += best_w_inc
                best_index = best_temp_index
                redo = 1

                best_w_inc = 0
                while sign != 0 and localit_ < 10 and \
                        best_acc - best_objective > self.tol and redo > 0:
                    redo = 0
                    if sign == 1:
                        w_inc_group = self.w_inc[:self.w_inc.shape[0] // 2]
                    else:
                        w_inc_group = self.w_inc[self.w_inc.shape[0] // 2:]

                    for w_inc in w_inc_group:
                        projection_new = projection + \
                                         data[:, update_index] * w_inc
                        raw_index_sorted_new = self.argsort(projection_new)
                        projection_new = projection_new[raw_index_sorted_new]
                        temp_acc, temp_b, temp_index = self.get_best_b(
                            labels, projection_new, best_index,
                            raw_index_sorted_new,
                            self.interval, plus, minus
                        )
                        if temp_acc > best_acc:
                            best_objective = best_acc
                            best_acc = temp_acc
                            best_w_inc = w_inc
                            best_b = temp_b
                            best_index = temp_index
                            temp_projection = projection_new
                            redo += 1
                        else:
                            redo += 0

                    projection = temp_projection
                    b = best_b
                    w[update_index] += best_w_inc
                    best_index = best_temp_index
                    localit_ += 1

            localit += 1

        return w, b

    def cal_acc(self, labels, yp, plus, minus):

        gt = labels.reshape((1, -1))
        sum_ = yp + gt
        plus_correct = (sum_ == 2).sum(axis=1)
        minus_correct = (sum_ == 0).sum(axis=1)

        # balanced accuracy formula
        acc = (plus_correct / plus + minus_correct / minus) / 2.0
        best_index = acc.argmax()

        return best_index, acc[best_index]

    def obtain_projection(self, x, w):

        return x.dot(w)

    def get_best_b(self, labels, projection, index, raw_index_sorted,
                   interval, plus, minus):
        """

        :param labels:
        :param projection:
        :param index:
        :param raw_index_sorted:
        :param interval:
        :param plus:
        :param minus:
        :return:
        """
        gt = labels[raw_index_sorted]
        if index is None:
            yp = self.yp
            best_index, acc = self.cal_acc(gt, yp, plus, minus)
            if best_index == 0:
                b = -1 * projection[best_index] + 0.01
            elif best_index == plus + minus - 1:
                b = -1 * projection[best_index] - 0.01
            else:
                b = -1 * projection[best_index - 1: best_index].mean()

        else:
            start_index = max(0, index - interval)
            end_index = min(gt.shape[0], index + interval)
            yp = self.yp[start_index: end_index]
            best_index, acc = self.cal_acc(gt, yp, plus, minus)
            best_index += start_index
            if best_index == 0:
                b = -1 * projection[best_index] + 0.01
            elif best_index == plus + minus - 1:
                b = -1 * projection[best_index] - 0.01
            else:
                b = -1 * projection[best_index - 1: best_index].mean()

        return acc, b, best_index

    def eval(self, data, labels, w, b, batch_size):

        # if batch_size is None:
        #     yp = (np.sign(data.dot(w_matrix) + b_matrix) + 1) // 2
        #     gt = labels.reshape((-1, 1))
        #     gt = np.hstack([gt for i in range(b_matrix.shape[1])])
        #     acc = (yp == gt).sum(axis=0) / gt.shape[0]
        #     index = np.argmax(acc)  # get the highest accuracy's index
        #
        #     return w_matrix[:, index], b_matrix[:, index], acc[index]
        #
        # else:
        #     pass
        if batch_size is None:
            yp = (np.sign(data.dot(w) + b) + 1) // 2
            acc = (yp == labels).mean()

            return acc

    def predict(self, x, kind='best', prob=False):
        """

        :param x:
        :param kind:
        :param prob:
        :return:
        """
        if kind == 'best':
            yp = (np.sign(x.dot(self.best_w) + self.best_b) + 1) // 2

        elif kind == 'vote':
            yp = ((np.sign(x.dot(self.w) + self.b) + 1) // 2).mean(axis=1)
            if prob:
                return yp
            yp = yp.round().astype(np.int8)

        return yp


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_breast_cancer()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.int8)
    train_data, test_data, train_label, test_label = train_test_split(x, y)

    # svc = LinearSVC()
    # svc.fit(train_data, train_label)
    # print(svc.score(test_data, test_label))

    scd = SCD(nrows=0.8, nfeatures=1.0, w_inc=1, tol=0.001, num_iters=100,
              round=8, interval=10, seed=2018, n_jobs=8)
    a = time()
    scd.train(train_data, train_label)

    print('cost %.3f seconds' % (time() - a))
    yp = scd.predict(test_data)

    print('Accuracy: ',
          accuracy_score(y_true=train_label, y_pred=scd.predict(train_data)))
    print('Best one Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
    yp = scd.predict(test_data, kind='vote')
    print('vote  Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
    # output probability
    yp = scd.predict(test_data, kind='vote', prob=True)