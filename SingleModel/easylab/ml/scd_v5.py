import numpy as np
from time import time
from multiprocessing import Pool


class SCD(object):
    """
    Stochastic coordinate descent for 01 loss optimization
    Numpy version Proto Type
    """

    def __init__(self, nrows, nfeatures, w_inc=100, tol=0.001, local_iter=10,
                 num_iters=100, interval=20, round=100, seed=2018,
                 update_features=500, adjust_inc=False, n_jobs=4):
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
        self.update_features = update_features
        self.w = []
        self.b = []
        self.best_w = None
        self.best_b = None
        self.best_acc = None
        self.best_w_index = None
        self.w_index_order = None
        self.obj = []
        self.orig_plus = 0
        self.orig_minus = 0
        self.plus_row_index = []
        self.minus_row_index = []
        self.yp = None
        self.interval = interval
        self.adjust_inc = adjust_inc
        self.inc_scale = w_inc
        # self.step = np.array([1, 10, 1e2, 1e3,
        #                                -1, -10, -1e2, -1e3],
        #                               dtype=np.float32)
        self.w_inc = None
        self.step = np.array([1, 2, 3, 4,
                              -1, -2, -3, -4],
                                      dtype=np.float32)
        self.ref_full_index = None
        self.ref_half_index = None
        self.w_index = []
        self.n_jobs = n_jobs
        self.w_inc_stats = []
        # np.random.seed(seed)

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

        # initialize up triangle matrix and reference index
        rows_sum = plus + minus
        self.yp = np.triu(np.ones(rows_sum, dtype=np.int8), 0)
        self.ref_full_index = np.repeat(
            np.arange(self.step.shape[0]).reshape((-1, 1)), rows_sum, axis=1)
        self.ref_half_index = np.repeat(
            np.arange(self.step.shape[0] // 2).reshape((-1, 1)), rows_sum,
            axis=1)

        # multi-process
        pool = Pool(self.n_jobs)
        results = []
        for r in range(self.round):
            column_indices = np.random.choice(np.arange(orig_cols), num_cols,
                                              replace=False)
            self.w_index.append(column_indices)
            # temp_w, temp_b, temp_obj = self.single_run(
            #     train_data[:, column_indices], train_labels, plus, minus, )
            results.append(pool.apply_async(self.single_run,
                                            args=(train_data[:, column_indices],
                                                  train_labels, plus, minus)))

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
        self.best_w_index = self.w_index[best_index]

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
        # L2 normalization
        w = w / np.linalg.norm(w)
        if self.adjust_inc:
            self.w_inc = self.step * w.max()/5
        else:
            self.w_inc = self.inc_scale * self.step
        # temp_w = w
        for i in range(self.num_iters):
            # print('current iters: %d' % i)
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
            if temp_acc > best_acc:
                w = temp_w
                b = temp_b
                best_acc = temp_acc
                if self.adjust_inc:
                    self.w_inc = self.step * w.max()/5

                print('iter %d temp acc: %.3f, w_inc: %.5f' % (i, temp_acc, w.max()/5))
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

    def argsort(self, x, axis=-1, kind='mergesort'):

        return np.argsort(x, axis=axis, kind=kind)

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
        # labels = label.copy()
        projection = self.obtain_projection(data, w)
        raw_index_sorted = self.argsort(projection)
        projection = projection[raw_index_sorted]
        # labels = labels[raw_index_sorted]
        best_acc, b, best_index = self.get_best_b(
            labels[raw_index_sorted], projection, None, None, plus, minus)

        localit = 0
        while best_acc - best_objective > self.tol and localit < self.local_iter:
            # print('local iter: %d' % localit)
            updation_order = np.random.permutation(w.shape[0])
            for update_index in updation_order[:self.update_features]:
                best_objective = best_acc
                sign = 0
                best_b = b
                best_temp_index = best_index
                localit_ = 0

                ###
                # updation
                w_ = np.repeat(w.reshape((-1, 1)), self.w_inc.shape[0], axis=1)
                w_[update_index, :] += self.w_inc
                w_ /= np.linalg.norm(w_, axis=0)
                projection = self.obtain_projection(data, w_).T  # Transpose
                raw_index_sorted = self.argsort(projection, axis=1)
                projection = projection[self.ref_full_index, raw_index_sorted]
                temp_labels = labels[raw_index_sorted]

                for i in range(self.w_inc.shape[0]):
                    temp_acc, temp_b, temp_index = self.get_best_b(
                        temp_labels[i], projection[i], best_temp_index,
                        self.interval, plus, minus
                    )
                    if temp_acc > best_acc:
                        # print('update..0')
                        best_acc = temp_acc
                        best_w_inc = self.w_inc[i]
                        # print(best_w_inc)
                        sign = np.sign(best_w_inc)
                        best_b = temp_b
                        best_temp_index = temp_index
                        w = w_[:, i]
                        redo = 1

                while sign != 0 and localit_ < 10 and \
                        best_acc - best_objective > self.tol and redo > 0:
                    best_objective = best_acc
                    redo = 0
                    if sign == 1:
                        w_inc_group = self.w_inc[:self.w_inc.shape[0] // 2]
                    else:
                        w_inc_group = self.w_inc[self.w_inc.shape[0] // 2:]
                    #################
                    w_ = np.repeat(w.reshape((-1, 1)), w_inc_group.shape[0],
                                   axis=1)
                    w_[update_index, :] += w_inc_group
                    w_ /= np.linalg.norm(w_, axis=0)
                    projection = self.obtain_projection(data, w_).T  # Transpose
                    raw_index_sorted = self.argsort(projection, axis=1)
                    projection = projection[
                        self.ref_half_index, raw_index_sorted]
                    temp_labels = labels[raw_index_sorted]

                    for i in range(w_inc_group.shape[0]):
                        temp_acc, temp_b, temp_index = self.get_best_b(
                            temp_labels[i], projection[i], best_temp_index,
                            self.interval, plus, minus
                        )

                        if temp_acc > best_acc:
                            # print('update..1')
                            best_acc = temp_acc
                            best_w_inc = w_inc_group[i]
                            # print(best_w_inc)
                            sign = np.sign(best_w_inc)
                            best_b = temp_b
                            best_temp_index = temp_index
                            w = w_[:, i]
                            redo += 1
                        else:
                            redo += 0

                    localit_ += 1
                    # print('inner loop iter: %d' % localit_)

            localit += 1

        return w, best_b

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

    def get_best_b(self, labels, projection, index, interval, plus, minus):
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
        gt = labels.copy()
        if index is None:
            yp = self.yp
            best_index, acc = self.cal_acc(gt, yp, plus, minus)
            if best_index == 0:
                b = -1 * projection[best_index] + 0.01
            # elif best_index == plus + minus - 1:
            #     b = -1 * projection[best_index] - 0.01
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
            # elif best_index == plus + minus - 1:
            #     b = -1 * projection[best_index] - 0.01
            else:
                b = -1 * projection[best_index - 1: best_index].mean()

        return acc, b, best_index

    def eval(self, data, labels, w, b, batch_size):
        """

        :param data:
        :param labels:
        :param w_matrix:
        :param b_matrix:
        :param batch_size:
        :return:
        """
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
            yp = (np.sign(x[:, self.best_w_index].dot(self.best_w) +
                          self.best_b) + 1) // 2

        elif kind == 'vote':
            yp = np.zeros((x.shape[0], self.round), dtype=np.float32)
            for i in range(self.round):
                yp[:, i] = (np.sign(x[:, self.w_index[i]].dot(self.w[:, i]) +
                                    self.b[:, i]) + 1) // 2

            if prob:
                return yp.mean(axis=1)
            yp = yp.mean(axis=1).round().astype(np.int8)

        return yp


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    np.random.seed(20156)
    data = load_breast_cancer()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.int8)
    train_data, test_data, train_label, test_label = train_test_split(x, y)

    # svc = LinearSVC()
    # svc.fit(train_data, train_label)
    # print(svc.score(test_data, test_label))

    scd = SCD(nrows=0.6, nfeatures=0.5, w_inc=1, tol=0.001, local_iter=100,
              num_iters=100, round=64, interval=10, seed=2018, adjust_inc=True,
              n_jobs=8)
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