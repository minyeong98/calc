import numpy as np
import keras
from random import seed
from random import randint
from math import sqrt
from sklearn.metrics import mean_squared_error

def add_function(cnt, num, max):
    x = list()
    y = list()
    for i in range(cnt):
        input = [randint(1, max) for _ in range(num)]
        output = input[0] + input[1]
        # print(input, output)
        x.append(input)
        y.append(output)
    x = np.array(x)
    y = np.array(y)
    # print(x)
    # print(y)
    x = x.astype('float') / float(max * num)
    y = y.astype('float') / float(max * num)
    # print(x)
    # print(y)
    return x, y

def sub_function(cnt, num, max):
    x = list()
    y = list()
    for i in range(cnt):
        input = [randint(1, max) for _ in range(num)]
        output = input[0] - input[1]
        # print(input, output)
        x.append(input)
        y.append(output)
    x = np.array(x)
    y = np.array(y)
    # print(x)
    # print(y)
    x = (x.astype('float') + float(100)) / float(max * num)
    y = (y.astype('float') + float(100)) / float(max * num)
    # print(x)
    # print(y)
    return x, y

def mul_function(cnt, num, max):
    x = list()
    y = list()
    for i in range(cnt):
        input = [randint(1, max) for _ in range(num)]
        output = input[0] * input[1]
        # print(input, output)
        x.append(input)
        y.append(output)
    x = np.array(x)
    y = np.array(y)
    # print(x)
    # print(y)
    x = (x.astype('float')/float(100)) / float(max * num)
    y = (y.astype('float')/float(100)) / float(max * num)
    # print(x)
    # print(y)
    return x, y

def div_function(cnt, num, max):
    x = list()
    y = list()
    for i in range(cnt):
        input = [randint(1, max) for _ in range(num)]
        output = input[0] / input[1]
        # print(input, output)
        x.append(input)
        y.append(output)
    x = np.array(x)
    y = np.array(y)
    # print(x)
    # print(y)
    x = (x.astype('float')*float(100)) / float(max * num)
    y = (y.astype('float')*float(100)) / float(max * num)
    # print(x)
    # print(y)
    return x, y

def invert_add(val, num, max):
    return np.round(val * float(max * num))

def invert_sub(val, num, max):
    return np.round(val * float(max * num)-float(100))

def invert_mul(val, num, max):
    return np.round(val * float(max * num) * float(100))

def invert_div(val, num, max):
    return (val * float(max * num) / float(100))

# seed(1)
cnt = 100
num = 2
max = 100

n_batch = 2
n_epoch = 200
# n_epoch = 500

# 모델 생성
model = keras.Sequential()
model.add(keras.layers.Dense(4, input_dim=num))
model.add(keras.layers.Dense(2))
model.add(keras.layers.Dense(1))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')

# 덧셈 모델 훈련
def train_add():
    for _ in range(n_epoch):
        x, y = add_function(cnt, num, max)
        model.fit(x, y, epochs=30, batch_size=n_batch, verbose=2)
    print("<< 덧셈 훈련 완료 >>")
    result = model.predict(x, batch_size=2, verbose=2)

    expected = [invert_add(x, num, max) for x in y]
    predicted = [invert_add(x, num, max) for x in result[:, 0]]

    rmse = sqrt(mean_squared_error(expected, predicted))
    print('RMSE_덧셈: %f' % rmse)

    err_sum = 0
    for i in range(10):
        error = expected[i] - predicted[i]
        print(invert_add(x[i][0], num, max), "+", invert_add(x[i][1], num, max), "=", invert_add(y[i], num, max),
              "==> 정답 = %d, 예측 = %d, Error = %d" % (expected[i], predicted[i], error))
        err_sum += error
    print("add_avg_Err: {}".format(err_sum/len(expected)))

    correct = 0
    for i in range(len(expected)):
        if(expected[i] == predicted[i]):
            correct += 1

    print("정확도 = %.2f" % (correct/len(expected)))

# 뺄셈 모델 훈련
def train_sub():
    for _ in range(n_epoch):
        x, y = sub_function(cnt, num, max)
        model.fit(x, y, epochs=30, batch_size=n_batch, verbose=2)
    print("<< 뺄셈 훈련 완료 >>")
    result = model.predict(x, batch_size=2, verbose=2)

    expected = [invert_sub(x, num, max) for x in y]
    predicted = [invert_sub(x, num, max) for x in result[:, 0]]

    rmse = sqrt(mean_squared_error(expected, predicted))
    print('RMSE_뺄셈: %f' % rmse)

    err_sum = 0
    for i in range(10):
        error = expected[i] - predicted[i]
        print(invert_sub(x[i][0], num, max), "-", invert_sub(x[i][1], num, max), "=", invert_sub(y[i], num, max),
              "==> 정답 = %d, 예측 = %d, Error = %d" % (expected[i], predicted[i], error))
        err_sum += error
    print("sub_avg_Err: {}".format(err_sum/len(expected)))

    correct = 0
    for i in range(len(expected)):
        if(expected[i] == predicted[i]):
            correct += 1

    print("정확도 = %.2f" % (correct/len(expected)))

# 곱셈 모델 훈련
def train_mul():
    for _ in range(n_epoch):
        x, y = mul_function(cnt, num, max)
        model.fit(x, y, epochs=10, batch_size=n_batch, verbose=2)
    print("<< 곱셈 훈련 완료 >>")
    result = model.predict(x, batch_size=2, verbose=2)

    expected = [invert_mul(x, num, max) for x in y]
    predicted = [invert_mul(x, num, max) for x in result[:, 0]]

    rmse = sqrt(mean_squared_error(expected, predicted))
    print('RMSE_곱셈: %f' % rmse)

    err_sum = 0
    for i in range(10):
        error = expected[i] - predicted[i]
        print(invert_mul(x[i][0], num, max), "*", invert_mul(x[i][1], num, max), "=", invert_mul(y[i], num, max),
              "==> 정답 = %d, 예측 = %d, Error = %d" % (expected[i], predicted[i], error))
        err_sum += error
    print("mul_avg_Err: {}".format(err_sum/len(expected)))

    correct = 0
    for i in range(len(expected)):
        if(expected[i] == predicted[i]):
            correct += 1

    print("정확도 = %.2f" % (correct/len(expected)))

# 나눗셈 모델 훈련
def train_div():
    for _ in range(n_epoch):
        x, y = div_function(cnt, num, max)
        model.fit(x, y, epochs=10, batch_size=n_batch, verbose=2)
    print("<< 나눗셈 훈련 완료 >>")
    result = model.predict(x, batch_size=2, verbose=2)

    expected = [invert_div(x, num, max) for x in y]
    predicted = [invert_div(x, num, max) for x in result[:, 0]]

    rmse = sqrt(mean_squared_error(expected, predicted))
    print('RMSE_나눗셈: %f' % rmse)

    err_sum = 0
    for i in range(10):
        error = expected[i] - predicted[i]
        print(invert_div(x[i][0], num, max), "/", invert_div(x[i][1], num, max), "=", invert_div(y[i], num, max),
              "==> 정답 = %.2f, 예측 = %.2f, Error = %.2f" % (expected[i], predicted[i], error))
        err_sum += error
    print("add_avg_Err: {}".format(err_sum/len(expected)))

    correct = 0
    for i in range(len(expected)):
        if(expected[i] == predicted[i]):
            correct += 1

    print("정확도 = %.2f" % (correct/len(expected)))

print("<< 덧셈 >>")
train_add()

print("<< 뺼셈 >>")
train_sub()

print("<< 곱셈 >>")
train_mul()

print("<< 나눗셈 >>")
train_div()