import random
from random import seed, randint
from math import ceil, log10
from numpy import array, argmax
from keras.utils import pad_sequences
from tensorflow import keras


# 각 시퀀스의 정수 수, 가장 큰 정수를 사용하여 모델링을 위해 X, Y 쌍의 데이터를 생성하고 반환하는 함수
def random_sum_pairs(n_samples, n_numbers, largest):
    X, Y = list(), list()
    for i in range(n_samples):
        in_pattern = [random.randrange(1, largest) for _ in range(n_numbers)]
        out_pattern = in_pattern[0] / in_pattern[1]
        #print(in_pattern, out_pattern)
        X.append(in_pattern)
        Y.append(out_pattern)
    return X, Y

#정수를 문자열로 반환
def to_string(X, Y, n_numbers, largest):
    max_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1
    Xstr = list()
    for pattern in X:
        strp = '/'.join([str(n) for n in pattern])
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        Xstr.append(strp)
    max_length = ceil(log10(largest + 1))
    Ystr = list()
    for pattern in Y:
        strp = str(pattern)
        strp = '.'.join([' ' for _ in range(max_length - len(strp))]) + strp
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        Ystr.append(strp)
    return Xstr, Ystr

#문자열의 각 문자를 정수 값으로 인코딩
def integer_encode(X, Y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    Yenc = list()
    for pattern in Y:
        integer_encoded = [char_to_int[char] for char in pattern]
        Yenc.append(integer_encoded)
    return Xenc, Yenc

#정수 인코딩 시퀀스를 이진 인코딩
def one_hot_encode(X, Y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    Yenc = list()
    for seq in Y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Yenc.append(pattern)
    return Xenc, Yenc


def generate_data(n_samples, n_numbers, largest, alphabet):
    X, Y = random_sum_pairs(n_samples, n_numbers, largest)
    X, Y = to_string(X, Y, n_numbers, largest)
    X, Y = integer_encode(X, Y, alphabet)
    X, Y = one_hot_encode(X, Y, len(alphabet))
    X, Y = array(X), array(Y)
    #print(X.shape, Y.shape)
    return X, Y

#인코딩을 반전하여 출력 벡터를 다시 숫자로 변환
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)


n_samples = 1000
n_numbers = 2
largest = 1000
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', ' ', '.']
n_chars = len(alphabet)
n_in_seq_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1
n_out_seq_length = ceil(log10(n_numbers * (largest + 1)))
n_batch = 10

model = keras.Sequential()
model.add(keras.layers.LSTM(100, input_shape=(n_in_seq_length, n_chars)))
model.add(keras.layers.RepeatVector(n_out_seq_length))
model.add(keras.layers.LSTM(50, return_sequences=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(n_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())

X, Y = generate_data(n_samples, n_numbers, largest, alphabet)
# X = keras.preprocessing.sequence.pad_sequences(X)
# Y = keras.preprocessing.sequence.pad_sequences(Y)
print(X.dtype, Y.dtype)

model.fit(X, Y, epochs=150, batch_size=n_batch)
result = model.predict(X, batch_size=n_batch, verbose=0)

expected = [invert(X, alphabet) for X in Y]
predicted = [invert(X, alphabet) for X in result]

for i in range(30):
    print(i + 1, ':', invert(X[i], alphabet), '==> (예측값 = %s, 정답 = %s)' % (predicted[i], expected[i]))
