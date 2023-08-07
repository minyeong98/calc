from random import seed, randint
from math import ceil, log10
from numpy import array, argmax
from tensorflow import keras

#데이터 생성
def random_sum_pairs(n_samples, n_numbers, largest):
    X, Y = list(), list()
    for i in range(n_samples):
        inputN = [randint(1, largest) for _ in range(n_numbers)]
        outputN = inputN[0] - inputN[1]
        #print(inputN, outputN)
        X.append(inputN)
        Y.append(outputN)
    return X, Y

#정수->문자열
def to_string(X, Y, n_numbers, largest):
    max_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1
    Xstr = list()
    for pattern in X:
        strp = '-'.join([str(n) for n in pattern])
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        Xstr.append(strp)
    max_length = ceil(log10(largest + 1))
    Ystr = list()
    for pattern in Y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        Ystr.append(strp)
    return Xstr, Ystr

#문자열의 각 문자를 정수 값으로 인코딩
def integer_encode(X, Y, alphabet):
    ctoi = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [ctoi[char] for char in pattern]
        Xenc.append(integer_encoded)
    Yenc = list()
    for pattern in Y:
        integer_encoded = [ctoi[char] for char in pattern]
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
    #print(X.dtype, Y.dtype)
    return X, Y

#인코딩을 반전하여 출력 벡터를 다시 숫자로 변환
def invert(seq, alphabet):
    itoc = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = itoc[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)

n_samples = 1000
n_numbers = 2
largest = 1000
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ' ']
n_chars = len(alphabet)
n_in_seq_length = n_numbers * ceil(log10(largest+1)) + n_numbers - 1
n_out_seq_length = ceil(log10(n_numbers * (largest+1)))

model = keras.Sequential()
model.add(keras.layers.LSTM(100, input_shape=(n_in_seq_length, n_chars)))
model.add(keras.layers.RepeatVector(n_out_seq_length))
model.add(keras.layers.LSTM(50, return_sequences=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(n_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

X, Y = generate_data(n_samples, n_numbers, largest, alphabet)
model.fit(X, Y, epochs=150, batch_size=10)
result = model.predict(X, batch_size=10, verbose=0)

expected = [invert(X, alphabet) for X in Y]
predicted = [invert(X, alphabet) for X in result]

#테스트 연산 생성
for i in range(30):
    print(i+1, ':', invert(X[i], alphabet), '==> (예측값 = %s, 정답 = %s)' % (predicted[i], expected[i]))
