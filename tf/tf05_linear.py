import tensorflow as tf
tf.set_random_seed(777)

# x and y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# tf.random_normal([1]) 랜덤하게 값을 하나 넣겠다

hypothesis = x_train * w + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # == model.compile(loss='mse'...

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)    # == model.compile(optimizer='...

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer()) # 텐서플로는 변수를 선언하면 항상 초기화를 해줘야함

    # Fit the line
    for step in range(2001):    # epochs
        _, cost_val, w_val, b_val = sess.run([train, cost, w, b])

        if step % 20 == 0:
            print(step, cost_val, w_val, b_val)
