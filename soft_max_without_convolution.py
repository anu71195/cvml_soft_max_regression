import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Shape of feature matrix:", mnist.train.images.shape)
print("Shape of target matrix:", mnist.train.labels.shape)

# Each target label is already provided in one-hot encoded form.
print("One-hot encoding for 1st observation:\n", mnist.train.labels[0])
# visualize data by plotting images
fig,ax = plt.subplots(10,10)
k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(mnist.train.images[k].reshape(28,28), aspect='auto')
        k += 1
# plt.show()

# number of features
num_features = 784
# number of target labels
num_labels = 10
# learning rate (alpha)
learning_rate = 0.03
# batch size
batch_size = 128
# number of epochs
num_steps = 15001
 
# input data
train_dataset = mnist.train.images
print(train_dataset);
print(train_dataset[0].shape);
train_labels = mnist.train.labels
test_dataset = mnist.test.images
test_labels = mnist.test.labels
valid_dataset = mnist.validation.images
valid_labels = mnist.validation.labels


# utility function to calculate accuracy
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu

# utility for tensorboard summaries
def variable_summaries(var):
   with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
    

# initialize a tensorflow graph
graph = tf.Graph()

with graph.as_default():
    """
    defining all the nodes
    """
 
    # Inputs
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
 
    # Weight matrix initialized to random values using a (truncated) normal distribution. 
    weights = tf.Variable(tf.truncated_normal([num_features, num_labels]))
    # variable_summaries(weights)
    # Biases initialized to zeroes
    biases = tf.Variable(tf.zeros([num_labels]))
    # variable_summaries(biases)
 
    # Training: Multiply the inputs with the weight matrix and add biases
    logits = tf.matmul(tf_train_dataset, weights) + biases
    
    """ 
    Cross-entropy is a distance calculation function which takes the calculated probabilities
    from softmax function and the created one-hot-encoding matrix to calculate the distance.
            D(Si,Ti)= -Ti.log(Si) where the '.' represents dot product, and log is applied elementwise on Si
    For calculating loss, we take the average of this cross-entropy across all training examples using tf.reduce_mean function
    """
    with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=tf_train_labels, logits=logits))
    tf.summary.scalar('loss', loss)
    # Optimizer: We're using gradient descent method for minimizing loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
 
    # Predictions for the training, validation, and test data.
    # Variables for accuracy calculation
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        # accuracy using tf
        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(tf_train_labels,1))
      with tf.name_scope('accuracy'):
        accuracy_node = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy_node)

    merged = tf.summary.merge_all()

with tf.Session(graph=graph) as session:
    # initialize weights and biases
    tf.global_variables_initializer().run()
    print("Initialized")

    # tensorboard graph
    writer = tf.summary.FileWriter('/tmp/tf_graphs/',session.graph)

 
    for step in range(num_steps):
        # pick a randomized offset
        offset = np.random.randint(0, train_labels.shape[0] - batch_size - 1)
 
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
 
        # Prepare the feed dict
        feed_dict = {tf_train_dataset : batch_data,
                     tf_train_labels : batch_labels}
 
        # run one step of computation
        _, l, predictions = session.run([optimizer, loss, train_prediction],
                                        feed_dict=feed_dict)
 
        summary,result = session.run([merged,accuracy_node], feed_dict=feed_dict)
        writer.add_summary(summary,step)

        if (step % 500 == 0):

            print("Minibatch loss at step {0}: {1}".format(step, l))
            print("Minibatch accuracy: {:.1f}%".format( 
                accuracy(predictions, batch_labels)))
            print("Validation accuracy: {:.1f}%".format(
                accuracy(valid_prediction.eval(), valid_labels)))
 
    print("\nTest accuracy: {:.1f}%".format(
        accuracy(test_prediction.eval(), test_labels)))