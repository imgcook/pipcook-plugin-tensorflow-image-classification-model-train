import tensorflow as tf
import sys

def train(xs, ys, model, j, batchesPerEpoch, trainSummaryWriter, iteration):
  xs = tf.stack(xs)
  ys = tf.stack(ys)
  trainRes = model.train_on_batch(xs, ys)
  if (j % (int(batchesPerEpoch / 10)) == 0):
    print('Iteration {}/{} result --- loss: {} accuracy: {}'.format(j, batchesPerEpoch, trainRes[0], trainRes[1]))
  sys.stdout.flush()
  with trainSummaryWriter.as_default():
    tf.summary.scalar('loss', trainRes[0], step=iteration)
    tf.summary.scalar('accuracy', trainRes[1], step=iteration)

  
def evaluate(xs, ys, model):
  xs = tf.stack(xs)
  ys = tf.stack(ys)
  res = model.evaluate(xs, ys, verbose=0)
  return res
