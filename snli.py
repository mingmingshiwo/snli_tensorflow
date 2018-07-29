import numpy as np
import logging
import tensorflow as tf
import os
from tqdm import tqdm, trange
logging.basicConfig(format='%(asctime)s,%(msecs)-3d %(levelname)-5s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

'''
https://github.com/Smerity/keras_snli
'''

# GPU_IDXS = [0,1]
GPU_IDXS = [2,3]
# GPU_IDXS = [0,1,2,3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(e) for e in GPU_IDXS)

tqdm.monitor_interval = 0

flags = tf.flags
flags.DEFINE_integer("num_parallel_calls", 20, "")
flags.DEFINE_string("mode", "gpu", "")
flags.DEFINE_string("version", "5", "")
flags.DEFINE_integer("max_length", 42, "")
flags.DEFINE_integer("hidden", 300, "")
flags.DEFINE_integer("num_labels", 3, "")
flags.DEFINE_integer("max_epoch", 100, "")
flags.DEFINE_integer("batch_size", 512 * len(GPU_IDXS), "")
flags.DEFINE_integer("num_gpus", len(GPU_IDXS), "")
flags.DEFINE_integer("shuffle_batch", 30, "")
flags.DEFINE_integer("log_loss_period", 200, "")
flags.DEFINE_float("keep_prob", 0.8, "")
flags.DEFINE_float("L2_reg", 4e-6, "")
flags.DEFINE_integer("patience_epoch", 8, "")
flags.DEFINE_integer("log_period_step", 10, "")
flags.DEFINE_float("loss_base_line", 0.6, "")
flags.DEFINE_string("summaries_dir", "/tmp/tfb", "")

config = flags.FLAGS
assert config.version
flags.DEFINE_string("word_emb_mat_file", 'snli/word_emb_mat_{}'.format(config.version), "")
flags.DEFINE_string("tarin_record_file", 'snli/train_record_{}'.format(config.version), "")
flags.DEFINE_string("dev_record_file", 'snli/dev_record_{}'.format(config.version), "")
flags.DEFINE_string("test_record_file", 'snli/test_record_{}'.format(config.version), "")

logger.info("[GPU_IDXS]{} [version]{}".format(GPU_IDXS, config.version))

def _parse_func(example_proto):
    features = tf.parse_single_example(example_proto,
                                       features={
                                           "q1_idx": tf.FixedLenFeature([config.max_length], tf.int64),
                                           "q2_idx": tf.FixedLenFeature([config.max_length], tf.int64),
                                           "label": tf.FixedLenFeature([3], tf.int64),
                                           "id": tf.FixedLenFeature([1], tf.int64)
                                       })
    q1_idx = features['q1_idx']
    q2_idx = features['q2_idx']
    label = features['label']
    return q1_idx,q2_idx,label

def _get_batch_dataset(record_file , is_train):
    size = sum(1 for _ in tf.python_io.tf_record_iterator(record_file))

    dataset = tf.data.TFRecordDataset(record_file)
    dataset = dataset.map(_parse_func, num_parallel_calls=config.num_parallel_calls)
    if is_train:
        dataset = dataset.shuffle(config.batch_size * config.shuffle_batch)
        dataset = dataset.batch(config.batch_size)
        dataset = dataset.repeat()
        batch_num = size // config.batch_size
        if size % config.batch_size > 0:
            batch_num += 1
        return dataset, size, batch_num

    else:
        dataset = dataset.batch(size)
        return dataset, size, 1

def _average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train():
    #dataset
    train_dataset, train_size, train_batch_num = _get_batch_dataset(config.tarin_record_file, True)
    dev_dataset, dev_size, _ = _get_batch_dataset(config.dev_record_file, False)
    test_dataset, test_size, _ = _get_batch_dataset(config.test_record_file, False)
    logger.info("Dataset size [train]{} [dev]{} [test]{}".format(train_size, dev_size, test_size,))

    #placeholder
    handle = tf.placeholder(tf.string, shape=[])
    is_train = tf.placeholder(tf.bool, shape=[])

    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes
    )

    #func
    def dropout(args, keep_prob=config.keep_prob):
        return tf.cond(is_train, lambda :tf.nn.dropout(args, keep_prob), lambda: args)

    l2_reg = tf.contrib.layers.l2_regularizer(config.L2_reg)

    #emb
    word_emb_np = np.load(config.word_emb_mat_file + '.npy')
    x1_idx_all, x2_idx_all, y_all = iterator.get_next()

    #grads
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer()
    tower_grads = []
    loss_list = []
    accurate_num_list = []

    actual_batch_size = tf.shape(x1_idx_all)[0]
    tf.assert_greater(actual_batch_size, config.num_gpus)
    single_gpu_batch_size = tf.cast(tf.divide(actual_batch_size, config.num_gpus), tf.int32)

    for i in range(config.num_gpus):
        with tf.device('/gpu:{}'.format(i)):
            with tf.name_scope("tower_{}".format(i)):
                with tf.name_scope("slice"):
                    start = i * single_gpu_batch_size
                    end = tf.cond(tf.equal(i, config.num_gpus-1), lambda :actual_batch_size, lambda :(i + 1) * single_gpu_batch_size)
                    x1_idx = x1_idx_all[start: end]
                    x2_idx = x2_idx_all[start: end]
                    y = y_all[start: end]

                with tf.variable_scope("emb"):
                    word_emb = tf.Variable(tf.constant(word_emb_np, dtype=tf.float32), False, name = 'word_emb')
                    x1 = tf.nn.embedding_lookup(word_emb, x1_idx)
                    x2 = tf.nn.embedding_lookup(word_emb, x2_idx)
    
                with tf.variable_scope("time_distributed", reuse=tf.AUTO_REUSE):
                    shape = x1.shape
                    x1 = tf.reshape(x1, [-1, shape[2]])
                    x2 = tf.reshape(x2, [-1, shape[2]])
    
                    w_t = tf.get_variable('w_t', shape=[config.hidden, config.hidden], initializer=tf.contrib.layers.xavier_initializer())
                    b_t = tf.get_variable('b_t', shape=[config.hidden], initializer=tf.random_normal_initializer())
    
                    x1 = tf.nn.relu(tf.nn.xw_plus_b(x1, w_t, b_t))
                    x2 = tf.nn.relu(tf.nn.xw_plus_b(x2, w_t, b_t))
    
                    x1 = tf.reshape(x1, [-1, shape[1], shape[2]])
                    x2 = tf.reshape(x2, [-1, shape[1], shape[2]])
    
                with tf.name_scope("sum_pool"):
                    pool_1 = tf.reduce_sum(x1, 1)
                    with tf.variable_scope("pool_1", reuse=tf.AUTO_REUSE):
                        pool_1 = tf.layers.batch_normalization(pool_1, training=is_train)
                    pool_2 = tf.reduce_sum(x2, 1)
                    with tf.variable_scope("pool_2", reuse=tf.AUTO_REUSE):
                        pool_2 = tf.layers.batch_normalization(pool_2, training=is_train)
    
                with tf.name_scope("concat"):
                    s_v = tf.concat([pool_1, pool_2], 1)
                    s_v = dropout(s_v)
    
                with tf.variable_scope("L1", reuse=tf.AUTO_REUSE):
                    w_1 = tf.get_variable('w_1', shape=[config.hidden * 2, config.hidden * 2],initializer=tf.contrib.layers.xavier_initializer(), regularizer=l2_reg)
                    l1 = tf.matmul(s_v, w_1)
                    l1 = tf.layers.batch_normalization(l1, training=is_train)
                    l1 = tf.nn.relu(l1)
                    l1 = dropout(l1)

                with tf.variable_scope("L2", reuse=tf.AUTO_REUSE):
                    w_2 = tf.get_variable('w_2', shape=[config.hidden * 2, config.hidden * 2],initializer=tf.contrib.layers.xavier_initializer(), regularizer=l2_reg)
                    l2 = tf.matmul(l1, w_2)
                    l2 = tf.layers.batch_normalization(l2, training=is_train)
                    l2 = tf.nn.relu(l2)
                    l2 = dropout(l2)

                with tf.variable_scope("L3", reuse=tf.AUTO_REUSE):
                    w_3 = tf.get_variable('w_3', shape=[config.hidden * 2, config.hidden * 2],initializer=tf.contrib.layers.xavier_initializer(), regularizer=l2_reg)
                    l3 = tf.matmul(l2, w_3)
                    l3 = tf.layers.batch_normalization(l3, training=is_train)
                    l3 = tf.nn.relu(l3)
                    l3 = dropout(l3)

                with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
                    w_y = tf.get_variable('w_y', shape=[config.hidden * 2, config.num_labels],initializer=tf.contrib.layers.xavier_initializer())
                    b_y = tf.get_variable('b_y', shape=[config.num_labels],initializer=tf.random_normal_initializer())
                    y_scores = tf.nn.xw_plus_b(l3, w_y, b_y)
    
                with tf.name_scope("loss"):
                    loss_labels = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=y_scores,
                        labels=y
                    )
                    loss_op = tf.reduce_mean(loss_labels)
                    loss_list.append(loss_op)
    
                with tf.name_scope("grads"):
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        grads = optimizer.compute_gradients(loss_op)
                        tower_grads.append(grads)

        with tf.device("/cpu"):
            with tf.name_scope("metrics"):
                actual = tf.argmax(y, 1)
                predicted = tf.argmax(y_scores, 1)

                accurate_num = tf.count_nonzero(tf.cast(tf.equal(actual, predicted), tf.int32))
                accurate_num_list.append(accurate_num)

    with tf.name_scope("collect_loss"):
        loss_avg = tf.reduce_mean(loss_list)
        reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if reg_set:
            l2_loss = tf.add_n(reg_set)
            loss_avg += l2_loss

    with tf.name_scope("collect_accu"):
        accurate_num_total = tf.reduce_sum(accurate_num_list)
        accu_rate = tf.divide(tf.cast(accurate_num_total, tf.float32), tf.cast(actual_batch_size, tf.float32))

    with tf.name_scope("collect_grads"):
        grads_avg = _average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(grads_avg, global_step=global_step)

    tf.summary.scalar("met/loss", loss_avg)
    tf.summary.scalar("met/accu", accu_rate)

    tf.summary.histogram("his/w_t", w_t)
    tf.summary.histogram("his/w_1", w_1)
    tf.summary.histogram("his/w_2", w_2)
    tf.summary.histogram("his/w_3", w_3)
    tf.summary.histogram("his/l1", l1)
    tf.summary.histogram("his/l2", l2)
    tf.summary.histogram("his/l3", l3)

    tf.summary.histogram("his/w_y", w_y)
    tf.summary.histogram("his/y_scores", y_scores)

    summaries = tf.summary.merge_all()

    #sess
    session_config = tf.ConfigProto()
    # session_config.log_device_placement = True
    session_config.allow_soft_placement = True

    with tf.Session(config=session_config) as sess:
        # from tensorflow.python import debug as tf_debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        train_writer = tf.summary.FileWriter(config.summaries_dir + "/train", sess.graph)
        dev_writer = tf.summary.FileWriter(config.summaries_dir + "/dev")
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        # exit()

        best_loss = config.loss_base_line
        best_loss_step = 0
        best_loss_epoch = 0
        patience = 0

        train_handle = sess.run(train_dataset.make_one_shot_iterator().string_handle("train_hd"))
        #batch train
        for epoch in range(0, config.max_epoch):
            with trange(train_batch_num) as t:
                for batch_idx in t:
                    t.set_description('EPOCH {:>2}'.format(epoch))

                    if batch_idx % config.log_period_step != 0:
                        _, step = sess.run([train_op, global_step],
                                           feed_dict={handle: train_handle, is_train: True})
                    else:
                        _, step, loss_train, accu_rate_train, summ_train = sess.run([train_op, global_step, loss_avg, accu_rate, summaries],
                                                                                    feed_dict={handle: train_handle, is_train: True})
                        t.set_postfix({"loss": "{:.4f}".format(loss_train), "accu": "{:.4f}".format(accu_rate_train)})
                        train_writer.add_summary(summ_train, step)

                    #epoch end
                    if batch_idx == train_batch_num - 1:
                        dev_handle = sess.run(dev_dataset.make_one_shot_iterator().string_handle("dev_hd"))
                        loss_dev, accu_rate_dev, summ_dev = sess.run([loss_avg, accu_rate, summaries],
                                                                     feed_dict={handle: dev_handle,is_train: False})
                        dev_writer.add_summary(summ_train, step)
                        t.set_postfix({"loss":"{:.4f}".format(loss_train), "accu":"{:.4f}".format(accu_rate_train), #same as above
                                       "dev_loss":"{:.4f}".format(loss_dev), "dev_accu":"{:.4f}".format(accu_rate_dev)})

                        if loss_dev < best_loss:
                            best_loss = loss_dev
                            best_loss_step = step
                            best_loss_epoch = epoch
                            patience = 0
                            saver.save(sess, 'ckp/snli_ckp', step)
                            test_handle = sess.run(test_dataset.make_one_shot_iterator().string_handle("test_hd"))
                            loss_test, accu_rate_test = sess.run([loss_avg, accu_rate],
                                                                 feed_dict={handle: test_handle,is_train: False})
                        else:
                            if best_loss_step:
                                patience += 1

                        if patience >= config.patience_epoch:
                            print()
                            logger.info("Early stop at [epoch]{} [step]{}".format(epoch, step))
                            logger.info("Best [epoch]{} [step]{} [loss]{}".format(best_loss_epoch, best_loss_step, best_loss))
                            logger.info('Test metric [loss]{:.6f} [accu rate]{:.6f}'.format(loss_test, accu_rate_test))
                            exit()

def main(args = None):
    train()

if __name__ == '__main__':
    tf.app.run()
