import os
import numpy as np
import tensorflow as tf
from scipy import sparse
import sys
import getopt
import time

from load_cf_data import load_tr_te_data, load_train_data
from load_text_data import load_data
from evaluate import evaluate_all
from nb_vae import NegativeBinomialVAE
from nb_vae_b import NegativeBinomialVAEb


def main():
    collection_step = 10
    total_anneal_steps = 20000
    data_dir = './datasets'
    batch_size = 500
    batch_size_vad = 2000

    n_epochs = 800

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hd:b:m:s:t:a:n:',
                                   ['dataset=', 'is-data-binary=' 'model-id=', 'save-dir=',
                                    'is-training=', 'arch-id=', 'num-training-epochs'])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for o, a in opts:
        if o in ('-d', '--dataset'):
            dataset = str(a)
        elif o in ('-b', '--is-data-binary'):
            is_data_binary = int(a)
        elif o in ('-m', '--model-id'):
            model_id = int(a)
        elif o in ('-s', '--save-dir'):
            base_dir = a
        elif o in ('-t', '--training'):
            is_training = int(a)
        elif o in ('-a', '--arch-id'):
            arch_id = int(a)
        elif o in ('-n', '--num-training-epochs'):
            n_epochs = int(a)
        elif o == '-h':
            print('-d or --dataset: dataset name (has to be the same to the folder that saves the dataset), e.g., '
                  '20NG, ML-10M ...')
            print('-m or --model-id: 1. NBVAE, 2. NBVAE_dm, 3. NBVAE_b')
            print('-s or --save-dir: the folder that saves the model and the log files')
            print('-t or --is-training: 1. training phrase, 0. testing phrase')
            print('-a or --arch-id: network architecture, 1. [128], 2. [64-128], 3. [32-64-128], 4. [128-256], '
                  '5. [256-512], 6. [200-600]')

            print('-n or --num-training-epochs')

            exit()

        else:
            assert False, "unhandled option %s %s" % (o, a)

    data_dir = '%s/%s' % (data_dir, dataset)

    if is_data_binary == 1:  # cf data

        pro_dir = os.path.join(data_dir, 'pro_sg')

        unique_sid = list()
        with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        V = len(unique_sid)
        train_data = load_train_data(os.path.join(pro_dir, 'train.csv'), V)
        vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
                                                   os.path.join(pro_dir, 'validation_te.csv'), V)
    else:  # text

        train_data, vad_data_tr, vad_data_te = load_data('%s/data.mat' % data_dir)

        vad_data_te_mask = vad_data_te.todense() > 0

        V = train_data.shape[1]

    if arch_id == 1:
        arch = [128, V]
    elif arch_id == 2:
        arch = [64, 128, V]
    elif arch_id == 3:
        arch = [32, 64, 128, V]
    elif arch_id == 4:
        arch = [128, 256, V]
    elif arch_id == 5:
        arch = [256, 512, V]
    else:
        arch = [200, 600, V]

    N = train_data.shape[0]

    N_vad = vad_data_tr.shape[0]
    idxlist_vad = range(N_vad)

    if is_data_binary == 1:
        anneal_cap = 0.2
    else:
        anneal_cap = 1.0



    tf.reset_default_graph()

    if model_id == 1:
        vae = NegativeBinomialVAE(arch, lr=1e-3, random_seed=1)
    else:
        vae = NegativeBinomialVAEb(arch, lr=1e-3, random_seed=1)

    saver, train_op, h_r, h_p = vae.build_graph()

    arch_str = '%s' % ('-'.join([str(d) for d in arch[:-1]]))

    if model_id == 1:
        model_str = 'nbvae'
    else:
        model_str = 'nbvae_b'

    model_save_dir = '{}/saved_models/{}/{}/{}'.format(base_dir, dataset, model_str, arch_str)

    log_dir = '{}/logs/{}/{}/{}/'.format(base_dir, dataset, model_str, arch_str)

    if is_training:

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        nms = []
        nss = []
        rms = []
        rss = []

        pps_test = []

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            best_ndcg_vad = -np.inf

            update_count = 0.0

            for epoch in range(n_epochs):

                idxlist = np.random.permutation(N)

                start_time = time.time()

                for bnum, st_idx in enumerate(range(0, N, batch_size)):

                    end_idx = min(st_idx + batch_size, N)

                    X = train_data[idxlist[st_idx:end_idx]]
                    if sparse.isspmatrix(X):
                        X = X.toarray()
                    X = X.astype('float32')

                    if anneal_cap > 0:
                        anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                    else:
                        anneal = 1.0

                    feed_dict = {vae.input_ph: X,
                                 vae.keep_prob_ph: 0.5,
                                 vae.anneal_ph: anneal,
                                 vae.is_training_ph: 1}
                    sess.run(train_op, feed_dict=feed_dict)

                    sys.stdout.write('\r')
                    sys.stdout.write('epoch: %d, %d' % (epoch, bnum))
                    sys.stdout.flush()

                    update_count += 1

                end_time = time.time()
                print('\n epoch: %d in %s secs' % (epoch, end_time - start_time))

                if epoch == n_epochs - 1 and is_data_binary == 0:
                    saver.save(sess, '{}/model-last'.format(model_save_dir))

                if epoch % collection_step == 0:

                    if is_data_binary == 1:

                        ndcgs = None
                        recalls = None

                        for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):

                            end_idx = min(st_idx + batch_size_vad, N_vad)

                            X_vad_seen = vad_data_tr[idxlist_vad[st_idx:end_idx]]
                            X_vad_unseen = vad_data_te[idxlist_vad[st_idx:end_idx]]

                            if sparse.isspmatrix(X_vad_seen):
                                X_vad_seen = X_vad_seen.todense()
                            X_vad_seen = X_vad_seen.astype('float32')

                            r, p = sess.run((h_r, h_p), feed_dict={vae.input_ph: X_vad_seen})
                            pred_l = vae.get_predictive_rate(r, p, X_vad_seen)

                            pred_l[X_vad_seen > 0] = -np.inf

                            ns, rs = evaluate_all(pred_l, X_vad_unseen)
                            if ndcgs is not None:
                                ndcgs = np.append(ndcgs, ns, axis=1)
                                recalls = np.append(recalls, rs, axis=1)
                            else:
                                ndcgs = ns
                                recalls = rs

                        nm = np.mean(ndcgs, axis=1)
                        ns = np.std(ndcgs, axis=1) / ndcgs.shape[1]
                        rm = np.mean(recalls, axis=1)
                        rs = np.std(recalls, axis=1) / ndcgs.shape[1]
                        nms.append(nm)
                        nss.append(ns)
                        rms.append(rm)
                        rss.append(rs)

                        np.save('%s/nms' % (log_dir), nms)
                        np.save('%s/nss' % (log_dir), nss)
                        np.save('%s/rms' % (log_dir), rms)
                        np.save('%s/rss' % (log_dir), rss)

                        print('vad-ndcg@50: %f' % nm[-1])

                        if nm[-1] > best_ndcg_vad:
                            best_ndcg_vad = nm[-1]
                            saver.save(sess, '{}/model'.format(model_save_dir))
                    else:

                        log_prob_test = 0
                        word_count_test = 0

                        for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):

                            end_idx = min(st_idx + batch_size_vad, N_vad)

                            X_vad_seen = vad_data_tr[idxlist_vad[st_idx:end_idx]]

                            X_vad_unseen = vad_data_te[idxlist_vad[st_idx:end_idx]]
                            X_vad_unseen_mask = vad_data_te_mask[idxlist_vad[st_idx:end_idx]]

                            if sparse.isspmatrix(X_vad_seen):
                                X_vad_seen = X_vad_seen.todense()
                            X_vad_seen = X_vad_seen.astype('float32')

                            r, p = sess.run((h_r, h_p), feed_dict={vae.input_ph: X_vad_seen})

                            pred_l = vae.get_predictive_rate(r, p, X_vad_seen)

                            pred_l = pred_l / np.sum(pred_l, axis=1).reshape(pred_l.shape[0], 1)

                            log_prob = np.multiply(X_vad_unseen[X_vad_unseen_mask],
                                                   np.log(pred_l[X_vad_unseen_mask]))

                            log_prob_test += np.sum(log_prob)
                            word_count_test += np.sum(X_vad_unseen)

                        pp_test = log_prob_test / word_count_test
                        pp_test = np.exp(-pp_test)
                        pps_test.append(pp_test)

                        print('test-pp: %f' % pp_test)

                        np.save('%s/pps-test' % (log_dir), pps_test)


    else:

        if is_data_binary == 1:
            test_data_tr, test_data_te = load_tr_te_data(
                os.path.join(pro_dir, 'test_tr.csv'),
                os.path.join(pro_dir, 'test_te.csv'), V)
            print("saved model directory: %s" % model_save_dir)

            with tf.Session() as sess:
                saver.restore(sess, '{}/model'.format(model_save_dir))

                ndcgs = None
                recalls = None

                N_test = test_data_tr.shape[0]

                for bnum, st_idx in enumerate(range(0, N_test, batch_size_vad)):
                    end_idx = min(st_idx + batch_size_vad, N_vad)
                    X_test_seen = test_data_tr[idxlist_vad[st_idx:end_idx]]

                    X_test_unseen = test_data_te[idxlist_vad[st_idx:end_idx]]

                    if sparse.isspmatrix(X_test_seen):
                        X_test_seen = X_test_seen.todense()
                    X_test_seen = X_test_seen.astype('float32')

                    r, p = sess.run((h_r, h_p), feed_dict={vae.input_ph: X_test_seen})
                    pred_l = vae.get_predictive_rate(r, p, X_test_seen)

                    pred_l[X_test_seen > 0] = -np.inf

                    ns, rs = evaluate_all(pred_l, X_test_unseen)
                    if ndcgs is not None:
                        ndcgs = np.append(ndcgs, ns, axis=1)
                        recalls = np.append(recalls, rs, axis=1)

                    else:
                        ndcgs = ns
                        recalls = rs
                nm = np.mean(ndcgs, axis=1)
                ns = np.std(ndcgs, axis=1) / ndcgs.shape[1]
                rm = np.mean(recalls, axis=1)
                rs = np.std(recalls, axis=1) / ndcgs.shape[1]

                np.save('%s/nm-test' % (log_dir), nm)
                np.save('%s/ns-test' % (log_dir), ns)
                np.save('%s/rm-test' % (log_dir), rm)
                np.save('%s/rs-test' % (log_dir), rs)

                print('test NDCG@1: %f, @5: %f, @10: %f, @20: %f, @50: %f\n' % (nm[0], nm[1], nm[2], nm[3], nm[4]))
                print('test Recall@1: %f, @5: %f, @10: %f, @20: %f, @50: %f\n' % (rm[0], rm[1], rm[2], rm[3], rm[4]))


if __name__ == "__main__":
    main()
