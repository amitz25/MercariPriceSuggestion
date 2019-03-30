#!/usr/bin/env python

from __future__ import print_function
import time
import sys
from optparse import OptionParser
from nn.common import *
import os
from nn.model import Model
from nn.logger import Logger
from nn.dataset import *
from nn.common import DBType
from nn.eval import Evaluator

parser = OptionParser()
parser.add_option('--config', type=str, help="configuration", default="config.yaml")

MIN_NAME_DF = 1000


def train_tf_idf(min_df, kn, raw_data):
    v = TfidfVectorizer(analyzer="word", min_df=min_df)
    v = v.fit(raw_data[kn])
    return v


def main(argv):
    (opts, args) = parser.parse_args(argv)
    config = ConfigParser(opts.config)

    if torch.cuda.is_available():
        gpu_ids = np.array(config.general.gpu_ids.split(' ')).astype(np.int)
        device = torch.device('cuda:{0}'.format(gpu_ids[0]))
    else:
        device = torch.device('cpu')

    # torch.cuda.set_device(device)

    raw_df = pd.read_csv(config.dataset.raw_path, sep="\t")
    name_vectorizer = train_tf_idf(MIN_NAME_DF, 'name', raw_df)

    train_loader, dataset = init_dataset(config, DBType.Train, name_vectorizer, raw_df)

    current_iteration_path = os.path.join(config.general.output_path, config.general.current_iteration_file_name)

    if os.path.isfile(current_iteration_path):
        start_epoch, epoch_iteration = np.loadtxt(current_iteration_path, delimiter=',', dtype=int)
        print('resuming from epoch %d at iteration %d' % (start_epoch, epoch_iteration))
    else:
        start_epoch, epoch_iteration = 0, 0

    tmp_start = epoch_iteration
    model = Model(config, dataset)

    # model = torch.nn.DataParallel(model)
    model.train()

    dataset_size = len(dataset)
    logger = Logger(config)
    current_step = start_epoch * dataset_size + epoch_iteration

    steps_counter = 0
    accumulated_loss = 0
    freq_loss = 0

    evaluator = Evaluator(DBType.Validation, config, name_vectorizer, raw_df)
    raw_df = None

    # if start_epoch % config.train.lr_update_freq == 0:
    #     model.update_learning_rate()

    # if len(gpu_ids) > 1:
    #    model = nn.DataParallel(model)

    model.to(device)
    freq_start_time = time.time()

    current_eval = last_eval = 99999999
    tmp_count = 0

    for epoch in range(start_epoch, config.train.num_epochs):
        epoch_start_time = time.time()

        if epoch != start_epoch:
            epoch_iteration = 0

        for i, data in enumerate(train_loader, start=epoch_iteration):
            if steps_counter % 500 == 0:
                print('{} / {}'.format(epoch_iteration, dataset_size))

            current_step += config.train.batch_size
            epoch_iteration += config.train.batch_size

            name = data['name'].to(device)
            cid = data['cid'].to(device)
            c_name = data['c_name'].to(device)
            b_name = data['b_name'].to(device)
            price = data['price'].to(device).unsqueeze(1)
            shipping = data['shipping'].to(device)
            desc = data['desc'].to(device)
            desc_len = data['desc_len'].to(device)

            loss = model(name, cid, c_name, b_name, shipping, desc, desc_len, price)
            loss = torch.mean(loss)

            model.optimizer.zero_grad()
            loss.backward()

            if config.general.clip_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            model.optimizer.step()

            accumulated_loss += loss.item()
            freq_loss += loss.item()

            if (steps_counter % config.general.print_logs_freq == 0) and steps_counter != 0:
                freq_loss = freq_loss / config.general.print_logs_freq

                print('freq_loss {}. time {}'.format(freq_loss, time.time() - freq_start_time))

                losses_dict = {'loss': loss.item(), 'freq_loss': freq_loss}
                logger.dump_current_errors(losses_dict, current_step)

                freq_loss = 0
                freq_start_time = time.time()

            if (steps_counter % config.general.save_checkpoint_freq == 0) and steps_counter != 0:
                print('========== saving model (epoch %d, total_steps %d) =========' % (epoch, current_step))
                model.save('latest')
                np.savetxt(current_iteration_path, (epoch, epoch_iteration), delimiter=',', fmt='%d')

            steps_counter += 1

        print('end of epoch %d / %d \t time taken: %d sec' %
              (epoch, config.train.num_epochs, time.time() - epoch_start_time))

        accumulated_loss = accumulated_loss / (i + 1 - tmp_start)
        tmp_start = 0

        print('accumulated loss {}'.format(accumulated_loss))

        losses_dict = {'accumulated_loss': accumulated_loss}
        logger.dump_current_errors(losses_dict, current_step)

        accumulated_loss = 0
        model.save('latest')
        model.save(str(epoch))

        np.savetxt(current_iteration_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        # if epoch % config.general.eval_epcohs_freq == 0:
        current_eval = evaluator.eval(model, max_iterations=config.train.max_eval_iterations)

        # if epoch % config.train.lr_update_freq == 0:
        if current_eval > last_eval:
            tmp_count += 1

            if tmp_count == 3:
                model.update_learning_rate()
                tmp_count = 0
                last_eval = current_eval
        else:
            tmp_count = 0
            last_eval = current_eval


if __name__ == '__main__':
    main(sys.argv)