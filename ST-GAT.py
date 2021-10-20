import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_preprocessing import data_load
import datetime
import time
import os
import csv
import copy
import argparse
import configparser
from torchsummaryX import summary
os.environ['CUDA_LAUNCH_BLOCKING']='1'
torch.manual_seed(10)
np.random.seed(10)

device = torch.device('cuda:0')

args = argparse.ArgumentParser(description='args')
args.add_argument('--mode', default='train', type=str)
args.add_argument('--conf', type=str)
args_string = args.parse_args()
print(args)
try:
    config = configparser.ConfigParser()
    config.read(args_string.conf)
except:
    print("Config is not exist!")
    print(args_string.conf)
    exit()

#If you need to change the data, change the two lines below
args.add_argument('--data', default=config['data']['dataset'], type=str)
args.add_argument('--seg_num', default=config['data']['seg_num'], type=int)                 # d4: 307, metr-la: 207, bay: 325
args.add_argument('--saved_model', type=str)

#Model params
args.add_argument('--input_time', default=config['data']['input_time'], type=int)
args.add_argument('--input_feature', default=config['data']['input_feature'], type=int)
args.add_argument('--con_feature', default=config['model']['context_feature'], type=int)
args.add_argument('--emb_feature', default=config['model']['embed_feature'], type=int)
args.add_argument('--prediction_step', default=config['data']['prediction_step'], type=int)

#Hyper params
args.add_argument('--train_epoch', default=config['train']['train_epoch'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--learning_rate', default=config['train']['learning_rate'], type=float)
args.add_argument('--dropout_ratio', default=config['train']['dropout_ratio'], type=float)
args.add_argument('--early_stop_patient', default=config['train']['early_stop_patient'], type=int)
args = args.parse_args()

if args.saved_model:
    load_model = True
else:
    load_model = False

if args.mode=='test':
    t_epoch = 0
else:
    t_epoch=args.train_epoch


#Paths
log_file = 'log.txt'

datafile = './data/window_'+args.data+'.npz'
data_array, scaler = data_load(datafile, args.seg_num, args.input_time, args.prediction_step)              ##########

utcnow = datetime.datetime.utcnow()
now = utcnow + datetime.timedelta(hours=9)

if not os.path.isdir('./out/'):                                                           
    os.mkdir('./out/')
out_path = './out/data_{}_ep_{}_bs_{}_lr_{}_dr_{}_er_{}_bn_ln'.format(args.data,args.train_epoch, args.batch_size, args.learning_rate, args.dropout_ratio, args.early_stop_patient) + now.strftime('_%y%m%d_%H%M%S')
if not os.path.isdir(out_path):                                                           
    os.mkdir(out_path)

print('Parameters: \n data: {}\n epoch: {}\n batch: {}\n lr_rate: {}\n args.dropout_ratio: {}\n patient: {}\n'.format(args.data,args.train_epoch, args.batch_size, args.learning_rate, args.dropout_ratio, args.early_stop_patient))

#Our Traffic prediction Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.node_cons = nn.Parameter(torch.FloatTensor(args.seg_num * args.input_time, args.con_feature))
        self.context_weight = nn.Parameter(torch.FloatTensor(args.con_feature, args.input_feature, args.emb_feature))
        self.context_bias = nn.Parameter(torch.FloatTensor(args.con_feature, args.emb_feature))
        self.bn1 = nn.BatchNorm1d(args.seg_num * args.input_time)
        self.bn2 = nn.BatchNorm1d(args.seg_num)
        self.FC = nn.Linear((args.input_time*args.emb_feature), int((args.emb_feature*args.input_time)/3))
        self.FC2 = nn.Linear(int((args.emb_feature*args.input_time)/3), args.prediction_step)
        self.AttFC = nn.Linear(args.emb_feature,args.emb_feature)
        self.ReLU = nn.ReLU()
        self.telayer = nn.TransformerEncoderLayer(d_model=args.emb_feature, nhead=2, batch_first=True)
        self.te = nn.TransformerEncoder(self.telayer, 1)
        nn.init.kaiming_normal_(self.node_cons)
        nn.init.kaiming_normal_(self.context_weight)
        nn.init.kaiming_normal_(self.context_bias)
        self.masking = torch.from_numpy(mask_(args.seg_num, args.input_time))

    def forward(self, x):
        x = x.to(device)
        do = torch.nn.Dropout(p=args.dropout_ratio)
        cwpl_weights = torch.einsum('ij,jkl->ikl', self.node_cons, self.context_weight)
        cwpl_bias = self.node_cons.matmul(self.context_bias)
        x = torch.einsum('bij,ijk->bik', x, cwpl_weights) + cwpl_bias
        x = self.bn1(x)
        output = self.te(x, mask=self.masking.to(device))
        output = torch.reshape(output, (-1, args.input_time, args.seg_num, args.emb_feature))
        output = torch.transpose(output, 1, 2)
        output = torch.reshape(output, (-1, args.seg_num, args.input_time * args.emb_feature))
        output = self.FC(output)
        output = self.bn2(output)
        output = self.ReLU(output)
        output = self.FC2(output)
        output = torch.transpose(output, 1, 2)
        output = torch.reshape(output, (-1, args.seg_num * args.prediction_step, 1))
        return output


def mask_(seg_num, input_time):
  masknp = np.empty((seg_num*input_time, seg_num*input_time))
  for i in range(input_time):
    tmp = np.empty((seg_num, input_time*seg_num))
    tmp[:, :(i+1)*seg_num] = False
    tmp[:, (i+1)*seg_num:] = True
    masknp[i*seg_num:(i+1)*seg_num, :] = tmp
  return masknp.astype('bool')


#Loss functions
def mape(pred, true, mask_value=0.):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)

    return torch.mean(torch.abs(torch.div((true - pred), true)))


def mae(pred, true, mask_value=0.):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))


def rmse(pred, true, mask_value=0.):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))
##

#Prepare training model
torch.cuda.empty_cache()
print(Net())
net = Net()
net.to(device)
if load_model:
    net = torch.load(args.saved_model)
    net.to(device)
    print(net)
criterion = mae
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

##Data loading
dataloader = DataLoader(data_array[:int(len(data_array) * 0.7)], batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(data_array[int(len(data_array) * 0.7):int(len(data_array) * 0.8)], batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(data_array[int(len(data_array) * 0.8):], batch_size=args.batch_size, shuffle=False)
train_log = open(os.path.join(out_path, log_file), 'w', newline='') #logfile

wait = 0
val_mae_min = np.inf
best_model = copy.deepcopy(net.state_dict())
train_maes, val_maes, test_maes = [], [], []

for epoch in range(0, t_epoch):

    if wait >= args.early_stop_patient:
        earlystop_log = 'Early stop at epoch: %04d' % (epoch)
        print(earlystop_log)
        train_log.write(earlystop_log + '\n')
        break
    
    mape_sum = 0
    loss_sum = 0
    mae_sum = 0
    rmse_sum = 0
    batch_num = 0
    cnt = 0
    net.train()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for batch_idx, samples in enumerate(dataloader):
        if batch_idx % 400 == 0:
            print(batch_idx, len(dataloader))
        optimizer.zero_grad()
        input_x = samples[:, :args.seg_num * args.input_time]
        input_x = np.reshape(input_x, [-1, args.seg_num * args.input_time, 1])
        labels = samples[:, args.seg_num * args.input_time:]
        labels = np.reshape(labels, [-1, args.seg_num * args.prediction_step])
        input_x, labels = input_x.to(device), labels.to(device)
        outputs = net(input_x)
        outputs = torch.reshape(outputs, (-1, args.seg_num * args.prediction_step))
        loss = criterion(outputs, labels)
        loss_sum += loss.item()
        loss.backward()
        mae_sum += np.nan_to_num(mae(outputs, labels.to(device)).item())
        rmse_sum += np.nan_to_num(rmse(outputs, labels.to(device)).item())
        mape_step = np.nan_to_num(mape(outputs, labels).item())
        mape_sum += mape_step
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
        batch_num = batch_idx
    batch_num += 1
    end.record()
    torch.cuda.synchronize()

    #Logging train, val(every) / test(every 5epoch) MAE loss
    ##Train_loss
    train_mae = mae_sum / batch_num
    train_maes.append(train_mae)
    train_log_str = ' train: %.5f\t\t%.5f\t\t%.5f' % (train_mae, rmse_sum / batch_num, mape_sum / batch_num)
    
    print('Epoch: ', epoch, ' / learning time: ', start.elapsed_time(end))
    print(train_log_str)
    train_log.write('Epoch: ' + str(epoch) + '\n')
    train_log.write(train_log_str + '\n')
    
    ##Val_loss
    with torch.no_grad():
        net.eval()
        loss_sum = 0
        mape_sum = 0
        rmse_sum = 0
        mae_sum = 0
        cnt = 0
        for batch_idx, samples in enumerate(val_dataloader):
            optimizer.zero_grad()
            input_x = samples[:, :args.seg_num * args.input_time]
            input_x = np.reshape(input_x, [-1, args.seg_num * args.input_time, 1])
            labels = samples[:, args.seg_num * args.input_time:]
            labels = np.reshape(labels, [-1, args.seg_num * args.prediction_step])
            input_x, labels = input_x.to(device), labels.to(device)
            outputs = net(input_x)
            outputs = torch.reshape(outputs, (-1, args.seg_num * args.prediction_step))
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            mae_sum += np.nan_to_num(mae(outputs, labels.to(device)).item())
            rmse_sum += np.nan_to_num(rmse(outputs, labels.to(device)).item())
            mape_step = np.nan_to_num(mape(outputs, labels).item())
            mape_sum += mape_step
            batch_num = batch_idx
        batch_num += 1
        
        val_mae = mae_sum / batch_num
        val_maes.append(val_mae)
        valid_log_str = ' Valid: %.5f\t\t%.5f\t\t%.5f' % (val_mae, rmse_sum / batch_num, mape_sum / batch_num)
        
        print(valid_log_str)
        train_log.write(valid_log_str + '\n')
        
        ##Model save
        torch.save(net, os.path.join(out_path,'savedmodel_epoch_{}'.format(epoch)))
        
        #Early Stopping
        if val_mae <= val_mae_min:
            log = ' Validation loss decrease (exist min: %.5f, new min: %.5f)' % (val_mae_min, val_mae)
            print(log)
            train_log.write(log + '\n')
            best_model = copy.deepcopy(net.state_dict())
            wait=0
            val_mae_min = val_mae
        else:
            wait += 1
    
    ##Train loss (5 epoch)
    if epoch % 5 == 0:
        with torch.no_grad():
            net.eval()
            loss_sum = 0
            mape_sum = 0
            rmse_sum = 0
            mae_sum = 0
            cnt = 0
            for batch_idx, samples in enumerate(test_dataloader):
                optimizer.zero_grad()
                input_x = samples[:, :args.seg_num * args.input_time]
                input_x = np.reshape(input_x, [-1, args.seg_num * args.input_time, 1])
                labels = samples[:, args.seg_num * args.input_time:]
                labels = np.reshape(labels, [-1, args.seg_num * args.prediction_step])
                input_x, labels = input_x.to(device), labels.to(device)
                outputs = net(input_x)
                outputs = torch.reshape(outputs, (-1, args.seg_num * args.prediction_step))

                loss = criterion(outputs, labels)
                loss_sum += loss.item()
                mae_sum += np.nan_to_num(mae(outputs, labels.to(device)).item())
                rmse_sum += np.nan_to_num(rmse(outputs, labels.to(device)).item())
                mape_step = np.nan_to_num(mape(outputs, labels).item())
                mape_sum += mape_step
                batch_num = batch_idx
            batch_num += 1
            
            test_mae = mae_sum / batch_num
            test_maes.append(test_mae)
            test_log_str = ' test: %.5f\t\t%.5f\t\t%.5f' % (test_mae, rmse_sum / batch_num, mape_sum / batch_num)
            print(test_log_str)
            train_log.write(test_log_str + '\n')
            

cw = csv.writer(open('./'+args.data+'.csv', 'w', newline=''))
#Logging last test mae loss
net.load_state_dict(best_model)
with torch.no_grad():
    net.eval()
    loss_sum = 0
    mape_sum = 0
    rmse_sum = 0
    mae_sum = 0
    cnt = 0
    x_np = np.array([])
    y_np = np.array([])
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for batch_idx, samples in enumerate(test_dataloader):
        optimizer.zero_grad()
        input_x = samples[:, :args.seg_num * args.input_time]
        input_x = np.reshape(input_x, [-1, args.seg_num * args.input_time, 1])
        labels = samples[:, args.seg_num * args.input_time:]
        labels = np.reshape(labels, [-1, args.seg_num * args.prediction_step])
        input_x, labels = input_x.to(device), labels.to(device)
        outputs = net(input_x)
        outputs = torch.reshape(outputs, (-1, args.seg_num * args.prediction_step))

        loss = criterion(outputs, labels)
        loss_sum += loss.item()
        mae_sum += np.nan_to_num(mae(outputs, labels.to(device)).item())
        rmse_sum += np.nan_to_num(rmse(outputs, labels.to(device)).item())
        mape_step = np.nan_to_num(mape(outputs, labels).item())
        mape_sum += mape_step
        batch_num = batch_idx
    batch_num += 1
    test_log_str = ' Test: %.5f\t\t%.5f\t\t%.5f' % (mae_sum / batch_num, rmse_sum / batch_num, mape_sum / batch_num)
    print(test_log_str)
    end.record()
    train_log.write(test_log_str+'\n')
    np.save('HY_pred', x_np)
    np.save('HY_real', y_np)
    print('Testinng time: ', start.elapsed_time(end))
    torch.save(net, os.path.join(out_path,'best_model'))
    print('Best model saved')

#Logging top 3 val/test mae loss
if t_epoch > 3:
    val_top3 = sorted(zip(val_maes, range(len(val_maes))))[:3]
    test_top3 = sorted(zip(test_maes, [i * 5 for i in range(len(test_maes))]))[:3]
    val_top3_log = \
        'Validation top 3\n 1st: %.5f / %depoch\n 2st: %.5f / %depoch\n 3st: %.5f / %depoch' % (val_top3[0][0], val_top3[0][1], val_top3[1][0], val_top3[1][1], val_top3[2][0], val_top3[2][1])
    test_top3_log = \
        'Test top 3\n 1st: %.5f / %depoch\n 2st: %.5f / %depoch\n 3st: %.5f / %depoch' % (test_top3[0][0], test_top3[0][1], test_top3[1][0], test_top3[1][1], test_top3[2][0], test_top3[2][1])

    print(val_top3_log)
    print(test_top3_log)
    train_log.write(val_top3_log + '\n')
    train_log.write(test_top3_log + '\n')
