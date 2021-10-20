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
from torchsummaryX import summary
os.environ['CUDA_LAUNCH_BLOCKING']='1'
torch.manual_seed(10)
np.random.seed(10)

device = torch.device('cuda:0')

#If you need to change the data, change the two lines below
data_name = 'pems-m'
node_num = 228                 # d4: 307, metr-la: 207, bay: 325, seoul_highway: 468

#Model params
input_time = 12
input_feature = 1
embedding_feature = 16
GCN_feature = 64
prediction_step = 12

#Hyper params
train_epoch = 300
batch_size = 32
learning_rate = 0.001
dropout_ratio = 0.5
early_stop_patient = 20

load_model = False

#Paths
log_file = 'log.txt'

datafile = './data/window_'+data_name+'.npz'
data_array, scaler = data_load(datafile, node_num)              ##########

utcnow = datetime.datetime.utcnow()
now = utcnow + datetime.timedelta(hours=9)

if not os.path.isdir('./out/'):                                                           
    os.mkdir('./out/')
out_path = './out/data_{}_ep_{}_bs_{}_lr_{}_dr_{}_er_{}_bn_ln'.format(data_name,train_epoch, batch_size, learning_rate, dropout_ratio, early_stop_patient) + now.strftime('_%y%m%d_%H%M%S')
if not os.path.isdir(out_path):                                                           
    os.mkdir(out_path)

print('Parameters: \n data: {}\n epoch: {}\n batch: {}\n lr_rate: {}\n dropout_ratio: {}\n patient: {}\n'.format(data_name,train_epoch, batch_size, learning_rate, dropout_ratio, early_stop_patient))

#Our Traffic prediction Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.node_embeddings = nn.Parameter(torch.FloatTensor(node_num * input_time, embedding_feature))
        self.GCN_weight = nn.Parameter(torch.FloatTensor(embedding_feature, input_feature, GCN_feature))
        self.GCN_bias = nn.Parameter(torch.FloatTensor(embedding_feature, GCN_feature))
        self.bn1 = nn.BatchNorm1d(node_num * input_time)
        self.bn2 = nn.BatchNorm1d(node_num)
        # self.ln1 = nn.LayerNorm([node_num, input_time, GCN_feature])
        self.ln1 = nn.LayerNorm([node_num*input_time, GCN_feature])
        self.ln2 = nn.LayerNorm([node_num*input_time, GCN_feature])
        # self.ln_train = nn.LayerNorm([int(data_array.shape[0]*0.7 % batch_size), node_num * input_time, GCN_feature])
        # self.ln_val = nn.LayerNorm([int(data_array.shape[0]*0.1 % batch_size), node_num * input_time, GCN_feature])
        # self.ln_test = nn.LayerNorm([int(data_array.shape[0]*0.2 % batch_size), node_num * input_time, GCN_feature])
        # self.ln1 = nn.LayerNorm([node_num * input_time, GCN_feature])
        # self.GCN_weight2 = nn.Parameter(torch.FloatTensor(embedding_feature, GCN_feature, GCN_feature))
        # self.GCN_bias2 = nn.Parameter(torch.FloatTensor(embedding_feature, GCN_feature))
        # self.gru = nn.GRU(input_size=GCN_feature, hidden_size=GCN_feature, batch_first=True, num_layers=1)
        # self.FC_weight = nn.Parameter(torch.FloatTensor(GCN_feature, input_time))
        # self.FC_bias = nn.Parameter(torch.FloatTensor(input_time))
        self.FC = nn.Linear((input_time*GCN_feature), int((GCN_feature*input_time)/3))
        self.FC3 = nn.Linear(int((GCN_feature*input_time)/3), 12)
        self.AttFC = nn.Linear(GCN_feature,GCN_feature)
        self.ReLU = nn.ReLU()
        # self.FC_weight2 = nn.Parameter(torch.FloatTensor(node_num, GCN_feature, 1))
        # self.FC_bias2 = nn.Parameter(torch.FloatTensor(node_num, 1))
        # self.query = nn.Linear(GCN_feature, GCN_feature, bias=False)
        # self.key = nn.Linear(GCN_feature, GCN_feature, bias=False)
        # self.value = nn.Linear(GCN_feature, GCN_feature, bias=False)
        self.att = nn.MultiheadAttention(GCN_feature, 2)
        self.telayer = nn.TransformerEncoderLayer(d_model=GCN_feature, nhead=2, batch_first=True)
        self.te = nn.TransformerEncoder(self.telayer, 1)
        nn.init.kaiming_normal_(self.node_embeddings)
        nn.init.kaiming_normal_(self.GCN_weight)
        nn.init.kaiming_normal_(self.GCN_bias)
        self.masking = torch.from_numpy(mask_(node_num, input_time))
        # nn.init.kaiming_normal_(self.GCN_weight2)
        # nn.init.kaiming_normal_(self.GCN_bias2)
        # nn.init.kaiming_normal_(self.FC_weight2)
        # nn.init.kaiming_normal_(self.FC_bias2)

    def forward(self, x):
        x = x.to(device)
        w = torch.FloatTensor(1).to(device)
        do = torch.nn.Dropout(p=dropout_ratio)
        gcn_weights = torch.einsum('ij,jkl->ikl', self.node_embeddings, self.GCN_weight)
        gcn_bias = self.node_embeddings.matmul(self.GCN_bias)
        x2 = torch.einsum('bij,ijk->bik', x, gcn_weights) + gcn_bias
        x2 = self.bn1(x2)
        x = do(F.relu(x2)) + x
        # x = torch.transpose(x, 0, 1)
        # z = torch.cat((x, self.node_embeddings.expand(x.shape[0], -1, -1)), dim=2)
        # z = torch.transpose(self.node_embeddings.expand(x.shape[0], -1, -1), 0, 1)
        # output, w = self.att(x, x, x, attn_mask = self.masking.to(device))
        output = self.te(x, mask=self.masking.to(device))
        # output = torch.transpose(output, 0, 1)
        # output = torch.matmul(w, x)
        # output = self.ln1(output)
        # output = F.relu(output)  # GCN
        # output = do(output)
        output = torch.reshape(output, (-1, input_time, node_num, GCN_feature))
        output = torch.transpose(output, 1, 2)
        output = torch.reshape(output, (-1, node_num, input_time*GCN_feature))

        output = self.FC(output)
        output = self.bn2(output)
        output = self.ReLU(output)
        # output = self.FC2(output)
        # output = self.ReLU(output)
        output = self.FC3(output)
        output = torch.transpose(output, 1, 2)
        output = torch.reshape(output, (-1, node_num * input_time, 1))
        return output, w


def mask_(node_num, input_time):
  masknp = np.empty((node_num*input_time, node_num*input_time))
  for i in range(input_time):
    tmp = np.empty((node_num, input_time*node_num))
    tmp[:, :(i+1)*node_num] = False
    tmp[:, (i+1)*node_num:] = True
    masknp[i*node_num:(i+1)*node_num, :] = tmp
  return masknp.astype('bool')


#Loss functions
def mape(output, label):
    return torch.mean(torch.abs(torch.div((label - output), label)))


def MAPE_torch(pred, true, mask_value=0.):
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
# net = nn.DataParallel(net)
net.to(device)
if load_model:
    # net = torch.load('./out/data_Seoul_cityroad_511_ep_300_bs_32_lr_0.001_dr_0.5_er_20_bn_ln_210825_201433/bestmodel', map_location={'cuda:0':'cuda:0', 'cuda:1':'cuda:1', 'cuda:2':'cuda:2', 'cuda:3':'cuda:3'})
    net = torch.load('./out/data_pems04_ep_300_bs_24_lr_0.001_dr_0.5_er_20_bn_ln_211015_165957/savedmodel_epoch_56')
    net.to(device)
    print(net)
criterion = mae
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

##Data loading
dataloader = DataLoader(data_array[:int(len(data_array) * 0.7)], batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(data_array[int(len(data_array) * 0.7):int(len(data_array) * 0.8)], batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(data_array[int(len(data_array) * 0.8):], batch_size=batch_size, shuffle=False)
train_log = open(os.path.join(out_path, log_file), 'w', newline='') #logfile
# print(summary(net, (node_num*input_time, 1), batch_size=batch_size)) 
#Model trainig Start
wait = 0
val_mae_min = np.inf
best_model = copy.deepcopy(net.state_dict())
train_maes, val_maes, test_maes = [], [], []
for epoch in range(0, train_epoch):
    
    if wait >= early_stop_patient:
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
        input_x = samples[:, :, 0]
        input_x = np.reshape(input_x, [-1, node_num * input_time, 1])
        labels = samples[:, :, 1]
        # labels = np.reshape(labels, [-1, node_num])
        labels = torch.from_numpy(np.trunc(scaler.inverse_transform(labels)))
        labels = np.reshape(labels, [-1, node_num * input_time])
        input_x, labels = input_x.to(device), labels.to(device)
        outputs, adj = net(input_x)
        outputs = torch.reshape(outputs, (-1, node_num * input_time))
        loss = criterion(outputs, labels)
        loss_sum += loss.item()
        loss.backward()
        mae_sum += np.nan_to_num(mae(outputs, labels.to(device)).item())
        rmse_sum += np.nan_to_num(rmse(outputs, labels.to(device)).item())
        mape_step = np.nan_to_num(MAPE_torch(outputs, labels).item())
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
            input_x = samples[:, :, 0]
            input_x = np.reshape(input_x, [-1, node_num * input_time, 1])
            labels = samples[:, :, 1]
            # labels = np.reshape(labels, [-1, node_num])
            labels = torch.from_numpy(np.trunc(scaler.inverse_transform(labels)))
            labels = np.reshape(labels, [-1, node_num * input_time])
            input_x, labels = input_x.to(device), labels.to(device)
            outputs, adj = net(input_x)
            outputs = torch.reshape(outputs, (-1, node_num * input_time))
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            mae_sum += np.nan_to_num(mae(outputs, labels.to(device)).item())
            rmse_sum += np.nan_to_num(rmse(outputs, labels.to(device)).item())
            mape_step = np.nan_to_num(MAPE_torch(outputs, labels).item())
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
                input_x = samples[:, :, 0]
                input_x = np.reshape(input_x, [-1, node_num * input_time, 1])
                labels = samples[:, :, 1]
                # labels = np.reshape(labels, [-1, node_num])
                labels = torch.from_numpy(np.trunc(scaler.inverse_transform(labels)))
                labels = np.reshape(labels, [-1, node_num * input_time])
                input_x, labels = input_x.to(device), labels.to(device)
                outputs, adj = net(input_x)
                outputs = torch.reshape(outputs, (-1, node_num * input_time))
                loss = criterion(outputs, labels)
                loss_sum += loss.item()
                mae_sum += np.nan_to_num(mae(outputs, labels.to(device)).item())
                rmse_sum += np.nan_to_num(rmse(outputs, labels.to(device)).item())
                mape_step = np.nan_to_num(MAPE_torch(outputs, labels).item())
                mape_sum += mape_step
                batch_num = batch_idx
            batch_num += 1
            
            test_mae = mae_sum / batch_num
            test_maes.append(test_mae)
            test_log_str = ' test: %.5f\t\t%.5f\t\t%.5f' % (test_mae, rmse_sum / batch_num, mape_sum / batch_num)
            print(test_log_str)
            train_log.write(test_log_str + '\n')
            # torch.save(net, os.path.join(out_path,'savedmodel_epoch_{}'.format(epoch)))            #####################################
            # df = pd.DataFrame(adj.cpu().detach().numpy())
            # df.to_csv(str(epoch)+'.csv', index=False)
            

cw = csv.writer(open('./'+data_name+'.csv', 'w', newline=''))
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
    # adj_mean = torch.zeros(input_time * node_num, input_time * node_num).to(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for batch_idx, samples in enumerate(test_dataloader):
        # print(samples.shape)
        # if samples.shape[0] < 1:
        #     continue
        optimizer.zero_grad()
        input_x = samples[:, :, 0]
        input_x = np.reshape(input_x, [-1, node_num * input_time, 1])
        labels = samples[:, :, 1]
        # labels = np.reshape(labels, [-1, node_num])
        labels = torch.from_numpy(np.trunc(scaler.inverse_transform(labels)))
        labels = np.reshape(labels, [-1, node_num * input_time])
        input_x, labels = input_x.to(device), labels.to(device)
        print(summary(net,input_x))
        outputs, adj = net(input_x)
        outputs = torch.reshape(outputs, (-1, node_num * input_time))
        # outputs = torch.reshape(outputs, (-1, input_time, node_num))[:, :prediction_step, :]
        
        # labels = torch.reshape(labels, (-1, input_time, node_num))[:, :prediction_step, :].to(device)
        
        # output_np = outputs.cpu().detach().numpy()
        # labels_np = labels.cpu().detach().numpy()
        # try:
        #     x_np = np.concatenate((x_np, output_np), axis=0)
        #     y_np = np.concatenate((y_np, labels_np), axis=0)
        # except:
        #     x_np = output_np
        #     y_np = labels_np
        # np.save('HY_pred.npy',output_np)
        # np.save('HY_real.npy', labels_np)
        # for l in range(0, len(output_np)):
        #     line = np.concatenate((output_np[l], labels_np[l]))
        #     cw.writerow(line)
 
        # adj_mean += torch.mean(adj, 0, True).squeeze()
        # outputs = torch.reshape(outputs, (-1, node_num * input_time))
        loss = criterion(outputs, labels)
        loss_sum += loss.item()
        mae_sum += np.nan_to_num(mae(outputs, labels.to(device)).item())
        rmse_sum += np.nan_to_num(rmse(outputs, labels.to(device)).item())
        mape_step = np.nan_to_num(MAPE_torch(outputs, labels).item())
        mape_sum += mape_step
        batch_num = batch_idx
    batch_num += 1
    # adj_mean = adj_mean / batch_idx
    # adj_mean = adj_mean.cpu().detach().numpy()
    # adj_0 = np.zeros((input_time, node_num))
    # linknum = 15
    # for i in range(0, 12):
    #   for j in range(0, node_num):
    #     adj_0[i,j] += np.sum(adj_mean[[linknum + k*node_num for k in range(0, 12)], i*node_num + j])
    # cw = csv.writer(open('./att_'+str(linknum)+'_metr.csv', 'w', newline=''))
    # for line in adj_0:
    #   cw.writerow(line)
    test_log_str = ' Test: %.5f\t\t%.5f\t\t%.5f' % (mae_sum / batch_num, rmse_sum / batch_num, mape_sum / batch_num)
    print(test_log_str)
    end.record()
    train_log.write(test_log_str+'\n')
    np.save('HY_pred', x_np)
    np.save('HY_real', y_np)
    print('Testinng time: ', start.elapsed_time(end))
#Logging top 3 val/test mae loss
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
