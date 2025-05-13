from import_library import *
from dataloader import *
from loss import *
from utils import *
from model import *

import warnings
warnings.filterwarnings(action='ignore')

import argparse
import configparser

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"   
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_num_threads(4) 

######## get arguments #########
parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str, help='Config path')
parser.add_argument('--frame_length')
parser.add_argument('--model_name')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config_file)

data_path = config['Base']['data_path']
save_path = config['Base']['save_path']
gpu = config['Base']['gpu']
seed = int(config['Base']['seed'])
device=f'cuda:{gpu}'

train_db = config['Data']['train_db']
target_fs = int(config['Data']['target_fs'])
max_duration = int(config['Data']['max_duration'])

model_name = args.model_name
S = int(config['Model']['S'])
C = int(config['Model']['C'])
frame_length = args.frame_length
activation = config['Model']['activation']
res_droprate = float(config['Model']['res_droprate'])
if frame_length!='False':
    frame_length = float(frame_length)
    max_frame_num = max_duration // frame_length
    max_frame_num = int(max_frame_num)
else:
    print('No using RPAM')
    max_frame_num = False

lambda_conf = float(config['Loss']['lambda_conf'])
lambda_class = float(config['Loss']['lambda_class'])
lambda_coord = float(config['Loss']['lambda_coord'])
lambda_noobj = float(config['Loss']['lambda_noobj'])

BATCH = int(config['SYS']['BATCH'])
EPOCHS = int(config['SYS']['EPOCHS'])
MOMENTUM = float(config['SYS']['MOMENTUM'])
LR = float(config['SYS']['LR'])
WD = float(config['SYS']['WD'])
optim_name = config['SYS']['optim_name']
early_stop_patience = int(config['SYS']['early_stop'])
lr_init = float(config['SYS']['lr_init'])
warmup_epochs = int(config['SYS']['warmup_epochs'])
iou_threshold = float(config['SYS']['iou_threshold'])

######## Set seed ########
random.seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
    
######### Data load #########
path = f'Datasets Folder Path/{train_db}'
dataprocessor = DataProcessor(target_fs)

wav_files = dataprocessor.get_data(path, 'train')
wav_files.extend(dataprocessor.get_data(path, 'test'))
try:
    os.listdir(f'{data_path}/{train_db}/Final')
except FileNotFoundError:
    if train_db == 'hf_lung':
        df = dataprocessor.load_data(path, 'train', wav_files, max_duration, S=S, C=C)
        df = pd.concat([df, dataprocessor.load_data(path, 'train', wav_files, max_duration, S=S, C=C)], ignore_index=True)
    elif train_db == 'icbhi':
        df = dataprocessor.load_data_icbhi(path, 'train', wav_files, max_duration, S=S, C=C)
        df = pd.concat([df, dataprocessor.load_data_icbhi(path, 'train', wav_files, max_duration, S=S, C=C)], ignore_index=True)

    subjects = np.asarray(list(set(df['PID'])))

    train_subjects, valid_subjects = train_test_split(subjects, test_size=0.2, shuffle=True, random_state=seed)

    os.makedirs(f'{data_path}/{train_db}/Final', exist_ok=True)
    with open(f'{data_path}/{train_db}/Final/dev_subjects.pkl', 'wb') as f:
        pickle.dump(train_subjects, f)
    with open(f'{data_path}/{train_db}/Final/val_subjects.pkl', 'wb') as f:
        pickle.dump(valid_subjects, f)
        

FOLDER= f'{save_path}/{train_db}/sum/S{S}_C{C}/{activation}_droprate{res_droprate}/lambda_conf{lambda_conf}_class{lambda_class}_coord{lambda_coord}_noobj{lambda_noobj}/BATCH{BATCH}_LR{LR}_WD{WD}/{model_name}/frame{frame_length}'
for fold_num in range(5):
    FOLDER_PATH = f'{FOLDER}/Final'

    if os.path.exists(FOLDER_PATH):
        continue
    else:
        start_epoch = 0
    os.makedirs(FOLDER_PATH, exist_ok=True)

    dev_subjects = pd.read_pickle(f'{data_path}/{train_db}/Final/dev_subjects.pkl')
    val_subjects = pd.read_pickle(f'{data_path}/{train_db}/Final/val_subjects.pkl')
    
    dev_wav_files = []
    val_wav_files = []
    for file in wav_files:
        fname = file.split(".wav")[0]
        if train_db == 'hf_lung':
            if 'steth_' in fname:
                pid = fname.split("steth_")[-1]
            elif 'trunc_' in fname:
                pid = fname.split("trunc_")[-1].split("-L")[0]
        elif train_db == 'icbhi':
            pid = fname.split("_")[0]

        if pid in dev_subjects:
            dev_wav_files.append(file)
        elif pid in val_subjects:
            val_wav_files.append(file)

    if train_db == 'hf_lung':
        dev_df = dataprocessor.load_data(path, 'train', dev_wav_files, max_duration, S, C)
        val_df = dataprocessor.load_data(path, 'train', val_wav_files, max_duration, S, C)
    elif train_db == 'icbhi':
        dev_df = dataprocessor.load_data_icbhi(path, 'train', dev_wav_files, max_duration, S, C)
        val_df = dataprocessor.load_data_icbhi(path, 'train', val_wav_files, max_duration, S, C)

    dev_dataset = CycleDataset(dev_df, max_duration, target_fs, S, C)
    val_dataset = CycleDataset(val_df, max_duration, target_fs, S, C)

    dev_loader = DataLoader(dev_dataset, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    model = Detector(model_name=model_name, activation=activation, max_frame_num=max_frame_num, S=S, C=C, res_droprate=res_droprate, device=device).to(device)
    lossfn = ODLoss(C=C, lambda_conf=lambda_conf, lambda_class=lambda_class, lambda_coord=lambda_coord, lambda_noobj=lambda_noobj).to(device)

    if optim_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WD, momentum=MOMENTUM)
    elif optim_name == 'SGD_nesterov':
        optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WD, momentum=MOMENTUM, nesterov=True)
    elif optim_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    elif optim_name == 'Adamw':
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    elif optim_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=LR, weight_decay=WD, momentum=MOMENTUM)
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)

    total_loss_per_epoch = []
    coord_loss_per_epoch = []
    class_loss_per_epoch = []
    conf_loss_per_epoch = []

    val_total_loss_per_epoch = []
    val_coord_loss_per_epoch = []
    val_class_loss_per_epoch = []
    val_conf_loss_per_epoch = []

    BESTLOSS = np.inf
    early_stop_count = 0

    warmup_epochs = warmup_epochs
    warmup_lr_initial = lr_init
    warmup_lr_final = LR
    for epoch in range(start_epoch, EPOCHS):
        if epoch <= warmup_epochs:
            warmup_lr = warmup_lr_initial + (epoch / warmup_epochs) * (warmup_lr_final - warmup_lr_initial)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        lr = optimizer.param_groups[0]["lr"]

        total_loss_batch = 0
        coord_loss_batch = 0
        class_loss_batch = 0
        conf_loss_batch = 0
        model.train()
        for x, y in tqdm(dev_loader):
            x = x.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()

            raw_pred_range, embeddings, pred_tsm = model(x)
            total_loss, coord_loss, class_loss, conf_loss = lossfn(raw_pred_range, y)

            total_loss.backward()
            optimizer.step()

            total_loss_batch += total_loss.item()
            coord_loss_batch += coord_loss.item()
            class_loss_batch += class_loss.item()
            conf_loss_batch += conf_loss.item()

        total_loss_per_epoch.append(total_loss_batch / len(dev_loader))
        coord_loss_per_epoch.append(coord_loss_batch / len(dev_loader))
        class_loss_per_epoch.append(class_loss_batch / len(dev_loader))
        conf_loss_per_epoch.append(conf_loss_batch / len(dev_loader))
        
        val_total_loss_batch = 0
        val_coord_loss_batch = 0
        val_class_loss_batch = 0
        val_conf_loss_batch = 0
        model.eval()
        for x, y in tqdm(val_loader):
            with torch.no_grad():
                x = x.float().to(device)
                y = y.float().to(device)

                raw_pred_range, embeddings, pred_tsm = model(x)
                total_loss, coord_loss, class_loss, conf_loss = lossfn(raw_pred_range, y)

                val_total_loss_batch += total_loss.item()
                val_coord_loss_batch += coord_loss.item()
                val_class_loss_batch += class_loss.item()
                val_conf_loss_batch += conf_loss.item()

        val_total_loss_per_epoch.append(val_total_loss_batch / len(val_loader))
        val_coord_loss_per_epoch.append(val_coord_loss_batch / len(val_loader))
        val_class_loss_per_epoch.append(val_class_loss_batch / len(val_loader))
        val_conf_loss_per_epoch.append(val_conf_loss_batch / len(val_loader))
        
        with open(FOLDER_PATH+'/Loss.txt', 'a') as f:
            f.write(f'epoch:{epoch+1} Train loss: {np.round(total_loss_per_epoch[-1], 4)} Validation loss: {np.round(val_total_loss_per_epoch[-1], 4)} LR: {lr}\n')

        lr_scheduler.step(val_total_loss_per_epoch[-1])

        if val_total_loss_per_epoch[-1] < BESTLOSS:
            early_stop_count = 0
            BESTLOSS = val_total_loss_per_epoch[-1]
            checkpoint = {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epochs": epoch+1,
                            "lr_scheduler" : lr_scheduler.state_dict(),
                        }
            torch.save(checkpoint, FOLDER_PATH+'/'+'bestmodel')
        else:
            early_stop_count += 1

        if early_stop_count == early_stop_patience:
            with open(FOLDER_PATH+'/Loss.txt', 'a') as f:
                f.write(f'early_stopping | epochs {epoch}')
            break

        df_loss_per_epoch = pd.DataFrame({'Total':total_loss_per_epoch, 'Coordinate':coord_loss_per_epoch, 'Classification':class_loss_per_epoch, 'Confidence':conf_loss_per_epoch})
        df_val_loss_per_epoch = pd.DataFrame({'Total':val_total_loss_per_epoch, 'Coordinate':val_coord_loss_per_epoch, 'Classification':val_class_loss_per_epoch, 'Confidence':val_conf_loss_per_epoch})
        df_loss_per_epoch.to_csv(FOLDER_PATH+'/'+'Train_loss.csv', index=False)
        df_val_loss_per_epoch.to_csv(FOLDER_PATH+'/'+'Validation_loss.csv', index=False)

        plt.rcParams.update({'font.size': 15, 'axes.titlesize': 20, 'axes.labelsize':15})
        row_nums = 2
        col_nums = 2
        titles = list(df_loss_per_epoch.columns)
        df_loss_per_epoch.plot(subplots=True, 
                layout=(row_nums, col_nums), 
                title=titles,
                sharex=True,
                sharey=False,
                xlabel='Epochs', 
                ylabel='Values',
                figsize=(40, 15))
        plt.savefig(FOLDER_PATH+'/'+'Train_performance.png')
        plt.cla()

        df_val_loss_per_epoch.plot(subplots=True, 
                layout=(row_nums, col_nums), 
                title=titles,
                sharex=True,
                sharey=False,
                xlabel='Epochs', 
                ylabel='Values',
                figsize=(40, 15))
        plt.savefig(FOLDER_PATH+'/'+'Validation_performance.png')
        plt.cla()