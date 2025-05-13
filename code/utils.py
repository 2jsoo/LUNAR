from import_library import *
torch.set_num_threads(16)
import copy

class DataProcessor:
    def __init__(self, target_fs):
        self.target_fs = target_fs

    def compute_mel_spectrogram(self, y, n_fft=40, hop_length=20):
        S = librosa.feature.melspectrogram(y=y, sr=self.target_fs, n_fft=n_fft, hop_length=hop_length)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB

    def compute_mfcc(self, y, n_fft=40, hop_length=20): # nftt : 25ms의 크기를 기본으로 하고 있으며 16000Hz인 음성에서는 400에 해당하는 값 ==> 4000Hz & 25ms : 100 ==> 4000Hz & 10ms : 40 /// hop_length : 10ms 기본 16000Hz에서는 160에 해당 ==> 4000Hz & 10ms : 40 ==> 4000Hz & 5ms : 20
        mfcc = librosa.feature.mfcc(y=y, sr=self.target_fs, n_fft=n_fft, hop_length=hop_length)
        return mfcc

    def get_data(self, path, train_test):
        train_db = path.split("/")[-1]
        if train_db == 'hf_lung':
            file_list = os.listdir(f"{path}/{train_test}")
            wav_files = [f for f in file_list if f.endswith(".wav")]
        elif train_db == 'icbhi':
            file_list = os.listdir(path)
            wav_files = [f for f in file_list if f.endswith(".wav")]
        elif train_db == 'snuch_lung':
            if train_test == 'test':
                file_list = os.listdir(f'{path}/{train_test}/wav_files')
                label = pd.read_csv(f'{path}/{train_test}/label.csv')

                wav_files = os.listdir(f'{path}/{train_test}/wav_files')
                wav_files = [f for f in wav_files if f in list(label['file_path'])] # label 있는 데이터만 추출
            else:
                label = pd.read_pickle(f'{path}/{train_test}/snuh_{train_test}_label_img.pkl')
                label_valid = pd.read_pickle(f'{path}/valid/snuh_valid_label_img.pkl')
                label.update(label_valid)

                wav_files = list(label.keys()) # train 필요없음 - 이미 pkl로 저장되어있음
        return wav_files

    def load_data(self, path, train_test, wav_files, max_duration, S, C):        
        df = []
        for f in wav_files:
            fname = f.split(".wav")[0]
    
            if 'steth_' in fname:
                pid = fname.split("steth_")[-1]
            elif 'trunc_' in fname:
                pid = fname.split("trunc_")[-1].split("-L")[0]
    
            ### sound data
            try:
                fs, f_wav = wavfile.read(f"{path}/{train_test}/{f}")
            except FileNotFoundError: # best model 만들때는 train, test 구분 없이 다 합쳐서 했음
                if train_test == 'train':
                    fs, f_wav = wavfile.read(f"{path}/test/{f}")
                elif train_test == 'test':
                    fs, f_wav = wavfile.read(f"{path}/train/{f}")
            try:
                try:
                    ### label data
                    f_txt = pd.read_csv(f"{path}/{train_test}/{fname}_label.txt", sep=" ", header=None)
                except FileNotFoundError:
                    if train_test == 'train':
                        f_txt = pd.read_csv(f"{path}/test/{fname}_label.txt", sep=" ", header=None)
                    elif train_test == 'test':
                        f_txt = pd.read_csv(f"{path}/test/{fname}_label.txt", sep=" ", header=None)
                ### inhalation이랑 exhalation이 같은 위치로 라벨링된 데이터가 있음 -> figure&table 1페이지 참고 ==> 이런 경우 해당 I, E모두 제거 ** mody4에서 추가
                f_txt = f_txt[(f_txt[0].isin(['I', 'E']))].drop_duplicates(subset=[1, 2], keep=False)
                
                # inhale만 남기기
                f_txt = f_txt[f_txt[0] == 'I'].drop_duplicates()
                # 시간 string -> 초로 변경
                f_txt[1] = [pd.to_timedelta(val).total_seconds() for val in f_txt[1]]
                f_txt[2] = [pd.to_timedelta(val).total_seconds() for val in f_txt[2]]    
            except pd.errors.EmptyDataError: # file 중에서, 빈 텍스트인 경우 있음 ex) train folder : trunc_2019-07-17-10-07-09-L1_1
                continue
            
            cycle_num = len(f_txt)
            if (cycle_num <= 1):
                continue

            f_wav = self.denoise(f_wav, fs) # high pass filtering

            # analytic_signal = hilbert(f_wav) # inhale만 극대화해주기
            # envelope = np.abs(analytic_signal)
            # f_wav *= envelope

            
            resampled_f_wav, _ = processing.resample_sig(f_wav, fs=fs, fs_target=self.target_fs)
            norm_f_wav = self.normalize_data(resampled_f_wav)

            total_length = int(max_duration*self.target_fs)

            ### norm_f_wav가 total_length보다 작으면 zero padding하기
            f_wav_length = norm_f_wav.shape[0]
            if f_wav_length < total_length:
                norm_f_wav = self.is_pad(norm_f_wav, total_length)

            ### Label 연속 구간 찾아서 [inhale여부(0이면 inhale), confidence(inahel이면 1, 아니면 0), 중점, width] 만들기
            inhale_range = [(0, (f_txt.iloc[i][1] + f_txt.iloc[i][2])/2*self.target_fs/total_length, (f_txt.iloc[i][2]-f_txt.iloc[i][1])*self.target_fs/total_length) for i in range(len(f_txt))] 
            original_range = [(0, f_txt.iloc[i][1]*self.target_fs/total_length, f_txt.iloc[i][2]*self.target_fs/total_length) for i in range(len(f_txt))]

            label_matrix = torch.zeros((S, C + 3)) # class0 class_exist box1(x_mid, w)
            for box in inhale_range:
                class_label, x_mid, width = box 
                class_label = int(class_label)

                x_mid = x_mid
                width = width # 전체(최대) segment length에서 현재 데이터의 width가 차지하고 있는 비율
    
                # i : object의 중점이 몇 번 째 grid cell인지에 위치해 있는지
                # x_cell : 해당 grid cell 기준으로 box의 중점이 어디에 위치해 있는지 ex grid cell 중점에 box 중점 있으면 x_cell = 0.5
                i = int(S * x_mid) # ex 중점이 50임 -> 1번째 grid cell => 10 * 50/1000  = 0.5 int(0.5) = 0 ///// 중점이 150임 -> 2번째 grid cell => 10 * 150/1000 = 1.5 int(1.5) = 1 /// 중점이 450 -> 5번째 grid cell => 10 * 450/1000 = int(4.5) = 4
                x_cell = S * x_mid - i
    
                width_cell = width * S  # width가 지금 image_width로 나눠져 있으니까 width * self.S 하면 현재 grid cell에 대한 너비 비율 나옴 .ex) width = 1 -> 1 * self.S(=10) = 10 => grid cell 하나의 너비보다 10배 큼
                
                if label_matrix[i, C] == 0: # i번째 grid cell에 class_exist가 0으로 되어있으면
                    
                    label_matrix[i, C] = 1 # 1로 바꿔서 i번째 grid cell에 해당하는 box로 그림 그려야 한다는거 알려주기
    
                    # Box coordinates
                    box_coordinates = torch.tensor([x_cell, width_cell])
    
                    label_matrix[i, C+1:C+3] = box_coordinates
    
                    # Set one hot encoding for class_label
                    label_matrix[i, class_label] = 1 # class_label이 1이면(=inhale) 0번째에 1로 표시해서 이 box는 inhale이라는거 알려주기

            df.append({'PID':pid, 'raw':resampled_f_wav, 'data':torch.tensor(norm_f_wav).unsqueeze(0), 'cycle_num':cycle_num, 'raw_fs':fs, 'inhale_range':label_matrix, 'original_range':original_range})

        df = pd.DataFrame(df)
        return df

    def load_data_icbhi(self, path, train_test, wav_files, max_duration, S, C):    
        train_test_info = pd.read_csv(f'{path}/icbhi_train_test.txt', sep="\t", header=None)
        train_test_info.columns = ["fname", "train_test"]
        train_test_files = list(train_test_info[train_test_info["train_test"]==train_test]['fname'])

        wav_files = [f for f in wav_files if f.split(".wav")[0] in train_test_files] # train 혹은 test에 해당하는 파일만 남겨서 처리하기

        df = []
        for f in wav_files:
            fname = f.split(".wav")[0]
            pid = fname.split("_")[0]
    
            ### sound data
            fs, f_wav = wavfile.read(f"{path}/{f}")
    
            try:
                ### label data
                f_txt = pd.read_csv(f"{path}/{fname}.txt", sep="\t", header=None)
                f_txt.columns = ["start", "end", "crackle", "wheeze"]

                f_txt = f_txt.drop_duplicates(subset=["start", "end"], keep=False)
                
            except pd.errors.EmptyDataError: # file 중에서, 빈 텍스트인 경우 있음 ex) train folder : trunc_2019-07-17-10-07-09-L1_1
                continue
            
            cycle_num = len(f_txt)
            if (cycle_num <= 1):
                continue

            f_wav = self.denoise(f_wav, fs) # high pass filtering

            resampled_f_wav, _ = processing.resample_sig(f_wav, fs=fs, fs_target=self.target_fs)
            norm_f_wav = self.normalize_data(resampled_f_wav)

            total_length = int(max_duration*self.target_fs)

            ### norm_f_wav가 total_length보다 작으면 zero padding하고 크면 split하기 ==> 길이가 total_length의 절반보다 작으면 버리기
            f_wav_length = norm_f_wav.shape[0]
            
            for seg_num, seg_start_idx in enumerate(range(0, f_wav_length-total_length, total_length)):
                # seg_start_idx : point 단위
                f_wav_seg = norm_f_wav[seg_start_idx: seg_start_idx+total_length]
                seg_start = seg_num * max_duration
                seg_end = (seg_num + 1) * max_duration

                f_wav_seg_length = f_wav_seg.shape[0]
                if f_wav_seg_length < total_length:
                    if f_wav_seg_length < total_length / 2:
                        continue
                    f_wav_seg = self.is_pad(f_wav_seg, total_length)

                f_txt_seg = f_txt.copy()
                # 1. 일단 cycle이 seg 시작 전에 끝나면 해당 cycle 제거
                f_txt_seg = f_txt_seg[f_txt_seg['end'] >= seg_start]
                # 2. cycle 시작시점이 seg_start 이전부터 시작해서 seg_end 이전에 끝나는 경우 cycle의 시작 시점을 0으로 설정
                if len(f_txt_seg[(f_txt_seg['start'] <= seg_start)&(f_txt_seg['end'] <= seg_end)]) > 1: # 이러면 오류임
                    print(ValueError('ErrorErrorError'))
                elif len(f_txt_seg[(f_txt_seg['start'] <= seg_start)&(f_txt_seg['end'] <= seg_end)]) == 1:
                    f_txt_seg[(f_txt_seg['start'] <= seg_start)&(f_txt_seg['end'] <= seg_end)].iloc[0]['start'] = 0
                else:
                    pass
                # 3. cycle 종료시점이 seg_end 이후에 끝날 수 있음. 하지만 시작시점은 seg_end이전
                if len(f_txt_seg[(f_txt_seg['start'] <= seg_end)&(f_txt_seg['end'] >= seg_end)]) > 1: # 이러면 오류임
                    print(ValueError('ErrorErrorError'))
                elif len(f_txt_seg[(f_txt_seg['start'] <= seg_end)&(f_txt_seg['end'] >= seg_end)]) == 1:
                    f_txt_seg[(f_txt_seg['start'] <= seg_end)&(f_txt_seg['end'] >= seg_end)].iloc[0]['end'] = 15
                else:
                    pass
                # 4. 0 ~ 15 사이에 있는 데이터만 남기기
                f_txt_seg['start'] -= seg_start
                f_txt_seg['end'] -= seg_start
                f_txt_seg = f_txt_seg[(f_txt_seg['start'] >= 0)&(f_txt_seg['end'] <= 15)]

                cycle_num = len(f_txt_seg)
                if (cycle_num <= 1):
                    continue

                ### Label 연속 구간 찾아서 [inhale여부(0이면 inhale), confidence(inahel이면 1, 아니면 0), 중점, width] 만들기
                inhale_range = [(0, (f_txt_seg.iloc[i]['start'] + f_txt_seg.iloc[i]['end'])/2*self.target_fs/total_length, (f_txt_seg.iloc[i]['end']-f_txt_seg.iloc[i]['start'])*self.target_fs/total_length) for i in range(len(f_txt_seg))] 
                original_range = [(0, f_txt_seg.iloc[i]['start']*self.target_fs/total_length, f_txt_seg.iloc[i]['end']*self.target_fs/total_length) for i in range(len(f_txt_seg))]

                label_matrix = torch.zeros((S, C + 3)) # class0 class_exist box1(x_mid, w)
                for box in inhale_range:
                    class_label, x_mid, width = box 
                    class_label = int(class_label)

                    x_mid = x_mid
                    width = width # 전체(최대) segment length에서 현재 데이터의 width가 차지하고 있는 비율
        
                    # i : object의 중점이 몇 번 째 grid cell인지에 위치해 있는지
                    # x_cell : 해당 grid cell 기준으로 box의 중점이 어디에 위치해 있는지 ex grid cell 중점에 box 중점 있으면 x_cell = 0.5
                    i = int(S * x_mid) # ex 중점이 50임 -> 1번째 grid cell => 10 * 50/1000  = 0.5 int(0.5) = 0 ///// 중점이 150임 -> 2번째 grid cell => 10 * 150/1000 = 1.5 int(1.5) = 1 /// 중점이 450 -> 5번째 grid cell => 10 * 450/1000 = int(4.5) = 4
                    x_cell = S * x_mid - i
        
                    width_cell = width * S  # width가 지금 image_width로 나눠져 있으니까 width * self.S 하면 현재 grid cell에 대한 너비 비율 나옴 .ex) width = 1 -> 1 * self.S(=10) = 10 => grid cell 하나의 너비보다 10배 큼
                    
                    if label_matrix[i, C] == 0: # i번째 grid cell에 class_exist가 0으로 되어있으면
                        
                        label_matrix[i, C] = 1 # 1로 바꿔서 i번째 grid cell에 해당하는 box로 그림 그려야 한다는거 알려주기
        
                        # Box coordinates
                        box_coordinates = torch.tensor([x_cell, width_cell])
        
                        label_matrix[i, C+1:C+3] = box_coordinates
        
                        # Set one hot encoding for class_label
                        label_matrix[i, class_label] = 1 # class_label이 1이면(=inhale) 0번째에 1로 표시해서 이 box는 inhale이라는거 알려주기

                df.append({'PID':pid, 'seg_num':seg_num+1, 'seg_start':seg_start, 'seg_end':seg_end, 'raw':resampled_f_wav[seg_start_idx: seg_start_idx+total_length], 'data':torch.tensor(f_wav_seg).unsqueeze(0), 'cycle_num':cycle_num, 'raw_fs':fs, 'inhale_range':label_matrix, 'original_range':original_range})

        df = pd.DataFrame(df)
        return df

    def load_data_snuch(self, path, train_test, wav_files, max_duration, S, C):        
        df = []

        if train_test == 'train':
            data = pd.read_pickle(f'{path}/{train_test}/snuh_{train_test}_data_raw.pkl')
            label = pd.read_pickle(f'{path}/{train_test}/snuh_{train_test}_label_img.pkl')
            data_valid = pd.read_pickle(f'{path}/valid/snuh_valid_data_raw.pkl')
            label_valid = pd.read_pickle(f'{path}/valid/snuh_valid_label_img.pkl')
            data.update(data_valid)
            label.update(label_valid)
        
            for i, k in enumerate(wav_files):
                fname = k
                pid = fname.split("_")[2]
                seg_num = fname.split("_")[3]

                fs = 4000 # 이미 4000으로 맞춰져있음
                f_wav = data[k]

                f_txt = pd.DataFrame(label[k]) # class(=inhale), inhale중점, width고정
                f_txt['cycle'] = [0]*len(f_txt)
                f_txt['cycle_start_time'] = f_txt[1] - (f_txt[2]/2)
                f_txt['cycle_start_time'] *=  60000 # 이미 60000으로 normalization되어있음

                cycle_num = len(f_txt)
                if (cycle_num <= 1):
                    continue

                f_wav = self.denoise(f_wav, fs) # high pass filtering

                resampled_f_wav, _ = processing.resample_sig(f_wav, fs=fs, fs_target=self.target_fs)
                norm_f_wav = self.normalize_data(resampled_f_wav)

                total_length = int(max_duration*self.target_fs)

                ### norm_f_wav가 total_length보다 작으면 zero padding하기
                f_wav_length = norm_f_wav.shape[0]
                if f_wav_length < total_length:
                    norm_f_wav = self.is_pad(norm_f_wav, total_length)

                ### Label 연속 구간 찾아서 [inhale여부(0이면 inhale), confidence(inahel이면 1, 아니면 0), 중점, width] 만들기
                inhale_range = []
                for i in range(cycle_num):
                    start_ = f_txt.iloc[i]['cycle_start_time']
                    end_ = start_ + (1*self.target_fs) # width 1로 고정하기

                    if end_ > total_length:
                        end_ = total_length
                    
                    mid_ = (start_ + end_) / 2
                    width_ = end_ - start_
                    
                    inhale_range.append([0, mid_/total_length, width_/total_length])

                label_matrix = torch.zeros((S, C + 3)) # class0 class_exist box1(x_mid, w)
                for box in inhale_range:
                    class_label, x_mid, width = box 
                    class_label = int(class_label)

                    x_mid = x_mid
                    width = width # 전체(최대) segment length에서 현재 데이터의 width가 차지하고 있는 비율
        
                    # i : object의 중점이 몇 번 째 grid cell인지에 위치해 있는지
                    # x_cell : 해당 grid cell 기준으로 box의 중점이 어디에 위치해 있는지 ex grid cell 중점에 box 중점 있으면 x_cell = 0.5
                    i = int(S * x_mid) # ex 중점이 50임 -> 1번째 grid cell => 10 * 50/1000  = 0.5 int(0.5) = 0 ///// 중점이 150임 -> 2번째 grid cell => 10 * 150/1000 = 1.5 int(1.5) = 1 /// 중점이 450 -> 5번째 grid cell => 10 * 450/1000 = int(4.5) = 4
                    x_cell = S * x_mid - i
        
                    width_cell = width * S  # width가 지금 image_width로 나눠져 있으니까 width * self.S 하면 현재 grid cell에 대한 너비 비율 나옴 .ex) width = 1 -> 1 * self.S(=10) = 10 => grid cell 하나의 너비보다 10배 큼
                    
                    if label_matrix[i, C] == 0: # i번째 grid cell에 class_exist가 0으로 되어있으면
                        
                        label_matrix[i, C] = 1 # 1로 바꿔서 i번째 grid cell에 해당하는 box로 그림 그려야 한다는거 알려주기
        
                        # Box coordinates
                        box_coordinates = torch.tensor([x_cell, width_cell])
        
                        label_matrix[i, C+1:C+3] = box_coordinates
        
                        # Set one hot encoding for class_label
                        label_matrix[i, class_label] = 1 # class_label이 1이면(=inhale) 0번째에 1로 표시해서 이 box는 inhale이라는거 알려주기

                df.append({'PID':pid, 'raw':resampled_f_wav, 'data':torch.tensor(norm_f_wav).unsqueeze(0), 'cycle_num':cycle_num, 'raw_fs':fs, 'inhale_range':label_matrix, 'original_range':inhale_range})
        else: # test
            label = pd.read_csv(f'{path}/{train_test}/label.csv')

            for idx, f in enumerate(tqdm(wav_files)):
                fname = f.split(".wav")[0]
                pid = fname.split("_")[2]

                ### sound data
                fs, f_wav = wavfile.read(f"{path}/{train_test}/wav_files/{f}")

                f_txt = label[label['file_path'] == f] # 초로 되어있음

                cycle_num = len(f_txt)
                if (cycle_num <= 1):
                    continue

                f_wav = self.denoise(f_wav, fs) # high pass filtering

                resampled_f_wav, _ = processing.resample_sig(f_wav, fs=fs, fs_target=self.target_fs)
                norm_f_wav = self.normalize_data(resampled_f_wav)

                total_length = int(max_duration*self.target_fs)

                ### norm_f_wav가 total_length보다 작으면 zero padding하기
                f_wav_length = norm_f_wav.shape[0]
                for seg_num, seg_start_idx in enumerate(range(0, f_wav_length-total_length, total_length)):
                    # seg_start_idx : point 단위
                    f_wav_seg = norm_f_wav[seg_start_idx: seg_start_idx+total_length]
                    seg_start = seg_num * max_duration
                    seg_end = (seg_num + 1) * max_duration

                    f_wav_seg_length = f_wav_seg.shape[0]
                    if f_wav_seg_length < total_length:
                        if f_wav_seg_length < total_length / 2:
                            continue
                        f_wav_seg = self.is_pad(f_wav_seg, total_length)

                    f_txt_seg = f_txt.copy()
                    # 1. 일단 cycle이 seg 시작 전에 끝나면 해당 cycle 제거
                    f_txt_seg = f_txt_seg[f_txt_seg['cycle_end_time'] >= seg_start]
                    # 2. cycle 시작시점이 seg_start 이전부터 시작해서 seg_end 이전에 끝나는 경우 cycle의 시작 시점을 0으로 설정
                    if len(f_txt_seg[(f_txt_seg['cycle_start_time'] <= seg_start)&(f_txt_seg['cycle_end_time'] <= seg_end)]) > 1: # 이러면 오류임
                        print(ValueError('ErrorErrorError'))
                    elif len(f_txt_seg[(f_txt_seg['cycle_start_time'] <= seg_start)&(f_txt_seg['cycle_end_time'] <= seg_end)]) == 1:
                        f_txt_seg[(f_txt_seg['cycle_start_time'] <= seg_start)&(f_txt_seg['cycle_end_time'] <= seg_end)].iloc[0]['cycle_start_time'] = 0
                    else:
                        pass
                    # 3. cycle 종료시점이 seg_end 이후에 끝날 수 있음. 하지만 시작시점은 seg_end이전
                    if len(f_txt_seg[(f_txt_seg['cycle_start_time'] <= seg_end)&(f_txt_seg['cycle_end_time'] >= seg_end)]) > 1: # 이러면 오류임
                        print(ValueError('ErrorErrorError'))
                    elif len(f_txt_seg[(f_txt_seg['cycle_start_time'] <= seg_end)&(f_txt_seg['cycle_end_time'] >= seg_end)]) == 1:
                        f_txt_seg[(f_txt_seg['cycle_start_time'] <= seg_end)&(f_txt_seg['cycle_end_time'] >= seg_end)].iloc[0]['cycle_end_time'] = 15
                    else:
                        pass
                    # 4. 0 ~ 15 사이에 있는 데이터만 남기기
                    f_txt_seg['cycle_start_time'] -= seg_start
                    f_txt_seg['cycle_start_time'] -= seg_start
                    f_txt_seg = f_txt_seg[(f_txt_seg['cycle_start_time'] >= 0)&(f_txt_seg['cycle_end_time'] <= 15)]

                    cycle_num = len(f_txt_seg)
                    if (cycle_num <= 1):
                        continue

                    ### Label 연속 구간 찾아서 [inhale여부(0이면 inhale), confidence(inahel이면 1, 아니면 0), 중점, width] 만들기
                    inhale_range = [(0, (f_txt_seg.iloc[i]['cycle_start_time'] + f_txt_seg.iloc[i]['cycle_end_time'])/2*self.target_fs/total_length, (f_txt_seg.iloc[i]['cycle_end_time']-f_txt_seg.iloc[i]['cycle_start_time'])*self.target_fs/total_length) for i in range(len(f_txt_seg))] 
                    original_range = [(0, f_txt_seg.iloc[i]['cycle_start_time']*self.target_fs/total_length, f_txt_seg.iloc[i]['cycle_end_time']*self.target_fs/total_length) for i in range(len(f_txt_seg))]

                    label_matrix = torch.zeros((S, C + 3)) # class0 class_exist box1(x_mid, w)
                    for box in inhale_range:
                        class_label, x_mid, width = box 
                        class_label = int(class_label)

                        x_mid = x_mid
                        width = width # 전체(최대) segment length에서 현재 데이터의 width가 차지하고 있는 비율
            
                        # i : object의 중점이 몇 번 째 grid cell인지에 위치해 있는지
                        # x_cell : 해당 grid cell 기준으로 box의 중점이 어디에 위치해 있는지 ex grid cell 중점에 box 중점 있으면 x_cell = 0.5
                        i = int(S * x_mid) # ex 중점이 50임 -> 1번째 grid cell => 10 * 50/1000  = 0.5 int(0.5) = 0 ///// 중점이 150임 -> 2번째 grid cell => 10 * 150/1000 = 1.5 int(1.5) = 1 /// 중점이 450 -> 5번째 grid cell => 10 * 450/1000 = int(4.5) = 4
                        x_cell = S * x_mid - i
            
                        width_cell = width * S  # width가 지금 image_width로 나눠져 있으니까 width * self.S 하면 현재 grid cell에 대한 너비 비율 나옴 .ex) width = 1 -> 1 * self.S(=10) = 10 => grid cell 하나의 너비보다 10배 큼
                        
                        if label_matrix[i, C] == 0: # i번째 grid cell에 class_exist가 0으로 되어있으면
                            
                            label_matrix[i, C] = 1 # 1로 바꿔서 i번째 grid cell에 해당하는 box로 그림 그려야 한다는거 알려주기
            
                            # Box coordinates
                            box_coordinates = torch.tensor([x_cell, width_cell])
            
                            label_matrix[i, C+1:C+3] = box_coordinates
            
                            # Set one hot encoding for class_label
                            label_matrix[i, class_label] = 1 # class_label이 1이면(=inhale) 0번째에 1로 표시해서 이 box는 inhale이라는거 알려주기

                    df.append({'PID':pid, 'seg_num':seg_num+1, 'seg_start':seg_start, 'seg_end':seg_end, 'raw':resampled_f_wav[seg_start_idx: seg_start_idx+total_length], 'data':torch.tensor(f_wav_seg).unsqueeze(0), 'cycle_num':cycle_num, 'raw_fs':fs, 'inhale_range':label_matrix, 'original_range':original_range})

        df = pd.DataFrame(df)
        return df

    def is_pad(self, tens, total_len):
        seq_len = tens.shape[0]
        if seq_len < total_len:
            padd_len = total_len - seq_len
            tens = np.concatenate([tens, np.array([0]*padd_len)], axis=-1)
        return tens
        
    def normalize_data(self, data):
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data.reshape(-1, 1)).squeeze()
        return normalized_data
        
    def denoise(self, data, fs):
        # high-pass filtering
        sos = signal.butter(8, 60, 'highpass', fs=fs, output='sos') # (60 ~ 2100인데 우리는 Fs=4000이니까 2000보다 밑으로해야함 그니까 그냥 60으로)reference : american journal of respiratory and critical care medicine(IF over 23) 2000
        data = signal.sosfilt(sos, data)

        return data

def cal_iou(pred_box, target_box):
    """
    pred_box  : (#, 2) -> 2 : x_mid, width
    """
    box1_xmin = pred_box[..., 0:1] - pred_box[..., 1:2] / 2
    box2_xmin = target_box[..., 0:1] - target_box[..., 1:2] / 2

    box1_xmax = pred_box[..., 0:1] + pred_box[..., 1:2] / 2
    box2_xmax = target_box[..., 0:1] + target_box[..., 1:2] / 2

    xmin = torch.max(box1_xmin, box2_xmin)
    xmax = torch.min(box1_xmax, box2_xmax)

    intersect_length = (xmax - xmin).clamp(0) # xmax - xmin이 음수라는 것은, 두 선이 겹치지 않음을 의미함. -> clamp(0)을 통해 0으로 만들어줌
    
    union_length = (box2_xmax - box2_xmin) + (box1_xmax - box1_xmin) - intersect_length

    iou = intersect_length / (union_length + 1e-6) # 1e-6 하는 이유는 division by 0 막기 위함

    return iou

def cal_giou(pred_box, target_box):
    box1_xmin = pred_box[..., 0:1] - pred_box[..., 1:2] / 2
    box2_xmin = target_box[..., 0:1] - target_box[..., 1:2] / 2

    box1_xmax = pred_box[..., 0:1] + pred_box[..., 1:2] / 2
    box2_xmax = target_box[..., 0:1] + target_box[..., 1:2] / 2

    xmin = torch.max(box1_xmin, box2_xmin)
    xmax = torch.min(box1_xmax, box2_xmax)
    
    intersect_length = (xmax - xmin).clamp(0) # xmax - xmin이 음수라는 것은, 두 선이 겹치지 않음을 의미함. -> clamp(0)을 통해 0으로 만들어줌
    union_length = (box2_xmax - box2_xmin) + (box1_xmax - box1_xmin) - intersect_length
    
    xmin = torch.min(box1_xmin, box2_xmin)
    xmax = torch.max(box1_xmax, box2_xmax)
    enclosure_length = (xmax-xmin).clamp(0)

    iou = cal_iou(pred_box, target_box)
    giou = iou - (enclosure_length - union_length)/(enclosure_length + 1e-6) # https://github.com/CoinCheung/pytorch-loss/blob/master/generalized_iou_loss.py

    return giou

def cal_diou(pred_box, target_box):
    """
    pred_box  : (#, 2) -> 2 : x_mid, width
    """
    iou = cal_iou(pred_box, target_box) # 얼마나 겹치는지

    # euclidean_distance = torch.square(pred_box[..., 0:1] - target_box[..., 0:1]) # 중점 간 거리
    euclidean_distance = torch.abs(pred_box[..., 0:1] - target_box[..., 0:1]) # 중점 간 거리

    box1_xmin = pred_box[..., 0:1] - pred_box[..., 1:2] / 2
    box2_xmin = target_box[..., 0:1] - target_box[..., 1:2] / 2
    box1_xmax = pred_box[..., 0:1] + pred_box[..., 1:2] / 2
    box2_xmax = target_box[..., 0:1] + target_box[..., 1:2] / 2
    xmin = torch.min(box1_xmin, box2_xmin)
    xmax = torch.max(box1_xmax, box2_xmax)
    # c = torch.square(xmax - xmin) # Diagonal length of the smallest enclosing box covering the two boxes -> 우리는 1d니까 diagonal할 때 y축 무시하고 그냥 양끝간 거리만 고려
    c = torch.abs(xmax - xmin) # Diagonal length of the smallest enclosing box covering the two boxes -> 우리는 1d니까 diagonal할 때 y축 무시하고 그냥 양끝간 거리만 고려

    penalty_term = euclidean_distance / (c + 1e-6) # 중심간 거리

    diou = iou - penalty_term # diou loss = 1 - iou + penaly_term

    return diou

def cal_ciou(pred_box, target_box):
    """
    pred_box  : (#, 2) -> 2 : x_mid, width
    """
    iou = cal_iou(pred_box, target_box) # 얼마나 겹치는지

    euclidean_distance = torch.square(pred_box[..., 0:1] - target_box[..., 0:1]) # 중점 간 거리
    box1_xmin = pred_box[..., 0:1] - pred_box[..., 1:2] / 2
    box2_xmin = target_box[..., 0:1] - target_box[..., 1:2] / 2
    box1_xmax = pred_box[..., 0:1] + pred_box[..., 1:2] / 2
    box2_xmax = target_box[..., 0:1] + target_box[..., 1:2] / 2
    xmin = torch.min(box1_xmin, box2_xmin)
    xmax = torch.max(box1_xmax, box2_xmax)
    c = torch.square(xmax - xmin) # Diagonal length of the smallest enclosing box covering the two boxes -> 우리는 1d니까 diagonal할 때 y축 무시하고 그냥 양끝간 거리만 고려
    penalty_term = euclidean_distance / (c + 1e-6) # 중심간 거리

    v = (4 / math.pi) * (math.atan(target_box[..., 1:2]) - math.atan(pred_box[..., 1:2])) ** 2
    a = v / ((1 - iou) + v)

    ciou = iou - penalty_term - a*v
    
    return ciou

# NMS
def nms(data_boxes, iou_threshold=0.5, confidence_threshold=0.5, diou=False):
    # data_boxes = [class, confidence_score, xmid, width]
    data_boxes = [sub_box for sub_box in data_boxes if sub_box[1] > confidence_threshold] # confidence threshold보다 큰 confidence score 갖는 box만 남기기

    data_boxes = sorted(data_boxes, key = lambda x: x[1], reverse=True) # confidence score 큰 순으로 정렬(내림차순)

    data_boxes_after_nms = []
    while data_boxes: # data_boxes에 남아있는 box 없을 때까지
        cur_box = data_boxes.pop(0) # 현재 남아있는 box들 중 가장 confidence score 큰 box 꺼내서
        data_boxes_after_nms.append(cur_box) # data_boxes_after_nms에 넣고

        if diou:
            data_boxes = [sub_box for sub_box in data_boxes
                        if (cal_diou(torch.tensor(sub_box[2:]), torch.tensor(cur_box[2:])) < iou_threshold)
                        or (sub_box[0] != cur_box[0])] # 남아있는 box들과 iou 구해서 iou_threhsold보다 적으면 그대로 두고(이유 : iou 값이 iou_threshold보다 작다는 것은 두 box가 그만큼 적게 겹친다는 의미니까 서로 다른 물체를 보고 있다고 판단함), 또는 각자가 예측한 clas label이 다르면 그대로 두기
        else:
            data_boxes = [sub_box for sub_box in data_boxes
                            if (cal_iou(torch.tensor(sub_box[2:]), torch.tensor(cur_box[2:])) < iou_threshold)
                            or (sub_box[0] != cur_box[0])] # 남아있는 box들과 iou 구해서 iou_threhsold보다 적으면 그대로 두고(이유 : iou 값이 iou_threshold보다 작다는 것은 두 box가 그만큼 적게 겹친다는 의미니까 서로 다른 물체를 보고 있다고 판단함), 또는 각자가 예측한 clas label이 다르면 그대로 두기

    return data_boxes_after_nms

def cal_performance(object_predictions, object_true, original_label, S, C, nms_iou_threshold, confidence_threshold, iou_threshold, diou, total_length, fs, error_threshold=1, savepath=False):
    pred_boxes, target_boxes = get_bboxes(object_predictions, object_true, S=S, C=C, iou_threshold=nms_iou_threshold, confidence_threshold=confidence_threshold, diou=diou)
    
    total_test_map, mean_precision, mean_recall, mean_f1, total_iou, correct, already_detect, small_iou_values, check_data_idxs = mean_average_precision(pred_boxes, target_boxes, iou_threshold=iou_threshold, num_classes=C, savepath=savepath)

    new_total_precision, new_total_recall, new_total_f1, new_total_map, _, _, _, _, _, _ = new_metric_test(original_label, pred_boxes, error_threshold, num_classes=C, total_length=total_length, fs=fs)

    return total_test_map, mean_precision, mean_recall, mean_f1, total_iou, new_total_precision, new_total_recall, new_total_f1, new_total_map

# MAP -> github : https://github.com/motokimura/yolo_v1_pytorch/blob/master/evaluate.py
def compute_average_precision(recall, precision):
    """ Compute AP for one class.
    Args:
        recall: (numpy array) recall values of precision-recall curve.
        precision: (numpy array) precision values of precision-recall curve.
    Returns:
        (float) average precision (AP) for the class.
    """
    # AP (AUC of precision-recall curve) computation using all points interpolation.
    # For mAP computation, you can find a great explaination below.
    # https://github.com/rafaelpadilla/Object-Detection-Metrics

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i -1], precision[i])

    ap = 0.0 # average precision (AUC of the precision-recall curve).
    for i in range(precision.size - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]

    return ap

def mean_average_precision(preds, targets, num_classes=1, iou_threshold=0.5, savepath=''):
    """ Compute mAP metric.
    Args:
        pred_boxes (list): list of lists containing all bboxes with each bboxes : [train_idx, class_prediction, prob_score, x_mid, width]
        threshold: (float) threshold for IoU to separate TP from FP.
    Returns:
        (list of float) list of average precision (AP) for each class.
    """
    aps = [] # list of average precisions (APs) for each class.
    precisions = []
    recalls = []
    f1s = []
    ious = []
    correct, already_detect, small_iou_values, check_data_idxs = 0, [], [], []

    fig, ax = plt.subplots(figsize=(7, 8))
    for class_lbl in range(num_classes):
        detections = [] # all predicted objects for this class.
        ground_truths = []
        
        for detection in preds:
            if detection[1] == class_lbl:
                detections.append(detection)
        for true_box in targets:
            if true_box[1] == class_lbl:
                ground_truths.append(true_box)

        if len(detections) == 0:
            ap = 0.0 # if no box detected, assigne 0 for AP of this class.
            print('---class {} AP {}---'.format('Inhale', ap))
            aps.append(ap)
            break

        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        eps = torch.finfo(torch.float64).eps

        iou_small = 0
        # already_detect = 0
        correct = 0
        small_iou_values = []
        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ] # 현재 image(=data)에 해당하는 true값만 가져옴

            num_gts = len(ground_truth_img)
            if num_gts == 0: # segment에 라벨 없으면 통과
                continue

            best_iou = 0

            for idx, box_gt in enumerate(ground_truth_img):
                iou_with_gt = cal_iou(torch.tensor(detection[3:]), torch.tensor(box_gt[3:]))

                if iou_with_gt > best_iou:
                    best_iou = iou_with_gt
                    best_gt_idx = idx

            if (best_iou >= iou_threshold):
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                    correct += 1
                    ious.append(best_iou.item()) # 정답에 대해서 얼만큼 가깝게 예측하는지
                else:
                    # already_detect += 1
                    already_detect.append(detection[0])
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                iou_small += 1
                FP[detection_idx] = 1
                if best_iou != 0:
                    ious.append(best_iou.item())
                    small_iou_values.append((detection[0], best_iou.item(), ground_truth_img[idx][-1])) # file 번호, iou 값, 그 때의 target length
                else:
                    ious.append(best_iou)
                    # small_iou_values.append(best_iou)
                    check_data_idxs.append(detection)              

        # Compute AP from `tp` and `fp`.
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        
        precision = TP_cumsum / np.maximum(TP_cumsum + FP_cumsum, eps)
        recall = TP_cumsum / float(total_true_bboxes)

        precision_ = (torch.sum(TP)/np.maximum((torch.sum(TP) + torch.sum(FP)), eps)).item()
        recall_ = (torch.sum(TP)/float(total_true_bboxes)).item()
        f1_ = 2 * (precision_ * recall_) / (precision_ + recall_ + eps)

        ap = compute_average_precision(recall, precision)
        if savepath:
            print('---class {} AP {}---'.format('Inhale', ap))
            print('---class {} Precision {}---'.format('Inhale', precision_))
            print('---class {} Recall {}---'.format('Inhale', recall_))
            print('---class {} F1 {}---'.format('Inhale', f1_))
        aps.append(ap)
        precisions.append(precision_)
        recalls.append(recall_)
        f1s.append(f1_)

        if savepath:
            display = PrecisionRecallDisplay(
                recall=recall,
                precision=precision,
                average_precision=np.trapz(precision, recall),
            )
            display.plot(ax=ax, name=f"Precision-recall for class label {class_lbl} AP {np.round(ap, 3)} Precision {np.round(precision_, 3)} Recall {np.round(recall_, 3)} F1 {np.round(f1_, 3)}")

            with open(f'{savepath}.txt', 'a') as f:
                f.write(f'class label {class_lbl} AP {ap} Precision {precision_} Recall {recall_} F1 {f1_}\n')
                f.write(f'class label {class_lbl} TP {max(TP_cumsum)} FP {max(FP_cumsum)} total_true_bboxes {total_true_bboxes}\n')

    if savepath:
        ax.set_xlim((-0.05, 1.05))
        ax.set_ylim((-0.05, 1.05))
        fig.savefig(f'{savepath}.png')
        plt.clf()
        plt.cla()

    # Compute mAP by averaging APs for all classes.
    if savepath:
        print('---mAP {}---'.format(np.mean(aps)))
        print('---mPrecision {}---'.format(np.mean(precisions)))
        print('---mRecall {}---'.format(np.mean(recalls)))
        print('---mF1 {}---'.format(np.mean(f1s)))
        print('---mIoU {}---'.format(np.mean(ious)))

    return np.mean(aps), np.mean(precisions), np.mean(recalls), np.mean(f1s), np.mean(ious), correct, already_detect, small_iou_values, check_data_idxs # , check_data_idxs

def get_bboxes(predictions, true, S, C, iou_threshold, confidence_threshold, diou):
    all_pred_boxes = []
    all_true_boxes = []

    train_idx = 0
    
    true_bboxes = cellboxes_to_boxes(true, S=S, C=C)
    bboxes = cellboxes_to_boxes(predictions, S=S, C=C)
    
    bn = len(predictions)
    for idx in range(bn):
        nms_boxes = nms(
            bboxes[idx],
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            diou=diou,
        )

        for nms_box in nms_boxes:
            all_pred_boxes.append([train_idx] + nms_box)

        for box in true_bboxes[idx]:
            # many will get converted to 0 pred
            if box[1] > confidence_threshold:
                all_true_boxes.append([train_idx] + box)

        train_idx += 1
        
    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions, S, C=1):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    bboxes = predictions[..., C+1:C+3]
    cell_indices = torch.arange(S).repeat(batch_size, 1).unsqueeze(-1)
    x_mid = 1 / S * (bboxes[..., :1] + cell_indices)
    width = 1 / S * bboxes[..., 1:2]
    converted_bboxes = torch.cat((x_mid, width), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    confidence = F.sigmoid(predictions[..., C:C+1])
    converted_preds = torch.cat((predicted_class, confidence, converted_bboxes), dim=-1)
    return converted_preds

def cellboxes_to_boxes(out, S, C=1):
    converted_pred = convert_cellboxes(out, S, C=C) #.reshape(out.shape[0], S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

# ### new metric
# - inhale start point에 대해서만 평가
# - 실제 start point를 중심으로 error_threshold(ex. 0.5s) 내에 예측한 start point 들어오면 정답
def new_metric_test(true_raw, pred_boxes, error_threshold, num_classes=1, total_length=15, fs=4000):
    average_precisions = []
    total_precision = []
    total_recall = []
    total_f1 = []

    # fig, ax = plt.subplots(figsize=(7, 8))
    for c in range(num_classes): # inhale에 대해서만 평가
        detections = []
        ground_truths = []


        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                # #### Inhalation이 0.17 ~ 1.64이하인 데이터 제거하기
                # if (detection[-1]*(total_length*fs) < 0.17*fs) or (detection[-1]*(total_length*fs) > 1.64*fs):
                #     pass
                # else:
                    detections.append(detection)

        for idx, (sub_true_raw) in enumerate(true_raw): # 각 데이터에 존재하는 모든 박스들 => true_raw는 dataset으로 변환한 값이 아니라 실제 .pkl값
            for sub_sub_true in sub_true_raw:
                if sub_sub_true[0] == c: # 해당 데이터의 라벨이 c랑 같음
                    raw_true_box = [idx]
                    raw_true_box.extend(sub_sub_true) # true_raw : class, x_mid, width
                    ground_truths.append(raw_true_box)
        
    
        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)


        epsilon = 1e-8

        iou_small = 0
        already_detect = 0
        correct = 0
        small_iou_values = []
        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ] # 현재 image(=data)에 해당하는 true값만 가져옴

            num_gts = len(ground_truth_img)
            if num_gts == 0: # segment에 라벨 없으면 통과
                continue

            best_error = np.inf

            for idx, gt in enumerate(ground_truth_img):
                gt_start_point = gt[2]

                pred_mid = detection[3]
                pred_width = detection[4]
                pred_start_point = pred_mid - (pred_width / 2)


                error_abs = np.abs(gt_start_point - pred_start_point)

                if error_abs < best_error:
                    best_error = error_abs
                    best_gt_idx = idx

            # if (best_error*(total_length*fs) <= (error_threshold/2*fs)): # mody3
            if (best_error*(total_length) <= (error_threshold/2*fs)): ####  ** mody4에서 추가
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                    correct += 1
                else:
                    already_detect += 1
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                iou_small += 1
                small_iou_values.append(best_error)
                FP[detection_idx] = 1
                # print(best_error, gt_start_point, pred_start_point)

        precision_ = (torch.sum(TP)/(torch.sum(TP)+torch.sum(FP))).item()
        recall_ = (torch.sum(TP)/(total_true_bboxes)).item()
        f1_ = np.round(2 * (precision_ * recall_) / (precision_ + recall_ + epsilon), 3)

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

        total_precision.append(precision_)
        total_recall.append(recall_)
        total_f1.append(f1_)

    total_map = sum(average_precisions) / (len(average_precisions) + 1e-6)
    
    total_precision = sum(total_precision) / (len(total_precision)+ 1e-6)
    total_recall = sum(total_recall) / (len(total_recall)+ 1e-6)
    total_f1 = sum(total_f1) / (len(total_f1)+ 1e-6)

    print(f'New metric | Precision {total_precision} Recall {total_recall} F1 {total_f1} MAP {total_map}')

    plt.close()
    plt.cla()

    return total_precision, total_recall, total_f1, total_map, recalls, precisions, correct, iou_small, small_iou_values, already_detect