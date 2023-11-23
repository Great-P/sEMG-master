import sys

sys.path.append("../..")

import os.path

from scipy.io import loadmat
from tqdm import tqdm
import h5py
from utils.Methods.methods import *


class Matloader:
    def __init__(self, loaddir, savedir):
        self.__loaddir = loaddir
        self.__savedir = savedir
        self.filename = loaddir.split("/")[-1][:-4]
        self.__loaddata()
        self.additional_loads = {}

    def __loaddata(self):
        try:
            self.__NinaData = loadmat(self.__loaddir)
        except:
            raise Exception("Load error,must be a mat file")
        else:
            print(self.__repr__())

    def getdata(self, data=None, key=None):
        if data is None:
            data = data if data else self.__NinaData
        if key is None:
            return data
        else:
            try:
                return data[key]
            except:
                raise Exception("You have a wrong key or dataset!Plz check")

    def reload(self, newdir=None):
        if newdir is not None:
            self.__loaddir = newdir
        self.__loaddata()

    def change_savedir(self, savedir):
        self.__savedir = savedir

    def save(self, data, savedir=None, suffix=None):
        # data = pd.DataFrame(data)
        tosavedir = savedir if savedir else self.__savedir
        if not os.path.exists(tosavedir):
            os.makedirs(tosavedir)
        filepath = tosavedir \
                   + "/" + self.filename + \
                   (("_" + suffix) if suffix else "") + ".h5"
        # print(filepath)
        # exit()
        # data.to_csv(filepath, index=False, header=False)
        with h5py.File(filepath, "w") as f:
            f.create_dataset("featureset", data=data, compression="gzip", compression_opts=5)

    def __repr__(self):
        reprstr = "=================================\n"
        reprstr += "Datafile:" + self.__loaddir + "\n"
        reprstr += "keys:" + str(self.__NinaData.keys()) + "\n"
        reprstr += "================================="
        return reprstr

    def __sizeof__(self):
        return len(self.__NinaData)

    def __add__(self, other):
        # print(other.__NinaData.keys())
        print(f"[.]Loading extractor of {other.filename} on extractor of {self.filename}")
        self.additional_loads[other.filename] = other.__NinaData
        self.additional_loads.update(other.additional_loads)
        if self.filename != "S0_1_E2_A1":
            print("[!]Rename the new data as subject S01")
            self.filename = "S0_1_E2_A1"
        return self


class Extractor(Matloader):
    def __init__(self, DataDir, SaveDir, windowsize=200, step=1):
        super().__init__(DataDir, SaveDir)
        self.__windowsize = windowsize
        self.__step = step
        self.__dir = DataDir

    def settings(self, windowsize, step):
        self.__windowsize = windowsize
        self.__step = step

    def extract(self, data=None, method=None):
        if method is None or method == "rms":
            method = [rms]
        if method == "TDDLF":
            method = [m0, m2, m4, PS, SE, USTD]
        assert type(method) == list, "Plz input a list or TDDLF"

        data = data if (data is not None) else self.getdata("emg")
        featuremap = []
        print("[.]Now feature is being extracted...")
        for eachmethod in method:
            feature_k = np.zeros([(data.shape[0] - self.__windowsize + 1) // self.__step
                                     , data.shape[1]])
            j = 0
            for i in tqdm(range(0, data.shape[0], self.__step)):
                if i + self.__windowsize > data.shape[0]:
                    break
                for eachchannel in range(data.shape[1]):
                    feature_k[j, eachchannel] = eachmethod(data[i:i + self.__windowsize, eachchannel])
                j += 1
            featuremap.append(feature_k)
        featuremap = np.transpose(np.array(featuremap), [1, 2, 0])
        # if featuremap.shape[2] == 1:
        #     featuremap = featuremap[:, :, 0].reshape([featuremap.shape[0], featuremap.shape[1]])
        print("[*]Extract complete!")
        return featuremap

    def split_stimulus(self, data=None, label=17, keep_rest=True):
        stimulus = self.getdata(data, "restimulus")
        stimulus = stimulus.reshape([stimulus.shape[0]])
        index_list = [i for i, x in enumerate(stimulus) if (label == x)]
        raw_glove = self.getdata(data, "glove")
        raw_emg = self.getdata(data, "emg")
        emg_slice = []
        glove_slice = []
        if keep_rest:
            head, tail = index_list[0], index_list[-1]
            emg_slice.extend(raw_emg[head:tail])
            glove_slice.extend(raw_glove[head:tail])
        else:
            for i in index_list:
                emg_slice.append(raw_emg[i])
                glove_slice.append(raw_glove[i])
        return np.array(emg_slice), np.array(glove_slice)

    def split_dataset(self, data, division=1, train_ratio=0.8, if_val=True):
        data = data[:data.shape[0] // division]
        # print(data.shape)
        rmslen = data.shape[0]
        trainlen = int(rmslen * train_ratio)
        traindata = data[:trainlen]
        if if_val:
            testlen = (rmslen - trainlen) // 2
            testdata = data[trainlen:trainlen + testlen]
            valdata = data[trainlen + testlen:]
            return traindata, testdata, valdata
        else:
            testlen = rmslen - trainlen
            testdata = data[trainlen:trainlen + testlen]
            return traindata, testdata, []

    def choose_save_actions(self, stimulus=None, savedir="../Data/featureset", with_additional_loads=True,
                            train_ratio=0.8, if_val=True, extract_method="rms", keep_rest=False):
        if stimulus is not None:
            assert type(stimulus) == list
        else:
            stimulus = [18, 19, 22, 25, 27, 37]

        data_list = [(self.filename, None)]

        total_glove_train = []
        total_glove_test = []
        total_glove_val = []
        total_emg_train = []
        total_emg_test = []
        total_emg_val = []
        if with_additional_loads:
            additional_list = [(k, v) for (k, v) in self.additional_loads.items()]
            data_list.extend(additional_list)
        for cur_name, cur_data in data_list:
            for i in stimulus:
                print(f"[.]Now splitting activity {i} on dataset {cur_name}")
                emg, glove = self.split_stimulus(cur_data, label=i, keep_rest=keep_rest)

                emg_train, emg_test, emg_val = self.split_dataset(emg, train_ratio=train_ratio, if_val=if_val)
                glove_train, glove_test, glove_val = self.split_dataset(glove, train_ratio=train_ratio, if_val=if_val)

                total_emg_train.extend(emg_train)
                total_emg_test.extend(emg_test)
                total_emg_val.extend(emg_val)

                total_glove_train.extend(glove_train)
                total_glove_test.extend(glove_test)
                total_glove_val.extend(glove_val)
            print("[*]Done!")

        total_emg_train = np.array(total_emg_train)
        total_emg_test = np.array(total_emg_test)
        total_emg_val = np.array(total_emg_val)

        feature_train = self.extract(total_emg_train, method=extract_method)
        feature_test = self.extract(total_emg_test, method=extract_method)

        total_glove_train = np.array(total_glove_train)[:-self.__windowsize + 1]
        total_glove_test = np.array(total_glove_test)[:-self.__windowsize + 1]

        if if_val:
            feature_val = self.extract(total_emg_val, method=extract_method)
            total_glove_val = np.array(total_glove_val)[:-self.__windowsize + 1]

        print("[.]Now writing files……")
        self.save(feature_train, savedir=savedir, suffix=f"{extract_method}_train_ff")
        self.save(feature_test, savedir=savedir, suffix=f"{extract_method}_test_ff")

        self.save(total_glove_train, savedir=savedir, suffix="glove_train_ff")
        self.save(total_glove_test, savedir=savedir, suffix="glove_test_ff")

        if if_val:
            self.save(feature_val, savedir=savedir, suffix=f"{extract_method}_val_ff")
            self.save(total_glove_val, savedir=savedir, suffix="glove_val_ff")
        print("[*]Written!")


if __name__ == "__main__":
    save_dir = "../Data/featureset"
    # data_dir_list = [
    #     "../Data/rawdata/S1_E2_A1.mat",
    #     "../Data/rawdata/S2_E2_A1.mat",
    #     "../Data/rawdata/S3_E2_A1.mat",
    #     "../Data/rawdata/S13_E2_A1.mat",
    #     "../Data/rawdata/S25_E2_A1.mat",
    #
    #     "../Data/rawdata/S11_E2_A1.mat",
    #     "../Data/rawdata/S14_E2_A1.mat",
    #     "../Data/rawdata/S18_E2_A1.mat",
    #     "../Data/rawdata/S19_E2_A1.mat",
    #     "../Data/rawdata/S22_E2_A1.mat",
    #
    #     "../Data/rawdata/S5_E2_A1.mat",
    #     "../Data/rawdata/S6_E2_A1.mat",
    #     "../Data/rawdata/S7_E2_A1.mat",
    #     "../Data/rawdata/S8_E2_A1.mat",
    #     "../Data/rawdata/S9_E2_A1.mat",
    #
    #     "../Data/rawdata/S10_E2_A1.mat",
    #     "../Data/rawdata/S12_E2_A1.mat",
    #     "../Data/rawdata/S15_E2_A1.mat",
    #     "../Data/rawdata/S16_E2_A1.mat",
    #     "../Data/rawdata/S17_E2_A1.mat",
    # ]
    # numlist = list(range(30))
    # numlist.remove(26-1)
    # numlist.remove(40-1)
    data_dir_list = [f"../Data/rawdata/S{x + 1}_E2_A1.mat" for x in range(40)]
    # print(data_dir_list)
    # exit(0)
    extractor_list = []
    for each_mat in data_dir_list:
        extractor_list.append(Extractor(each_mat, save_dir, windowsize=200, step=1))
    # extractor_fin = extractor_list[0]
    # for i in range(1, len(extractor_list)):
    #     extractor_fin += extractor_list[i]
    for each_extractor in extractor_list:
        each_extractor.choose_save_actions(train_ratio=0.007, if_val=False, stimulus=[18, 19, 22, 25, 27, 37],
                                      extract_method="rms", keep_rest=True)
    # extractor_fin.choose_save_actions(train_ratio=0.7, if_val=False, stimulus=[23, 24],
    #                                   extract_method="rms", keep_rest=True)
    # print(extractor_fin.filename)
    # print(extractor_fin.additional_loads)
    # splitemg18, glove18 = extractor2.split_stimulus(label=18)
    # print(splitemg18[33129])
    # print(glove18[33129])
